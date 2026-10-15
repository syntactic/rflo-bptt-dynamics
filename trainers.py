import copy

import numpy as np
import torch

from analysis import SUCCESS_FRAC, per_direction_metrics
from rnn import LeakyRNN


def make_feedback_matrix(n_rec, n_out, seed=0, device="cpu"):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    B = torch.randn(n_rec, n_out, generator=g, device=device) / n_out**0.5
    return B


def position_loss(effector_xy, target_xy):
    return torch.mean(torch.sum(torch.abs(effector_xy - target_xy), dim=-1))


def make_terminal_weighted_loss(w_term=0.5, k=10):
    """L1 position loss, (1 - w_term) * mean over time + w_term * mean over the
    last k steps. w_term=0 is position_loss.

    The trajectory loss weights the endpoint by only 1/T, so terminal error can
    settle at the success threshold. This adds pressure on the endpoint.
    """

    def loss_fn(effector_xy, target_xy):
        per_step = torch.sum(torch.abs(effector_xy - target_xy), dim=-1)  # (batch, T)
        return (1.0 - w_term) * per_step.mean() + w_term * per_step[:, -k:].mean()

    return loss_fn


class BaseTrainer:
    def __init__(self, net: LeakyRNN, env, loss_fn, device="cpu", record_weights=True):
        self.net = net.to(device)
        self.env = env.to(device)
        self.env.effector.to(device)
        self.env.effector.muscle.to(device)
        self.env.effector.skeleton.to(device)

        eval_env = copy.deepcopy(env)
        # MotorNet adds proprioceptive and visual noise even with deterministic=True,
        # so zero both channels to keep eval rollouts clean.
        eval_env.proprioception_noise = [0.0]
        eval_env.vision_noise = [0.0]
        self.eval_env = eval_env.to(device)
        self.eval_env.effector.to(device)
        self.eval_env.effector.muscle.to(device)
        self.eval_env.effector.skeleton.to(device)

        self.loss_fn = loss_fn
        self.device = device
        # Per-step W_rec snapshots are needed for weight-space analysis but dominate
        # artifact size. Turn off for behavior-only runs such as lr sweeps.
        self.record_weights = record_weights
        self.weight_history = []

    @torch.no_grad()
    def snapshot_weights(self):
        return self.net.W_rec.reshape(-1).detach().cpu().clone()

    @torch.no_grad()
    def inference(self, options=None, seed=0):
        # Sensory noise is zeroed on eval_env; deterministic=True drops action and obs noise
        options = {} if options is None else dict(options)
        options.setdefault("deterministic", True)
        obs, info = self.eval_env.reset(seed=seed, options=options)
        batch_size = obs.shape[0]
        h = self.net.init_hidden(batch_size, device=self.device)
        goal = info["goal"].detach().clone()
        direction_idx = info.get("direction_idx", None)
        if direction_idx is not None:
            direction_idx = direction_idx.detach().clone()
        H, Y, FT = [], [], []
        done = False

        while not done:
            x_t = obs
            u_t, h_t, z_t, y_t = self.net(x_t, h)
            obs, reward, done, truncated, info = self.eval_env.step(
                action=y_t, deterministic=True
            )
            H.append(h_t)
            Y.append(y_t)
            FT.append(info["states"]["fingertip"])
            h = h_t

        return torch.stack(H), torch.stack(Y), torch.stack(FT), goal, direction_idx

    def train(self, num_steps=500, batch_size=32, eval_every=20, seed=0):
        losses = []
        metrics = []
        self.env.reset(seed=seed, options={"batch_size": batch_size})
        for i in range(num_steps):
            if i % eval_every == 0:
                metrics.append((i, self.checkpoint_behavior()))
            loss = self.train_step(batch_size)
            losses.append(loss)
        # Record the final state even when num_steps isn't a multiple of eval_every
        metrics.append((num_steps, self.checkpoint_behavior()))
        return losses, metrics

    def checkpoint_behavior(self):
        options = {"direction_idx": np.arange(self.env.n_targets)}
        H, Y, FT, goals, direction_idx = self.inference(options=options)
        FT = FT.permute(1, 0, 2)
        # Scale the success radius with reach. A fixed 0.05 would be half the arm's
        # reach and a tenth of the point mass's.
        radius = SUCCESS_FRAC * self.env.reaching_distance
        return per_direction_metrics(FT, goals, direction_idx, success_radius=radius)


class BPTTTrainer(BaseTrainer):
    def __init__(self, net, env, loss_fn, lr=1e-3, grad_basis=None, **kw):
        super().__init__(net, env, loss_fn, **kw)
        self.opt = torch.optim.SGD(net.parameters(), lr=lr)
        self.P = None
        # Optional projection of the W_rec/W_in gradients onto span(grad_basis),
        # restricting BPTT to the update directions RFLO has with that B.
        if grad_basis is not None:
            q, _ = torch.linalg.qr(grad_basis)
            self.P = q @ q.T

    @torch.no_grad()
    def _project_grads(self):
        for W in (self.net.W_rec, self.net.W_in):
            projected_W = self.P @ W.grad
            W.grad.copy_(projected_W)

    def train_step(self, batch_size):
        h = self.net.init_hidden(batch_size, device=self.device)
        obs, info = self.env.reset(options={"batch_size": batch_size})
        xy, target = [], []
        done = False

        while not done:
            x_t = obs
            u_t, h_t, z_t, y_t = self.net(x_t, h)
            obs, reward, done, truncated, info = self.env.step(action=y_t)
            xy.append(info["states"]["fingertip"])
            target.append(info["goal"])
            h = h_t
        xy = torch.stack(xy, dim=1)
        target = torch.stack(target, dim=1)
        loss = self.loss_fn(xy, target)
        self.opt.zero_grad()
        loss.backward()
        if self.P is not None:
            self._project_grads()
        self.opt.step()
        if self.record_weights:
            self.weight_history.append(self.snapshot_weights())
        return loss.item()


class RFLOTrainer(BaseTrainer):
    def __init__(
        self,
        net,
        env,
        loss_fn,
        lr=1e-3,
        b_seed=0,
        b_norm_match=False,
        max_steps=100,
        **kw,
    ):
        super().__init__(net, env, loss_fn, **kw)
        # The reference notebook has a learning rate per weight matrix but sets
        # them all equal, so one lr here.
        self.lr = lr
        self.b_seed = b_seed
        # With aligned feedback (b_seed=None), rescale W_out to random B's expected norm
        self.b_norm_match = b_norm_match
        self.B = None
        if self.b_seed is not None:
            self.B = make_feedback_matrix(
                net.n_rec, net.n_out, seed=self.b_seed, device=self.device
            )

        # Per-step traces and states are written into preallocated buffers.
        # Cloning p (batch, n_rec, n_rec) into a list every step churned memory.
        self.max_steps = max_steps
        self._batch_size = None
        self._alloc_buffers(32, max_steps)

    def _alloc_buffers(self, batch_size, max_steps):
        net = self.net
        self._batch_size = batch_size
        self.p_buf = torch.zeros(
            max_steps, batch_size, net.n_rec, net.n_rec, device=self.device
        )
        self.q_buf = torch.zeros(
            max_steps, batch_size, net.n_rec, net.n_in, device=self.device
        )
        self.h_buf = torch.zeros(max_steps, batch_size, net.n_rec, device=self.device)
        self.y_buf = torch.zeros(max_steps, batch_size, net.n_out, device=self.device)

    def train_step(self, batch_size):
        if batch_size != self._batch_size:
            self._alloc_buffers(batch_size, self.max_steps)

        net = self.net
        alpha = self.net.alpha

        h = net.init_hidden(batch_size, self.device)
        # Eligibility traces for W_rec (p) and W_in (q)
        p = torch.zeros(batch_size, net.n_rec, net.n_rec, device=self.device)
        q = torch.zeros(batch_size, net.n_rec, net.n_in, device=self.device)

        obs, info = self.env.reset(options={"batch_size": batch_size})
        y_leaves = []
        xy, target = [], []
        done = False
        t = 0

        while not done:
            x_t = obs
            h_prev = h
            with torch.no_grad():
                # u_t: pre-tanh, h_t: hidden state, z_t: readout logit, y_t: sigmoid action
                u_t, h_t, z_t, y_t = net.step(x_t, h_prev)
                fp = net.f_prime(u_t)  # (batch, n_rec)

                # p[i, j]: leaky trace of how h_{t-1}[j] drives h_t[i].
                # Broadcasting gives the outer product without einsum's overhead.
                p = (1 - alpha) * p + alpha * (fp.unsqueeze(2) * h_prev.unsqueeze(1))
                # q[i, j]: the same for input x_t[j]
                q = (1 - alpha) * q + alpha * (fp.unsqueeze(2) * x_t.unsqueeze(1))

                self.p_buf[t].copy_(p)
                self.q_buf[t].copy_(q)
                self.h_buf[t].copy_(h_t)
                self.y_buf[t].copy_(y_t)
            # A detached leaf lets autograd return dL/dy_t through the effector
            # without reaching back into the network.
            y_leaf = y_t.detach().requires_grad_(True)
            y_leaves.append(y_leaf)
            obs, reward, done, truncated, info = self.env.step(action=y_leaf)
            xy.append(info["states"]["fingertip"])
            target.append(info["goal"])
            h = h_t
            t += 1

        T = t
        xy = torch.stack(xy, dim=1)
        target = torch.stack(target, dim=1)
        loss = self.loss_fn(xy, target)
        g_list = torch.autograd.grad(
            loss, y_leaves, allow_unused=True, materialize_grads=True
        )

        # Accumulate over the whole episode at once, using only the T steps run.
        G = torch.stack(g_list)  # (T, batch, n_out)
        Y = self.y_buf[:T]
        H = self.h_buf[:T]
        P = self.p_buf[:T]
        Q = self.q_buf[:T]

        E = G * (Y * (1.0 - Y))  # dL/dz, error at the readout logit
        if self.B is None:
            # Aligned feedback through the live readout
            W_fb = net.W_out.detach()
            if self.b_norm_match:
                # Random B has expected Frobenius norm sqrt(n_rec). W_out starts ~50x
                # smaller, which would also shrink the effective lr on W_rec/W_in.
                # Matching the norm leaves only the credit direction different.
                W_fb = W_fb * (net.n_rec**0.5 / W_fb.norm())
            C = E @ W_fb
        else:
            C = E @ self.B.T

        # dL/dW_out = sum over time and batch of e_t h_t^T, as one matmul
        dW_out = E.reshape(-1, net.n_out).T @ H.reshape(-1, net.n_rec)

        # RFLO update: feedback signal times eligibility trace, summed over time and batch
        dW_rec = (C.unsqueeze(3) * P).sum(dim=(0, 1))
        dW_in = (C.unsqueeze(3) * Q).sum(dim=(0, 1))

        with torch.no_grad():
            net.W_out -= self.lr * dW_out
            net.W_rec -= self.lr * dW_rec
            net.W_in -= self.lr * dW_in
        if self.record_weights:
            self.weight_history.append(self.snapshot_weights())
        return loss.item()
