import copy

import numpy as np
import torch

from analysis import per_direction_metrics
from rnn import LeakyRNN


def position_loss(effector_xy, target_xy):
    return torch.mean(torch.sum(torch.abs(effector_xy - target_xy), dim=-1))


def make_terminal_weighted_loss(w_term=0.5, k=10):
    """Blend trajectory-averaged and last-k-step L1 position loss:
    (1 - w_term) * mean_t + w_term * mean_last_k. w_term=0 is position_loss.

    The plain loss weights the endpoint only 1/T, so both rules park terminal
    error at the success threshold. This adds endpoint pressure to test whether
    the point-mass RFLO gap is objective-driven, not a capacity limit. Both
    trainers call loss_fn(xy, target), so the comparison stays matched.
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
        # MotorNet applies sensory noise unconditionally during get_proprioception/get_vision,
        # ignoring deterministic=True. Zero them here so eval_env always yields a clean readout.
        eval_env.proprioception_noise = [0.0]
        eval_env.vision_noise = [0.0]
        self.eval_env = eval_env.to(device)
        self.eval_env.effector.to(device)
        self.eval_env.effector.muscle.to(device)
        self.eval_env.effector.skeleton.to(device)

        self.loss_fn = loss_fn
        self.device = device
        # Per-step weight snapshots feed the weight-space (Q4) analysis but dominate
        # artifact size (one N x N matrix per step). Skip them for runs that only
        # need behavior/convergence, e.g. learning-rate exploration sweeps.
        self.record_weights = record_weights
        self.weight_history = []

    @torch.no_grad()
    def snapshot_weights(self):
        return self.net.W_rec.reshape(-1).detach().cpu().clone()

    @torch.no_grad()
    def inference(self, options=None, seed=0):
        # Deterministic readout: eval_env has sensory noise zeroed and suppresses action noise
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
        # Ensure final state is recorded even if num_steps is not a multiple of eval_every
        metrics.append((num_steps, self.checkpoint_behavior()))
        return losses, metrics

    def checkpoint_behavior(self):
        options = {"direction_idx": np.arange(self.env.n_targets)}
        H, Y, FT, goals, direction_idx = self.inference(options=options)
        FT = FT.permute(1, 0, 2)
        return per_direction_metrics(FT, goals, direction_idx)


class BPTTTrainer(BaseTrainer):
    def __init__(self, net, env, loss_fn, lr=1e-3, **kw):
        super().__init__(net, env, loss_fn, **kw)
        self.opt = torch.optim.SGD(net.parameters(), lr=lr)

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
        self.opt.step()
        if self.record_weights:
            self.weight_history.append(self.snapshot_weights())
        return loss.item()


class RFLOTrainer(BaseTrainer):
    def __init__(self, net, env, loss_fn, lr=1e-3, b_seed=0, max_steps=100, **kw):
        super().__init__(net, env, loss_fn, **kw)
        # the paper's notebook uses three different variables but practically they
        # set all of them to the same learning rate so I decided to condense to 'lr'
        self.lr = lr
        self.b_seed = b_seed
        self.B = None
        if self.b_seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.b_seed)
            self.B = (
                torch.randn(net.n_rec, net.n_out, generator=g, device=self.device)
                / net.n_out**0.5
            )

        # Per-timestep traces and states are written into these buffers in place.
        # Cloning p (shape (batch, n_rec, n_rec)) into a fresh list entry every
        # step churned heap memory; reusing one allocation avoids that GC pressure.
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
        # recurrent trace
        p = torch.zeros(batch_size, net.n_rec, net.n_rec, device=self.device)
        # input trace
        q = torch.zeros(batch_size, net.n_rec, net.n_in, device=self.device)

        obs, info = self.env.reset(options={"batch_size": batch_size})
        y_leaves = []
        xy, target = [], []
        done = False
        t = 0

        while not done:
            x_t = obs
            h_prev = h
            # turn off grad
            with torch.no_grad():
                # u_t is pre-activation (tanh), h_t is post-activation, z_t is logit, y_t is post-sigmoid action
                u_t, h_t, z_t, y_t = net.step(x_t, h_prev)
                fp = net.f_prime(u_t)  # postsynaptic sensitivity, shape (batch, n_rec)

                # here we update the eligibility traces
                # how much does hidden activation of unit j at time t-1
                # affect the hidden state of unit i at time t.
                # unsqueeze broadcasting forms the same outer product as
                # einsum("ni,nj->nij", ...) without the string-parse dispatch.
                p = (1 - alpha) * p + alpha * (fp.unsqueeze(2) * h_prev.unsqueeze(1))

                # how much does the external input j contribute to
                # hidden unit i's current state
                q = (1 - alpha) * q + alpha * (fp.unsqueeze(2) * x_t.unsqueeze(1))

                # in-place writes into the pre-allocated buffers, no per-step clone
                self.p_buf[t].copy_(p)
                self.q_buf[t].copy_(q)
                self.h_buf[t].copy_(h_t)
                self.y_buf[t].copy_(y_t)
            # give y to the effector as a differentiable leaf which allows us to
            # get dL/dy_t
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
        # get dL/dy_t: how loss changes with respect to the action
        g_list = torch.autograd.grad(
            loss, y_leaves, allow_unused=True, materialize_grads=True
        )

        # Vectorize the RFLO accumulation over the whole episode instead of a
        # Python loop over timesteps. Slice buffers to the T steps actually run.
        G = torch.stack(g_list)  # (T, batch, n_out)
        Y = self.y_buf[:T]
        H = self.h_buf[:T]
        P = self.p_buf[:T]
        Q = self.q_buf[:T]

        E = G * (Y * (1.0 - Y))  # dL/dz, error at the pre-activation readout
        if self.B is None:
            C = E @ net.W_out.detach()  # detach from autograd
        else:
            C = E @ self.B.T  # random feedback projection of the readout error

        # dL/dW_out: outer product of e_t and h_t, summed over batch and time.
        # Flattening (T, batch) into one axis turns the sum into a single matmul.
        dW_out = E.reshape(-1, net.n_out).T @ H.reshape(-1, net.n_rec)

        # RFLO replaces the true gradient with feedback x eligibility trace.
        # C.unsqueeze(3) broadcasts the (a) feedback signal across the (b) trace
        # axis; summing over time and batch yields the (a, b) weight updates.
        dW_rec = (C.unsqueeze(3) * P).sum(dim=(0, 1))
        dW_in = (C.unsqueeze(3) * Q).sum(dim=(0, 1))

        with torch.no_grad():
            net.W_out -= self.lr * dW_out
            net.W_rec -= self.lr * dW_rec
            net.W_in -= self.lr * dW_in
        if self.record_weights:
            self.weight_history.append(self.snapshot_weights())
        return loss.item()
