import torch
import numpy as np
import copy
from rnn import LeakyRNN
from analysis import per_direction_metrics

def position_loss(effector_xy, target_xy):
    return torch.mean(torch.sum(torch.abs(effector_xy - target_xy), dim=-1))

class BaseTrainer:

    def __init__(self, net: LeakyRNN, env, loss_fn, device='cpu'):
        eval_env = copy.deepcopy(self.env)
        self.net = net.to(device)
        self.env = env.to(device)
        self.eval_env = eval_env.to(device)
        self.env.effector.to(device)
        self.env.effector.muscle.to(device)
        self.env.effector.skeleton.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.weight_history = []

    @torch.no_grad()
    def snapshot_weights(self):
        # returns recurrent weights, technically we could also track input and output weights too if we want
        return self.net.W_rec.reshape(-1).detach().cpu().clone()

    @torch.no_grad()
    def inference(self, options=None, seed=0):
        options = {} if options is None else options
        obs, info = self.env.reset(seed=seed, options=options)
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
            # run inference
            u_t, h_t, z_t, y_t = self.net(x_t, h)
            # take the network's predicted action
            obs, reward, done, truncated, info = self.env.step(action=y_t)
            H.append(h_t)
            Y.append(y_t)
            FT.append(info["states"]["fingertip"])
            h = h_t

        return torch.stack(H), torch.stack(Y), torch.stack(FT), goal, direction_idx

    def train(self, num_steps=500, batch_size=32, eval_every=20, seed=0):
        losses = []
        metrics = []
        for i in range(num_steps):
            if i % eval_every == 0:
                metrics.append((i, self.checkpoint_behavior()))
            loss = self.train_step(batch_size, seed=seed+i)
            losses.append(loss)
        # guarantee the fully-trained network's behavior is captured, even if
        # num_steps isn't a multiple of eval_every (i inside the loop above
        # never reaches num_steps, so this is never a duplicate checkpoint)
        metrics.append((num_steps, self.checkpoint_behavior()))
        return losses, metrics

    def checkpoint_behavior(self):
        options = {'direction_idx': np.arange(self.env.n_targets)}
        H, Y, FT, goals, direction_idx = self.inference(options=options)
        FT = FT.permute(1, 0, 2)
        return per_direction_metrics(FT, goals, direction_idx)


class BPTTTrainer(BaseTrainer):
    def __init__(self, net, env, loss_fn, lr=1e-3, **kw):
        super().__init__(net, env, loss_fn, **kw)
        self.opt = torch.optim.SGD(net.parameters(), lr=lr)

    def train_step(self, batch_size, seed=0):
        h = self.net.init_hidden(batch_size, device=self.device)
        obs, info = self.env.reset(options={"batch_size": batch_size}, seed=seed)
        # keeping track of info from the effector/environment for loss
        xy, target = [], []
        done = False

        while not done:
            x_t = obs
            # run inference
            u_t, h_t, z_t, y_t = self.net(x_t, h)
            # take the network's predicted action
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
        self.weight_history.append(self.snapshot_weights())
        return loss.item()

class RFLOTrainer(BaseTrainer):

    def __init__(self, net, env, loss_fn, lr=1e-3, seed=0, **kw):
        super().__init__(net, env, loss_fn, **kw)
        # the paper's notebook uses three different variables but practically they
        # set all of them to the same learning rate so I decided to condense to 'lr'
        self.lr = lr
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        self.B = torch.randn(net.n_rec, net.n_out, generator=g, device=self.device) / net.n_out**0.5

    def train_step(self, batch_size, seed=0):
        net = self.net
        alpha = self.net.alpha

        h = net.init_hidden(batch_size, self.device)
        # recurrent trace
        p = torch.zeros(batch_size, net.n_rec, net.n_rec, device=self.device)
        # input trace
        q = torch.zeros(batch_size, net.n_rec, net.n_in, device=self.device)
        
        dW_in = torch.zeros_like(net.W_in)
        dW_rec = torch.zeros_like(net.W_rec)
        dW_out = torch.zeros_like(net.W_out)

        obs, info = self.env.reset(options={"batch_size": batch_size}, seed=seed)
        saved, y_leaves = [], []
        xy, target = [], []
        done = False

        while not done:
            x_t = obs
            h_prev = h
            # turn off grad
            with torch.no_grad():
                # u_t is pre-activation (tanh), h_t is post-activation, z_t is logit, y_t is post-sigmoid action
                u_t, h_t, z_t, y_t = net.step(x_t, h_prev)
                fp = net.f_prime(u_t) # postsynaptic sensitivity, shape (batch, n_rec)

                # here we update the eligibility traces
                # how much does hidden activation of unit j at time t-1
                # affect the hidden state of unit i at time t
                p = (1-alpha)*p + alpha * torch.einsum('ni,nj->nij', fp, h_prev)

                # how much does the external input j contribute to
                # hidden unit i's current state
                q = (1-alpha)*q + alpha * torch.einsum('ni,nj->nij', fp, x_t)
            # give y to the effeector as a differentiable leaf which allows us to
            # get dL/dy_t
            y_leaf = y_t.detach().requires_grad_(True)
            y_leaves.append(y_leaf)
            # save all the data at every timestep
            saved.append((h_t, y_t, p.clone(), q.clone()))
            obs, reward, done, truncated, info = self.env.step(action=y_leaf)
            xy.append(info["states"]["fingertip"])
            target.append(info["goal"])
            h = h_t

        xy = torch.stack(xy, dim=1)
        target = torch.stack(target, dim=1)
        loss = self.loss_fn(xy, target)
        # get dL/dy_t: how loss changes with respect to the action
        g_list = torch.autograd.grad(loss, y_leaves, allow_unused=True, materialize_grads=True)

        # get RFLO weight updates from the effector error
        for (h_t, y_t, p_t, q_t), g_t in zip(saved, g_list):
            sigmoid_prime = y_t * (1 - y_t)
            e_t = g_t * sigmoid_prime # dL/dz aka the error at the pre-activation readout
            c_t = e_t @ self.B.T # (batch, n_rec): random feedback projection of the readout error into the recurrent layer

            dW_out += torch.einsum("nk,ni->ki", e_t, h_t) # dL/dW_out, sum over the batch, outer product of e_t and h_t

            # RFLO replaces the true gradient with feedback × eligibility trace
            # contribution to synapse (a,b) = c_t[:,a] * p_t[:,a,b], summed over batch
            dW_rec += torch.einsum("na,nab->ab", c_t, p_t) 
            dW_in  += torch.einsum("na,nab->ab", c_t, q_t)

        with torch.no_grad():
            net.W_out -= self.lr * dW_out
            net.W_rec -= self.lr * dW_rec
            net.W_in -= self.lr * dW_in
        self.weight_history.append(self.snapshot_weights())
        return loss.item()
