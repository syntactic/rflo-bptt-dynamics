import torch
import torch.nn.functional as F
from torch import nn


class LeakyRNN(nn.Module):
    @staticmethod
    def f(u):
        return torch.tanh(u)

    def f_prime(self, u_t=None):
        # tanh'(u) = 1 - tanh(u)^2, and step() already computed tanh(u) as
        # self._last_act. Reusing it gives the exact derivative without a second
        # transcendental pass; fall back to recomputing when called standalone.
        if getattr(self, "_last_act", None) is not None:
            return 1.0 - self._last_act * self._last_act
        return 1.0 - torch.tanh(u_t) ** 2

    def __init__(self, n_in, n_rec, n_out, tau=10.0):
        super().__init__()
        self.n_in, self.n_rec, self.n_out, self.tau = n_in, n_rec, n_out, tau
        self.alpha = 1.0 / tau

        self.W_in = nn.Parameter(torch.empty(n_rec, n_in))
        self.W_rec = nn.Parameter(torch.empty(n_rec, n_rec))
        self.W_out = nn.Parameter(torch.empty(n_out, n_rec))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # for the below the input weight initalization might be better off symmetric around 0 instead of
            # how the original notebook had it which suits the toy task there
            self.W_in.uniform_(-0.1, 0)  # constrains weights to [-0.1, 0]
            # self.W_in.uniform_(-0.1, 0.1) # constraints weights to [-0.1, 0.1]
            # self.W_rec = nn.Parameter(1.5 * torch.randn(self.n_rec, self.n_rec) / (self.n_rec ** 0.5))
            self.W_rec.normal_(0.0, 1.0).mul_(1.5 / self.n_rec**0.5)
            # self.W_out = nn.Parameter(0.1 * (2 * torch.rand(self.n_out, self.n_rec) - 1) / (self.n_rec ** 0.5))
            self.W_out.uniform_(-0.1, 0.1).div_(self.n_rec**0.5)

    def init_hidden(self, batch_size, device=None):

        return torch.zeros(batch_size, self.n_rec, device=device)

    def step(self, x_t, h_prev):

        # pre-activations which are needed by RFLO. F.linear avoids the explicit
        # .T view that x @ W.T builds every step.
        u_t = F.linear(h_prev, self.W_rec) + F.linear(x_t, self.W_in)

        # cache tanh(u_t) so RFLO's f_prime can reuse it instead of recomputing
        self._last_act = torch.tanh(u_t)

        # leaky integration
        h_t = (1 - self.alpha) * h_prev + self.alpha * self._last_act

        # pre-sigmoid readout, needed by RFLO
        z_t = F.linear(h_t, self.W_out)

        # sigmoid because I think MotorNet is going to expect values between 0 and 1
        y_t = torch.sigmoid(z_t)

        return u_t, h_t, z_t, y_t

    def forward(self, x_t, h_prev):
        return self.step(x_t, h_prev)
