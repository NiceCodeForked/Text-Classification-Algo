"""
Credit from Lars Nieradzik https://lars76.github.io/2020/07/23/rsgd-in-pytorch.html
"""

import torch
from torch.optim.optimizer import Optimizer


@torch.jit.script
def lambda_x(x: torch.Tensor):
    return 2 / (1 - torch.sum(x ** 2, dim=-1, keepdim=True))


@torch.jit.script
def mobius_add(x: torch.Tensor, y: torch.Tensor):
    x2 = torch.sum(x ** 2, dim=-1, keepdim=True)
    y2 = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)

    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2

    return num / denom.clamp_min(1e-15)


@torch.jit.script
def expm_add(p: torch.Tensor, u: torch.Tensor):
    return p + u


@torch.jit.script
def expm_exp(p: torch.Tensor, u: torch.Tensor):
    # For exact exponential mapping
    norm = torch.sqrt(torch.sum(u ** 2, dim=-1, keepdim=True))
    return mobius_add(p, torch.tanh(0.5 * lambda_x(p) * norm) * u / norm.clamp_min(1e-15))


@torch.jit.script
def grad(p: torch.Tensor):
    p_sqnorm = torch.sum(p.data ** 2, dim=-1, keepdim=True)
    return p.grad.data * ((1 - p_sqnorm) ** 2 / 4).expand_as(p.grad.data)


class PoincareRiemannianSGD(Optimizer):

    def __init__(self, params, use_exp=False):
        super(PoincareRiemannianSGD, self).__init__(params, {})
        self.use_exp = use_exp

    def step(self, lr=0.3):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = grad(p)
                d_p.mul_(-lr)

                if self.use_exp:
                    p.data.copy_(expm_exp(p.data, d_p))
                else:
                    p.data.copy_(expm_add(p.data, d_p))