"""
Credit from Lars Nieradzik https://lars76.github.io/2020/07/23/rsgd-in-pytorch.html
"""

import torch
from torch.optim.optimizer import Optimizer


def expm(p : torch.Tensor, u : torch.Tensor):
    ldv = lorentzian_inner_product(u, u, keepdim=True).clamp_(min=1e-15).sqrt_()
    return torch.cosh(ldv) * p + torch.sinh(ldv) * u / ldv


def lorentzian_inner_product(u : torch.Tensor, v : torch.Tensor, keepdim=False):
    uv = u * v
    uv.narrow(-1, 0, 1).mul_(-1)
    return torch.sum(uv, dim=-1, keepdim=keepdim)


@torch.jit.script
def grad(p : torch.Tensor):
    d_p = p.grad
    d_p.narrow(-1, 0, 1).mul_(-1)
    return d_p


def proj(p : torch.Tensor, d_p : torch.Tensor):
    return d_p + lorentzian_inner_product(p.data, d_p, keepdim=True) * p.data


class HyperboloidRiemannianSGD(Optimizer):

    def __init__(self, params):
        super(HyperboloidRiemannianSGD, self).__init__(params, {})

    def step(self, lr=0.3):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = grad(p)
                d_p = proj(p, d_p)
                d_p.mul_(-lr)

                p.data.copy_(expm(p.data, d_p))