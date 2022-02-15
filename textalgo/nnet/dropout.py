import torch
import torch.nn as nn


class AutoDropout(nn.Module):
    """
    References
    ----------
    1. https://github.com/dhuynh95/AutoDropout
    """
    def __init__(self, p=0., requires_grad=False):
        super(AutoDropout, self).__init__()
        p = 1 - p
        p = torch.tensor(p)
        inverse_sigmoid = lambda p: torch.log(p/(1-p))
        gamma = inverse_sigmoid(p)

        if requires_grad:
            gamma = nn.Parameter(gamma)
            self.register_parameter("gamma", gamma)
        else:
            self.register_buffer("gamma", gamma)

    def forward(self, x):
        p = torch.sigmoid(self.gamma)
        ps = p.expand(x.shape[1:])
        m = torch.distributions.Bernoulli(ps).sample((1, )).squeeze(0)
        m  = ps + (m - ps).detach()
        z = x * m
        return z

    def extra_repr(self):
        return 'p={}'.format(torch.sigmoid(self.gamma))


class SpatialDropout(nn.Module):
    """
    References
    ----------
    1. https://arxiv.org/pdf/1411.4280v3.pdf
    2. https://blog.csdn.net/weixin_43896398/article/details/84762943
    """
    pass





class GaussianDropout(nn.Module):
    """
    Parameters
    ----------
    p: float
        Determines the standard deviation of the gaussian noise
        where sigma = p/(1-p).

    References
    ----------
    1. https://www.kaggle.com/cepheidq/gaussian-dropout-for-pytorch/notebook
    2. https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    """
    def __init__(self, p):
        super(GaussianDropout).__init__()
        assert 0 <= p < 1
        self.t_mean = torch.ones((0,))
        self.shape = ()
        self.p = p
        self.t_std = self.compute_std()

    def compute_std(self):
        return self.p / (1 - self.p)

    def forward(self, t_hidden):
        if self.training and self.p > 0.:
            if self.t_mean.shape != t_hidden.shape:
                self.t_mean = torch.ones_like(
                    input=t_hidden, 
                    dtype=t_hidden.dtype, 
                    device=t_hidden.device
                )
            elif self.t_mean.device != t_hidden.device:
                self.t_mean = self.t_mean.to(
                    device=t_hidden.device, 
                    dtype=t_hidden.dtype
                )
            t_gaussian_noise = torch.normal(self.t_mean, self.t_std)
            t_hidden = t_hidden.mul(t_gaussian_noise)
        return t_hidden