import torch.nn as nn


def count_parameters(module: nn.Module, trainable: bool=True):
    if trainable:
        num_parameters = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
    else:
        num_parameters = sum(p.numel() for p in module.parameters())
    return num_parameters