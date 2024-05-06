"""Utility functions for working with the lesson."""

from torch.utils.data import DataLoader


def cycle(dataloader: DataLoader):
    """Cycles through the dataloader class forever.

    Useful for when you want to cycle through a DataLoader for
    a finite number of timesteps.
    """
    while True:
        for data in dataloader:
            yield data


class EMAHelper(object):
    """Helper class for calculating the exponential moving average.

    Calculates the exponential moving average (EMA) of a model at
    each update. This class is for illustrative purposes only. There
    is a much more advanced EMA package at https://github.com/lucidrains/ema-pytorch
    which has a lot more functionality.
    """

    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.shadow[name].data

    def copy_params(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)
