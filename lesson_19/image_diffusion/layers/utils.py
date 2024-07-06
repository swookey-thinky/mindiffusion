from abc import abstractmethod
import torch


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return torch.nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return torch.nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return torch.nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return torch.nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return torch.nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return torch.nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class ContextBlock(torch.nn.Module):
    """Basic block which accepts a context conditioning."""

    @abstractmethod
    def forward(self, x, context):
        """Apply the module to `x` given `context` conditioning."""
