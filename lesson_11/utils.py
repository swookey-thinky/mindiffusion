"""Utility functions for working with the lesson."""

import torch
from torch.utils.data import DataLoader


def normalize_to_neg_one_to_one(img):
    """Converts tensors from (0,1) to (-1,1)."""
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """Converts tensors from (-1,1) to (0,1)."""
    return (t + 1) * 0.5


def extract(a, t, x_shape):
    """Helper function to extract the values from a up until time t.

    This function is used to extract ranges of different constants up to the
    given time. The timestep  is a batched timestep of shape (B,) and dtype=torch.int32.
    This will gather the values of a at indices (timesteps) t - shape (B,) - and then
    reshape the (B,) output to match x_shape, appending dimensions of length 1. So if
    x_shape has shape (B,C,H,W), then the output will be of shape (B,1,1,1).
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """Linear beta schedule, proposed in original ddpm paper."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cycle(dataloader: DataLoader):
    """Cycles through the dataloader class forever.

    Useful for when you want to cycle through a DataLoader for
    a finite number of timesteps.
    """
    while True:
        for data in dataloader:
            yield data


def freeze(model: torch.nn.Module):
    """Freeze the parameters of a model."""
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model
