import torch
from torch.utils.data import DataLoader

# Normalize our data samples around the mean and stdev, in order
# make the source distribution have mean 0 and unit variance.
MNIST_MEAN = 0.1307
MNIST_STDEV = 0.3081


def mnist_normalize(batch):
    return (batch - MNIST_MEAN) / MNIST_STDEV


def mnist_unnormalize(batch):
    return batch * MNIST_STDEV + MNIST_MEAN


def cycle(dataloader: DataLoader):
    # Loops through the dataloader class forever.
    while True:
        for data in dataloader:
            yield data


def generate_beta_array(num_timesteps: int, step1_beta: float = 0.001):
    # B = (T - t + 1)^-1
    beta = 1.0 / torch.linspace(num_timesteps, 2.0, num_timesteps)
    beta[0] = 2.0 * step1_beta
    return beta


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
