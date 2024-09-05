"""Utility functions for working with the lesson."""

from functools import partial
import importlib
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Any, Dict, Type, TypeVar
import yaml


class DotConfig:
    """Helper class to allow "." access to dictionaries."""

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k) -> Any:
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v

    def __getitem__(self, k) -> Any:
        return self.__getattr__(k)

    def __contains__(self, k) -> bool:
        try:
            v = self._cfg[k]
            return True
        except KeyError:
            return False

    def to_dict(self):
        return self._cfg


def load_yaml(yaml_path: str) -> DotConfig:
    """Loads a YAML configuration file."""
    with open(yaml_path, "r") as fp:
        return DotConfig(yaml.load(fp, yaml.CLoader))


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


def linear_beta_schedule(timesteps: int):
    """Linear beta schedule, proposed in original ddpm paper."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, max_beta: float = 0.999):
    """Cosine beta schedule, proposed in Improved DDPM."""
    alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.from_numpy(np.array(betas))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """Computes the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.

    Original implementation:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/losses.py#L12
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    Args:
        x: Tensor batch of target images. It is assumed that this was uint8 values,
            rescaled to the range [-1, 1].
        means: Tensor batch of the Gaussian mean.
        log_scales: Tensor batch of the Gaussian log stddev.

    Returns:
        A tensor batch of log probabilities (in nats), of the same shape as x.
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs


def cycle(dataloader: DataLoader):
    """Cycles through the dataloader class forever.

    Useful for when you want to cycle through a DataLoader for
    a finite number of timesteps.
    """
    while True:
        for data in dataloader:
            yield data


T = TypeVar("T", bound=torch.nn.Module)


def freeze(model: T) -> T:
    """Freeze the parameters of a model."""
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def instantiate_from_config(config, use_config_struct: bool = False) -> Any:
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if use_config_struct:
        return get_obj_from_str(config["target"])(config["params"])
    else:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))


def instantiate_partial_from_config(config) -> Any:
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return partial(get_obj_from_str(config["target"]), **config.get("params", dict()))


def type_from_config(config) -> Type:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])


def kwargs_from_config(config) -> Dict:
    if not "params" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return config.get("params", dict())


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
