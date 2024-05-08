"""Sampling procedure (reverse process) from DDIM."""

import torch
from tqdm import tqdm
from typing import Optional

from utils import (
    linear_beta_schedule,
    unnormalize_to_zero_to_one,
)


@torch.inference_mode()
def sample_ddim(
    epsilon_network: torch.nn.Module,
    image_size: int,
    num_channels: int,
    num_timesteps: int,
    context_dimension: int,
    batch_size: int = 16,
    x_t: Optional[torch.Tensor] = None,
    num_sampling_timesteps: Optional[int] = None,
    y: Optional[torch.Tensor] = None,
):
    """Performs determistic sampling according to DDIM.

    Implements equation 12 of the DDIM paper, with $sigma_t = 0$ for
    deterministic sampling.

    Args:
        epsilon_network: The score network from DDPM, condition on text embeddings
            (Lesson 5e).
        image_size: The spatial size of the output images.
        num_channels: Number of channels for the output images (1 for MNIST).
        num_timesteps: The number of forward process timesteps (1000 for DDPM).
        context_dimension: The context dimension for text embeddings.
        batch_size: The number of samples to generate.
        x_t: An initial starting set of latents, at timestep t
        num_sampling_timesteps: The number of steps to use in sampling (reverse process).
            Should be less than num_timesteps! DDIM paper used 50.
        y: Optional conditional context.

    Returns:
        Tensor batch of denoised samples.
    """
    # The output shape of the data.
    shape = (batch_size, num_channels, image_size, image_size)

    # Use the device that the current model is on.
    # Assumes all of the parameters are on the same device
    device = next(epsilon_network.parameters()).device

    # Line 1, initial image is pure noise
    if x_t is None:
        x_t = torch.randn(shape, device=device)

    # Unconditional context, if the context does not exist.
    if y is None:
        y = torch.zeros(
            (batch_size, context_dimension),
            dtype=torch.float32,
            device=device,
        )

    # Line 4, constants that we will use below.
    # Fixed linear schedule for Beta
    num_timesteps = num_timesteps
    sampling_timesteps = (
        50 if num_sampling_timesteps is None else num_sampling_timesteps
    )
    skip = num_timesteps // sampling_timesteps
    seq = range(0, num_timesteps, skip)

    betas = linear_beta_schedule(num_timesteps).to(torch.float32).to(device)
    x_0 = _generalized_steps(x_t, seq, epsilon_network, betas, y)
    x_0 = x_0[-1]

    # Unnormalize the sample images from (-1, 1) back to (0,1)
    return unnormalize_to_zero_to_one(x_0)


def _compute_alpha(beta, t):
    """Computes alpha (1 - beta) at timestep [0,t]."""
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def _generalized_steps(x, seq, epsilon_network, b, y, **kwargs):
    """Equation 12 from the DDIM paper, over all timesteps."""
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        for i, j in tqdm(
            zip(reversed(seq), reversed(seq_next)),
            desc="sampling loop time step",
            total=len(seq),
        ):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            alpha_t = _compute_alpha(b, t.long())
            alpha_t_next = _compute_alpha(b, next_t.long())
            xt = xs[-1].to("cuda")

            # Calculate the epsilon_theta prediction at time t
            # using the epsilon network.
            epsilon_t = epsilon_network(xt, t, y)

            # The "predicted x0" from Eq. 12
            x0_t = (xt - epsilon_t * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

            # This is sigma_t for the DDPM case. Note here we set "eta" to zero,
            # for deterministic sampling, so c1 = 0 for all t here.
            c1 = (
                kwargs.get("eta", 0)
                * (
                    (1 - alpha_t / alpha_t_next) * (1 - alpha_t_next) / (1 - alpha_t)
                ).sqrt()
            )

            # This is the constant in front of epsilon_theta, in the
            # "direction pointing to x_t" term.
            c2 = ((1 - alpha_t_next) - c1**2).sqrt()

            # The full equation 12. Note c1 is zero, so the random term drops away.
            xt_next = (
                alpha_t_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * epsilon_t
            )
            xs.append(xt_next.to("cpu"))

    return xs
