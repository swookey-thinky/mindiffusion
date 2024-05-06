"""Forward and reverse diffusion processes.

This package implements the forward and reverse diffusion processes from
the paper "Denoising Diffusion Probabilistic Models"
(https://arxiv.org/abs/2006.11239), with the context embedding architecture
introduced in High-Resolution Image Synthesis with Latent Diffusion Models
(https://arxiv.org/abs/2112.10752).
"""

from einops import reduce
import torch
from tqdm import tqdm
from typing import Type

from utils import (
    extract,
    linear_beta_schedule,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)


class GaussianDiffusion_ConditionalDDPM(torch.nn.Module):
    """Core DDPM algorithm.

    Defines the core DDPM diffusion algorithm for training and sampling. This implements
    Algorithms 1 and 2 from the DDPM paper.
    """

    def __init__(self, unet_type: Type):
        super().__init__()
        self._unet = unet_type()
        self._num_timesteps = 1000

    def algorithm1_train_on_batch(self, image, context):
        """
        This function defines the gradient descent step from Algorithm 1,
        namely lines 3-5. The optimizer step and batch loop is handled outside
        of this function, to take advantage of training utilities.
        """
        B, _, H, W = image.shape
        device = image.device

        # The images are normalized into the range (-1, 1),
        # from Section 3.3:
        # "We assume that image data consists of integers in {0, 1, . . . , 255} scaled linearly to [−1, 1]."
        x0 = normalize_to_neg_one_to_one(image)

        # Line 3, calculate the random timesteps for the training batch.
        t = torch.randint(0, self._num_timesteps, (B,), device=device).long()

        # Line 4, sample from a Gaussian with mean 0 and unit variance.
        epsilon = torch.randn_like(x0)

        # Line 5, calculate the parameters that we need to pass into
        # eps_theta (which is our unet).

        # Fixed linear schedule for Beta
        betas = linear_beta_schedule(self._num_timesteps).to(torch.float32).to(device)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # The first parameter is the square root of the cumululative product
        # of alpha at time t, times x0.
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        # Grab only the values for the timesteps in the batch.
        sqrt_alphas_cumprod_x0 = extract(sqrt_alphas_cumprod, t, x0.shape) * x0

        # The second parameter is the square root of 1 minus the cumulative product
        # of alpha at time t, times epsilon.
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # Grab only the values for the timesteps in the batch.
        sqrt_1_minus_alphas_cumprod_eps = (
            extract(sqrt_one_minus_alphas_cumprod, t, x0.shape) * epsilon
        )

        # Add the context to the unet forward pass. In order to maintain our
        # unconditional generation performance, we will drop the context
        # 25% of the time. A NULL conditioning consists of all zeros.
        if torch.rand(()) <= 0.25:
            context = torch.zeros_like(context)

        # Line 5, predict eps_theta given t. Add the two parameters we calculated
        # earlier together, and run the UNet with that input at time t.
        epsilon_theta = self._unet(
            sqrt_alphas_cumprod_x0 + sqrt_1_minus_alphas_cumprod_eps, t, context
        )

        # Line 5, calculate MSE of epsilon, epsilon_theta (predicted_eps)
        loss = torch.nn.functional.mse_loss(epsilon_theta, epsilon, reduction="none")
        loss = reduce(loss, "b ... -> b", "mean")
        return loss.mean()

    @torch.inference_mode()
    def algorithm2_sampling(
        self, image_size: int, num_channels: int, batch_size: int = 16, context=None
    ):
        """Implements Algorithm 2 of DDPM paper."""
        # The output shape of the data.
        shape = (batch_size, num_channels, image_size, image_size)

        # Use the device that the current model is on.
        # Assumes all of the parameters are on the same device
        device = next(self.parameters()).device

        # Line 1, initial image is pure noise
        x_t = torch.randn(shape, device=device)

        # Line 4, constants that we will use below.
        # Fixed linear schedule for Beta
        betas = linear_beta_schedule(self._num_timesteps).to(torch.float32).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        sqrt_recip_alpha = torch.sqrt(1.0 / alphas)

        # Line 2, repeat from T=num_timesteps (fully noised image) to
        # t=0 (fully unnoised image).
        for t in tqdm(
            reversed(range(0, self._num_timesteps)),
            desc="sampling loop time step",
            total=self._num_timesteps,
        ):
            # Add a batch dimension to the integer time (all items
            # in the batch are sampled at the same time).
            batched_times = torch.full(
                (batch_size,), t, device=device, dtype=torch.long
            )

            # Line 3, if t > 1 (1-indexed) then sample from a Gaussian,
            # else t is zero.
            z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

            # Line 4, lets start calculating all of the values we need. First
            # we need to calulate epsilon_theta, at time t.
            epsilon_theta = self._unet(x_t, batched_times, context)

            # Line 4, calculate the constant in front of epsilon_theta, which
            # corresponds to Beta / sqrt (1 - cumprod(alpha))
            epsilon_theta_coeff = (
                sqrt_recip_alpha * (1.0 - alphas) / sqrt_one_minus_alphas_cumprod
            )
            # Grab only the timesteps in the batch.
            epsilon_theta_coeff = extract(
                epsilon_theta_coeff, batched_times, epsilon_theta.shape
            )

            # Line 4, calculate the coefficent for x_t
            x_t_coeff = sqrt_recip_alpha
            # Grab only the values for the timesteps in the batch.
            x_t_coeff = extract(x_t_coeff, batched_times, x_t.shape)

            # Line 4, calculate the z coefficient. We use the 'fixedlarge'
            # calculation for the variance here, which corresponds to
            # beta.
            z_coeff_squared = betas
            # Grab only the values for the timesteps in the batch.
            z_coeff_squared = extract(z_coeff_squared, batched_times, z.shape)

            # Line 4, put it all together.
            x_t_minus_1 = (
                x_t_coeff * x_t
                - epsilon_theta_coeff * epsilon_theta
                + torch.sqrt(z_coeff_squared) * z
            )

            # Loop to the next timestep, setting x(t-1) to the
            # current time
            x_t = x_t_minus_1

        # Unnormalize the sample images from (-1, 1) back to (0,1)
        return unnormalize_to_zero_to_one(x_t)
