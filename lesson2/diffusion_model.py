"""Forward and reverse diffusion processes.

This package implements the forward and reverse diffusion processes from
the paper "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
(https://arxiv.org/abs/1503.03585).
"""

import numpy as np
from tqdm import tqdm
from typing import Optional
import torch

from noise_predictor import MeanAndCovarianceNetwork
from utils import (
    extract,
    mnist_normalize,
)


class GaussianDiffusion_DPM(torch.nn.Module):
    """Core DPM algorithm.

    This implements the training loss, forward process, and reverse process
    for a Diffusion Probabilistic Model based on the paper
    "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"
    (https://arxiv.org/abs/1503.03585).
    """

    def __init__(self, num_timesteps: int):
        """Initialized the module.

        Args:
            num_timesteps: The number of steps in the forward/reverse diffusion process.
        """
        super().__init__()

        self._num_timesteps = num_timesteps

        # Build the mu,sigma prediction model
        self._mu_sigma_predictor = MeanAndCovarianceNetwork(num_timesteps=num_timesteps)

    def calculate_loss_on_batch(self, x: torch.Tensor):
        """Calculates the loss on a batch of image data.

        Args:
            x: A batch of normalized image data, of shape (B, C, H, W)

        Returns:
            The loss, based on the negative log likelihood bound of the reverse
            process.
        """
        B, C, H, W = x.shape
        device = x.device

        # Calculate random timesteps
        x0 = mnist_normalize(x)

        # q(xt)
        xt, t, mu_posterior, sigma_posterior = self.forward_diffusion_full_trajectory(
            x0, num_timesteps=self._num_timesteps
        )

        # p(xt)
        mu, sigma = self._mu_sigma_predictor(xt, t)
        negL_bound = self._negative_log_likelihood_bound(
            mu=mu,
            sigma=sigma,
            mu_posterior=mu_posterior,
            sigma_posterior=sigma_posterior,
            num_timesteps=self._num_timesteps,
            betas=self._mu_sigma_predictor.beta,
        )

        # The loss is the difference between the negL_bound and log likelihood of
        # an isotropic Gaussian.
        negL_gauss = (
            0.5 * (1 + torch.log(torch.as_tensor(2.0 * np.pi, device=device)))
        ) + 0.5 * torch.log(torch.as_tensor(1.0, device=device))
        negL_diff = negL_bound - negL_gauss
        L_diff_bits = negL_diff / torch.log(torch.as_tensor(2.0, device=device))
        loss = L_diff_bits.mean()
        return loss

    def forward_diffusion_full_trajectory(
        self,
        x: torch.Tensor,
        num_timesteps: int,
        fixed_t: Optional[torch.Tensor] = None,
    ):
        """
        The forward diffusion process (q(x)) corrupts a training sample
        with t steps of Gaussian noise, returning the corrupted sample,
        as well as the mean and the covariance of the posterior q(x^{t-1}|x^t, x^0).

        Args:
            x: A tensor batch of normalized image data, of shape (B, C, H, W)
            num_timesteps: The number of timesteps in the forward/reverse diffusion process.
            fixed_t: [Optional] The timesteps to use for diffusion, otherwise random.
        """
        B, C, H, W = x.shape
        device = x.device

        # Choose a random timestep in the range (1, num_timesteps). The
        # reverse process is fixed for timestep 0, which is why we can ignore it.
        if fixed_t is not None:
            t = fixed_t
        else:
            t = torch.randint(low=1, high=num_timesteps, size=(B,), device=device)

        # Calculate a random normal distribution
        N = torch.randn(size=(B, C, H, W), device=device, dtype=x.dtype)

        # The covariances added at this timestep
        beta = self._mu_sigma_predictor.beta
        beta_forward = extract(beta, t, x.shape)
        alpha_forward = 1.0 - beta_forward

        alpha_arr = 1.0 - beta
        alpha_cum_forward_arr = torch.cumprod(alpha_arr, dim=0)
        alpha_cum_forward = extract(alpha_cum_forward_arr, t, x.shape)

        alpha_cum_prev_arr = torch.nn.functional.pad(
            alpha_cum_forward_arr[:-1], (1, 0), value=1.0
        )
        alpha_cum_prev = extract(alpha_cum_prev_arr, t, x.shape)

        x_noisy = x * torch.sqrt(alpha_cum_forward) + N * torch.sqrt(
            1.0 - alpha_cum_forward
        )

        # See https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process
        # for a derivation of the mu and sigma calulations here, which derive from
        # the tractable formulation of q0(xt_1 | xt, x0).
        sigma = torch.sqrt(
            beta_forward * (1.0 - alpha_cum_prev) / (1.0 - alpha_cum_forward)
        )
        mu = x_noisy * torch.sqrt(alpha_forward) * (1.0 - alpha_cum_prev) / (
            1.0 - alpha_cum_forward
        ) + x * torch.sqrt(alpha_cum_prev) * beta_forward / (1.0 - alpha_cum_forward)
        sigma = sigma.reshape((B, 1, 1, 1))
        return x_noisy, t, mu, sigma

    def reverse_diffusion_single_step(self, x_t: torch.Tensor, t: torch.Tensor):
        """Single step of the reverse diffusion process.

        Args:
            x_t: Tensor batch of normalized image data at timestep t
            t: Tensor batch of the current integer timestep

        Returns:
            Tensor batch of image data at timestep t-1.
        """
        assert t[0] != 0
        mu, sigma = self._mu_sigma_predictor(x_t, t)
        x_t_minus_1 = mu + torch.randn_like(x_t) * sigma
        return x_t_minus_1

    # Sampling From the Reverse Process
    def reverse_diffusion_full_trajectory(
        self, num_timesteps: int, num_samples: int, device, spatial_width: int = 28
    ):
        """Full reverse diffusion process.

        Starting with Gaussian noise, reverse diffuse the data for the
        given number of timessteps.

        Args:
            num_timesteps: The number of timesteps in the reverse diffusion process.
            num_samples: The number of samples to generate.
            spatial_width: The spatial width of the data.

        Returns:
            Batch of num_samples denoised images after running num_timesteps of
            reverse diffusion.
        """
        with torch.inference_mode():
            # The reverse process starts with isotropic Gaussian noise
            x_t = torch.randn(
                (num_samples, 1, spatial_width, spatial_width), device=device
            )
            for i in tqdm(
                reversed(range(num_timesteps)),
                total=num_timesteps,
                leave=False,
            ):
                # We don't need to sample at timestep 0
                if i > 0:
                    t = torch.ones((num_samples,), device=device, dtype=torch.int64)
                    t = t * (i)

                    # Predict the mean and covariance at time t
                    mu, sigma = self._mu_sigma_predictor(x_t, t)

                    # Perturb x_t with the predicted mean and variance.
                    x_t = mu + torch.randn_like(x_t) * sigma
            return x_t

    def _negative_log_likelihood_bound(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        mu_posterior: torch.Tensor,
        sigma_posterior: torch.Tensor,
        num_timesteps: int,
        betas: torch.Tensor,
    ):
        """
        Calculates the negative log likelihood bound between the forward
        process posterior q (x_{t-1} | x_t) and the reverse process p(x_{t-1} | x_t).

        This is Eq. 14 from the paper.

        Args:
            mu: The prediction of the reverse process mean.
            sigma: The prediction of the reverse process covariance.
            mu_posterior: The mean of the forward process.
            sigma_posterior: The covariance of the forward process.
            num_timesteps: The number of timesteps in the forward/reverse process.
            betas: Tensor batch of variances used in training/sampling.

        Returns:
            The negative log likelihood bound on the reverse process transition.
        """
        device = mu.device

        # The KL divergence between model reverse transition and posterior from data.
        # The below equation is the analytical form of the KL divergence between two
        # multivariate Gaussian densities with diagonal covariance matrices.
        # There is an excellent discussion of how to derive this here:
        # https://ai.stackexchange.com/questions/26366/how-is-this-pytorch-expression-equivalent-to-the-kl-divergence
        KL = (
            torch.log(sigma)
            - torch.log(sigma_posterior)
            + (sigma_posterior**2 + (mu_posterior - mu) ** 2) / (2 * sigma**2)
            - 0.5
        )

        two_pi = torch.as_tensor(2.0 * np.pi, device=device)

        alpha_arr = 1.0 - betas
        beta_full_trajectory = 1.0 - torch.exp(torch.sum(torch.log(alpha_arr)))

        # In order to calculate the entropies, we note that fact that all three
        # entropy parts are calculating the entropy of a Gaussian, with different
        # variances. For help deriving the entropy of a Gaussian, see
        # https://gregorygundersen.com/blog/2020/09/01/gaussian-entropy/ for a good
        # derivation.

        # Conditional entropy H_q(x^T|x^0) - A Gaussian whose variance
        # is cumulative product of all of the forward trajectory variances
        # (e.g. the cumprod of the beta array).
        H_endpoint = (0.5 * (1.0 + torch.log(two_pi))) + 0.5 * torch.log(
            beta_full_trajectory
        )

        # Conditional entopy H_q(x^1|x^0) - A Gaussian whose variance is the first
        # element in the beta array.
        H_startpoint = (0.5 * (1.0 + torch.log(two_pi))) + 0.5 * torch.log(betas[0])

        # Prior entropy H_p(x^T) - A Gaussian with a unit variance
        H_prior = (0.5 * (1.0 + torch.log(two_pi))) + 0.5 * torch.log(
            torch.as_tensor(1.0, device=device)
        )

        negL_bound = KL * num_timesteps - (H_endpoint - H_startpoint - H_prior)
        return negL_bound
