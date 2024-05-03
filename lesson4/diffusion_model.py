"""Forward and reverse diffusion processes.

This package implements the forward and reverse diffusion processes from
the paper "Improved Techniques for Training Score-Based Generative Models"
(https://arxiv.org/abs/2006.09011).
"""

import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

from score_network import NCSNv2


class GaussianDiffusion_NCSNv2(torch.nn.Module):
    """Core NCSN algorithms.

    This implements the training loss, forward process, and reverse process
    for a Noise Conditioned Score Network (v2) based on the paper
    "Improved Techniques for Training Score-Based Generative Models"
    (https://arxiv.org/abs/2006.09011).
    """

    def __init__(self):
        super().__init__()

        # Updated hyperparameters from the original paper.
        sigma_begin = 17.0
        sigma_end = 0.01
        L = 96

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), L))
        ).float()

        self.register_buffer("sigmas", sigmas)
        self._scorenet = NCSNv2(self.sigmas)

    def calculate_loss_on_batch(self, x: torch.Tensor):
        """Calculates the loss on a batch of image data.

        This is Eq. 6 from the original paper, and is unchanged in v2.

        Args:
            x: A batch of normalized image data, of shape (B, C, H, W)

        Returns:
            The loss, based on the negative log likelihood bound of the reverse
            process.
        """
        # IMPORTANT! This is described in Denoising Score Matching, where we
        # first perturb the datapoint in order to circumvent the trace operator
        # in the score function. This means we are estimating the score of the
        # perturbed datapoint.
        x = x / 256.0 * 255.0 + torch.rand_like(x) / 256.0

        # Random timesteps for labels
        y = torch.randint(0, len(self.sigmas), (x.shape[0],), device=x.device)
        loss = self._anneal_dsm_score_estimation(self._scorenet, x, y, self.sigmas)
        return loss

    def reverse_diffusion_full_trajectory(
        self,
        num_samples: int,
        device,
        spatial_width: int = 28,
        score_model: Optional[torch.nn.Module] = None,
    ):
        """Full reverse diffusion process.

        Starting with Gaussian noise, reverse diffuse the data for the
        given number of timessteps.

        This is Algorithm 1 (Annealed Langevin Dynamics) from the original paper,
        and includes the extra denoising step from v2.

        Args:
            num_samples: The number of samples to generate.
            spatial_width: The spatial width of the data.

        Returns:
            Batch of num_samples denoised images after running num_timesteps of
            reverse diffusion.
        """
        xt = torch.rand(num_samples, 1, spatial_width, spatial_width, device=device)

        scorenet = self._scorenet if score_model is None else score_model
        return self._anneal_langevin_dynamics(xt, scorenet, self.sigmas)

    def _anneal_dsm_score_estimation(self, scorenet, x, y, sigmas, anneal_power=2.0):
        """Calculate loss using Annealed Denoising Score Estimation.

        Implements Eq. 6 from Section 4.2 of the original paper, and is
        unchanged in v2.

        Args:
            scorenet: The score network used to predict the mean and variance of
                the reverse process.
            x: Tensor batch of samples used for prediction
            y: Tensor batch of integer indices into the variance array (sigmas)
            anneal_power: Constant used for annealing the loss.

        Returns:
            The loss calculated according to Eq. 6.
        """
        used_sigmas = sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        perturbed_samples = x + torch.randn_like(x) * used_sigmas
        target = -1 / (used_sigmas**2) * (perturbed_samples - x)
        scores = scorenet(perturbed_samples, y)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = (
            1
            / 2.0
            * ((scores - target) ** 2).sum(dim=-1)
            * used_sigmas.squeeze() ** anneal_power
        )

        return loss.mean(dim=0)

    def _anneal_langevin_dynamics(self, x_T, scorenet, sigmas, T=11, epsilon=9.1e-06):
        """Annealed Langevin Dynamics for sampling.

        This implements Algorithm 1 (Annealed Langevin Dynamics) from
        Section 2.2, which only differs from the original Algorithm 1 in
        Section 4.3 of the original paper by an extra denoising step at the
        end.

        Args:
            x_T: Tensor batch of initial samples at time T, typically a Gaussian.
            scorenet: The score network used to predict the reverse process transitions.
            T: The number of steps to use at each noise scale.
            epsilon: The Euler-Maruyama learning rate for each step.

        Returns:
            The fully denoised image at time 0.
        """
        progress_bar = tqdm(
            total=T * len(sigmas),
            desc="annealed Langevin dynamics sampling",
            leave=False,
        )
        x_mod = x_T
        with torch.inference_mode():
            for i, sigma in enumerate(sigmas):
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * i
                labels = labels.long()
                alpha = epsilon * (sigma / sigmas[-1]) ** 2
                for s in range(T):
                    x_mod = (
                        x_mod
                        + (alpha / 2.0) * scorenet(x_mod, labels)
                        + torch.sqrt(alpha).to(x_mod.device) * torch.randn_like(x_mod)
                    )
                    progress_bar.update(1)

            # Denoise the last step
            last_noise = (len(sigmas) - 1) * torch.ones(
                x_mod.shape[0], device=x_mod.device
            )
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)

            return torch.clamp(x_mod, 0.0, 1.0).to("cpu")
