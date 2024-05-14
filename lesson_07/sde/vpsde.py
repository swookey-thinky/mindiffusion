"""Variance Preserving SDE.

This package implements a variance preserving SDE, which is equivalent to
the DDPM implementation from "Denoising Diffusion Probabalistic Models".

This implementation follows the implementation here:
https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py#L112
"""

import torch
import numpy as np

from sde.base import SDE


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        """Construct a Variance Preserving SDE (DDPM).

        Args:
          beta_min: value of beta(0), which corresponds to $beta_min^bar$
          beta_max: value of beta(1), which corresponds to $beta_max^bar$
          N: number of discretization steps
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        """Drift and diffusion coefficients for DDPM.

        Eq. 11 from the Score-SDE paper.
        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """Eq. 33 from Score-SDE paper."""
        log_mean_coeff = (
            -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        )
        mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$.

        The prior sample for DDPM is a gaussian distribution of zero mean and unit
        variance.

        Args:
            x: Tensor batch of data of shape (B, C, H, W).
            t: Tensor batch of continuous time, as torch.float32

        Returns:
            Tensor batch of shape (B, C, H, W) sampling from the prior
            distribution $p_T(x)$.

        """
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3)) / 2.0
        return logps

    def discretize(self, x, t):
        """DDPM discretization in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G

    def score(self, x, t, score_model, continuous):
        """Calculates the score function of the SDE, $/grad{/log{p_t(x)}}$.

        Args:
            x: Tensor batch of data of shape (B, C, H, W).
            t: Tensor batch of continuous time, as torch.float32
            score_model: The score model to use with the SDE.
            continuous: True if this is a continuous time SDE.

        Returns:
            Tensor batch of shape (B, C, H, W) sampling from the prior
            distribution $p_T(x)$.
        """

        # Scale neural network output by standard deviation and flip sign
        if continuous:
            # For VP-trained models, t=0 corresponds to the lowest noise level
            labels = t * (self.N - 1)
            score = score_model(x, labels)
            std = self.marginal_prob(torch.zeros_like(x), t)[1]
        else:
            # For VP-trained models, t=0 corresponds to the lowest noise level
            labels = t * (self.N - 1)
            score = score_model(x, labels)
            std = self.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

        score = -score / std[:, None, None, None]
        return score
