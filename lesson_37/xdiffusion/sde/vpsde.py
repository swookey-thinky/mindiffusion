"""Variance preserving SDE."""

import torch

from xdiffusion.sde.base import SDE


class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20, N=1000, T=1.0):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__()
        self._beta_0 = beta_min
        self._beta_1 = beta_max
        self._N = N
        self._T = T

        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self._alphas = 1.0 - self.discrete_betas
        self._alphas_cumprod = torch.cumprod(self._alphas, dim=0)
        self._sqrt_alphas_cumprod = torch.sqrt(self._alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self._alphas_cumprod)

    @property
    def T(self):
        """End time of the SDE."""
        return self._T

    @property
    def N(self):
        """Number of discretization steps."""
        return self._N

    def sde(self, x, context):
        """Drift and diffusion coefficients for DDPM (VPSDE).

        Eq. 11 from the Score-SDE paper.
        """
        t = context["timestep"]
        beta_t = self._beta_0 + t * (self._beta_1 - self._beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$.

        Since the marginal distribution $p_t(x)$ in a diffusion model is
        constrained to be a Gaussian distribution, this function returns
        the mean and std deviation of the prior distribution at time t.

        For VP-SDE this is Eq. 33 from Score-SDE.

        Args:
            x: Tensor batch of data of shape (B, C, H, W).
            t: Tensor batch of continuous time, as torch.float32. In the
               range [0, self.T]

        Returns:
            Tensor batch of shape (B, C, H, W) describing the mean and std deviation
            at each pixel, of the prior distribution $p_t(x)$.
        """
        log_mean_coeff = (
            -0.25 * t**2 * (self._beta_1 - self._beta_0) - 0.5 * t * self._beta_0
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

    def discretize(self, x, context):
        """DDPM discretization."""
        t = context["timestep"]
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self._alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G
