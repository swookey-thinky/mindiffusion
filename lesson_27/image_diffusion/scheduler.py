"""Base class for noise sampling in diffusion models.

Implements the forward process and forward process posteriors shared
across diffusion model implementations.
"""

import torch

from image_diffusion.utils import extract


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule, proposed in Improved DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps):
    """Standard linear beta schedule from DDPM."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return (
        torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float64)
        ** 2
    )


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(-6, 6, timesteps, dtype=torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class NoiseScheduler(torch.nn.Module):
    """Base forward process helper class."""

    def __init__(
        self,
        beta_schedule: str,
        timesteps: int,
        loss_type: str,
    ):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError(
                f"Noise schedule {beta_schedule} not implemented."
            )

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.nn.functional.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0
        )

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        if loss_type == "l1":
            loss_fn = torch.nn.functional.l1_loss
        elif loss_type == "l2":
            loss_fn = torch.nn.functional.mse_loss
        elif loss_type == "huber":
            loss_fn = torch.nn.functional.smooth_l1_loss
        else:
            raise NotImplementedError(f"Loss function {loss_type} not implemented.")

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # Register buffer helper function to cast double back to float
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def sample_random_times(self, batch_size):
        """Samples random times for the forward diffusion process."""
        return torch.randint(
            0,
            self.num_timesteps,
            (batch_size,),
            device=self.betas.device,
            dtype=torch.long,
        )

    def variance_fixed_large(self, t, shape):
        """Calculates the "fixedlarge" variance from DDPM."""
        # The predicted variance is fixed. For an epsilon
        # only model, we use the "fixedlarge" estimate of
        # the variance.
        variance, log_variance = (
            self.betas,
            torch.log(
                torch.cat(
                    [
                        torch.unsqueeze(self.posterior_variance[1], dim=0),
                        self.betas[1:],
                    ]
                )
            ),
        )

        variance = extract(variance, t, shape)
        log_variance = extract(log_variance, t, shape)
        return variance, log_variance

    def q_posterior(self, x_start, x_t, t):
        """Compute the mean and variance of the diffusion posterior.

        Calculates $q(x_{t-1} | x_t, x_0)$

        Args:
            x_start: The initial starting state (or predicted starting state) of the distribution.
            x_t: The distribution at time t.
            t: The timestep to calculate the posterior.

        Returns:
            Tuple of:
                mean: Tensor batch of the mean of the posterior
                variance: Tensor batch of the variance of the posterior
                log_variance: Tensor batch of the log of the posterior variance, clipped.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        """Forward process for DDPM.

        Noise the initial sample x_0 to the timestep t, calculating $q(x_t | x_0)$.

        Args:
            x_start: Tensor batch of original samples at time 0
            t: Tensor batch of timesteps to noise to.
            noise: Optional fixed noise to add.

        Returns:
            Tensor batch of noised samples at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_sample_from_to(self, x_from, from_t, to_t, noise=None):
        """Forward process for DDPM.

        Noise the sample x_t_1 to the timestep t, calculating $q(x_t_2 | x_t_1)$.

        Args:
            x_from: Tensor batch of starting distribution.
            from_t: Timestep of the starting distribution.
            to_t: Timestep of the ending distribution.
            noise: Optional fixed noise to add.

        Returns:
            Tensor batch of noised samples at timestep to_t.
        """
        shape = x_from.shape

        if noise is None:
            noise = torch.randn_like(x_from)

        alpha = extract(self.sqrt_alphas_cumprod, from_t, shape)
        sigma = extract(self.sqrt_one_minus_alphas_cumprod, from_t, shape)
        alpha_next = extract(self.sqrt_alphas_cumprod, to_t, shape)
        sigma_next = extract(self.sqrt_one_minus_alphas_cumprod, to_t, shape)

        return (
            x_from * (alpha_next / alpha)
            + noise * (sigma_next * alpha - sigma * alpha_next) / alpha
        )

    def predict_xstart_from_epsilon(self, x_t, t, epsilon):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * epsilon
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def predict_xstart_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_xstart(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
