"""Base class for noise sampling in diffusion models.

Implements the forward process and forward process posteriors shared
across diffusion model implementations.
"""

from abc import abstractmethod
import numpy as np
import torch
from torch.distributions import LogisticNormal
from typing import Dict, Tuple

from xdiffusion.utils import (
    extract,
    broadcast_from_left,
    log1mexp,
    instantiate_from_config,
)


def cosine_logsnr_schedule(num_scales, logsnr_min, logsnr_max):
    b = np.arctan(np.exp(-0.5 * logsnr_max))
    a = np.arctan(np.exp(-0.5 * logsnr_min)) - b
    t = torch.linspace(0, 1, num_scales, dtype=torch.float32)
    return -2.0 * torch.log(torch.tan(a * t + b))


def linear_logsnr_schedule(num_scales, logsnr_min, logsnr_max):
    t = torch.linspace(0, 1, num_scales, dtype=torch.float32)
    return logsnr_max + (logsnr_min - logsnr_max) * t


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine beta schedule, proposed in Improved DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps, min_beta, max_beta):
    """Standard linear beta schedule from DDPM."""
    scale = 1000 / timesteps
    beta_start = scale * min_beta
    beta_end = scale * max_beta
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def quadratic_beta_schedule(timesteps, min_beta, max_beta):
    scale = 1000 / timesteps
    beta_start = scale * min_beta
    beta_end = scale * max_beta
    return (
        torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float64)
        ** 2
    )


def sigmoid_beta_schedule(timesteps, min_beta, max_beta):
    scale = 1000 / timesteps
    beta_start = scale * min_beta
    beta_end = scale * max_beta
    betas = torch.linspace(-6, 6, timesteps, dtype=torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class NoiseScheduler(torch.nn.Module):
    @abstractmethod
    def sample_random_times(
        self, batch_size, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def continuous(self) -> bool:
        pass

    @abstractmethod
    def variance_fixed_large(self, context, shape) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def q_posterior(
        self, x_start, x_t, context
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def q_sample(self, x_start, t, noise=None) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_x_from_epsilon(self, z, epsilon, context) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_x_from_v(self, z, v, context) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_v_from_x_and_epsilon(self, x, epsilon, t) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_epsilon_from_x(self, z, x, context) -> torch.Tensor:
        pass

    @abstractmethod
    def steps(self) -> int:
        pass

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        Args:
            ts: Tensor batch of int timesteps.
            losses: Tensor batch of float losses, one per timestep.
        """


class DiscreteNoiseScheduler(NoiseScheduler):
    """Base forward process helper class."""

    def __init__(
        self,
        schedule_type: str,
        num_scales: int,
        loss_type: str,
        min_beta: float = 0.0001,
        max_beta: float = 0.02,
        importance_sampler: Dict = {},
        **kwargs,
    ):
        super().__init__()

        timesteps = num_scales
        if schedule_type == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif schedule_type == "linear":
            betas = linear_beta_schedule(timesteps, min_beta, max_beta)
        elif schedule_type == "quadratic":
            betas = quadratic_beta_schedule(timesteps, min_beta, max_beta)
        elif schedule_type == "jsd":
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        elif schedule_type == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps, min_beta, max_beta)
        else:
            raise NotImplementedError(
                f"Noise schedule {schedule_type} not implemented."
            )

        self._importance_sampler = instantiate_from_config(importance_sampler)
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

    def steps(self) -> int:
        return self.num_timesteps

    def continuous(self) -> bool:
        return False

    def sample_random_times(
        self, batch_size, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples random times for the forward diffusion process."""
        return self._importance_sampler.sample(batch_size=batch_size, device=device)

    def variance_fixed_large(self, context, shape) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the "fixedlarge" variance from DDPM."""
        # The predicted variance is fixed. For an epsilon
        # only model, we use the "fixedlarge" estimate of
        # the variance.
        t = context["timestep"]
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

    def q_posterior(
        self, x_start, x_t, context
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        t = context["timestep"]
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None) -> torch.Tensor:
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

    def predict_x_from_epsilon(self, z, epsilon, context) -> torch.Tensor:
        t = context["timestep"]
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, z.shape) * z
            - extract(self.sqrt_recipm1_alphas_cumprod, t, z.shape) * epsilon
        )

    def predict_x_from_v(self, z, v, context):
        # From section 4 of https://arxiv.org/abs/2202.00512, the
        # v-parameterization of the score network yields:
        #   x_hat = alpha_t*z_t - sigma_t * v_hat
        t = context["timestep"]
        alpha_t = extract(self.sqrt_alphas_cumprod, t, z.shape)
        sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, t, z.shape)
        x_hat = alpha_t * z - sigma_t * v
        return x_hat

    def predict_v_from_x_and_epsilon(self, x, epsilon, t) -> torch.Tensor:
        alpha_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return alpha_t * epsilon - sigma_t * x

    def predict_epsilon_from_x(self, z, x, context) -> torch.Tensor:
        """eps = (z - alpha*x)/sigma."""
        t = context["timestep"]
        alpha_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sigma_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return (z - alpha_t * x) / sigma_t

    def update_with_all_losses(self, ts, losses):
        """Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        Args:
            ts: Tensor batch of int timesteps.
            losses: Tensor batch of float losses, one per timestep.
        """
        self._importance_sampler.update_with_all_losses(ts, losses)


class ContinuousNoiseScheduler(NoiseScheduler):
    """Base forward process helper class."""

    def __init__(
        self,
        num_scales: int,
        logsnr_schedule: str,
        loss_type: str,
        logsnr_min: float,
        logsnr_max: float,
        **kwargs,
    ):
        super().__init__()

        if logsnr_schedule == "cosine":
            gammas = cosine_logsnr_schedule(num_scales + 1, logsnr_min, logsnr_max)
        elif logsnr_schedule == "linear":
            gammas = linear_logsnr_schedule(num_scales + 1, logsnr_min, logsnr_max)
        else:
            raise NotImplementedError(
                f"Noise schedule {logsnr_schedule} not implemented."
            )

        self.num_timesteps = num_scales

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

        sigma2 = torch.nn.functional.sigmoid(-gammas)
        alphas = torch.sqrt(1.0 - sigma2)
        register_buffer("gammas", gammas)
        register_buffer("alphas", alphas)
        register_buffer("sigma2", sigma2)
        register_buffer("sqrt_sigma2", torch.sqrt(sigma2))

    def steps(self) -> int:
        return self.num_timesteps

    def continuous(self) -> bool:
        return True

    def sample_random_times(
        self, batch_size, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples random times for the forward diffusion process."""
        t = torch.rand(size=(batch_size,), dtype=torch.float32, device=device)
        weights = torch.ones_like(t)
        return t, weights

    def variance_fixed_large(self, context, shape) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the "fixedlarge" variance from DDPM."""
        # fixed_large variance setting, with gamma = 1.0
        # (last term of equation 5 from https://arxiv.org/abs/2202.00512)
        logsnr_t = broadcast_from_left(context["logsnr_t"], shape=shape)
        logsnr_s = broadcast_from_left(context["logsnr_s"], shape=shape)

        # Eq. 5 from https://arxiv.org/abs/2202.00512:
        #   r = e^(lambda_t - lambda_s)
        r = torch.exp(logsnr_t - logsnr_s)

        # expm1 is numerically stable according to section 4 of
        # https://arxiv.org/abs/2107.00630
        one_minus_r = -torch.expm1(logsnr_t - logsnr_s)
        log_one_minus_r = log1mexp(logsnr_s - logsnr_t)  # log(1-SNR(t)/SNR(s))

        # fixed_large variance setting, with gamma = 1.0
        # (last term of equation 5 from https://arxiv.org/abs/2202.00512)
        var = one_minus_r * torch.nn.functional.sigmoid(-logsnr_t)
        logvar = log_one_minus_r + torch.nn.functional.logsigmoid(-logsnr_t)
        return var, logvar

    def q_posterior(
        self, x_start, x_t, context
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        x_hat = x_start
        z_t = x_t

        logsnr_s = broadcast_from_left(context["logsnr_s"], z_t.shape)
        logsnr_t = broadcast_from_left(context["logsnr_t"], z_t.shape)
        assert torch.all(logsnr_s > logsnr_t)

        # Variance preserving diffusion process, so we have (See Section 2 of
        # https://arxiv.org/abs/2202.00512):
        #   lambda = log(alpha^2 / sigma^2)
        #   sigma^2 = 1 - alpha^2
        # yields:
        #   alpha = sqrt(sigmoid(lambda))
        #   sigma = sqrt(sigmoid(-lambda))
        alpha_s = torch.sqrt(torch.nn.functional.sigmoid(logsnr_s))

        # Numerically stable version of alpha_s/alpha_t, when t=1.0
        # (which would normally give a 0 in the denominator)
        alpha_st = torch.sqrt(
            (1.0 + torch.exp(-logsnr_t)) / (1.0 + torch.exp(-logsnr_s))
        )

        # Eq. 5 from https://arxiv.org/abs/2202.00512:
        #   r = e^(lambda_t - lambda_s)
        r = torch.exp(logsnr_t - logsnr_s)

        # expm1 is numerically stable according to section 4 of
        # https://arxiv.org/abs/2107.00630
        one_minus_r = -torch.expm1(logsnr_t - logsnr_s)

        # The first two terms in equation 5 from https://arxiv.org/abs/2202.00512
        mean = r * alpha_st * z_t + one_minus_r * alpha_s * x_hat

        # fixed_large variance setting, with gamma = 1.0
        # (last term of equation 5 from https://arxiv.org/abs/2202.00512)
        log_one_minus_r = log1mexp(logsnr_s - logsnr_t)  # log(1-SNR(t)/SNR(s))
        posterior_variance = one_minus_r * torch.nn.functional.sigmoid(-logsnr_s)
        posterior_log_variance = log_one_minus_r + torch.nn.functional.logsigmoid(
            -logsnr_s
        )
        return mean, posterior_variance, posterior_log_variance.clamp(min=1e-20)

    def q_sample(self, x_start, t, noise=None) -> torch.Tensor:
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

        t_idx = (t * self.num_timesteps).to(torch.long)
        return (
            extract(self.alphas, t_idx, x_start.shape) * x_start
            + extract(self.sqrt_sigma2, t_idx, x_start.shape) * noise
        )

    def logsnr(self, t):
        t_idx = torch.clamp(
            (t * self.num_timesteps).to(torch.long), 0, self.num_timesteps
        )
        return extract(self.gammas, t_idx, t.shape)

    def predict_x_from_epsilon(self, z, epsilon, context):
        """x = (z - sigma*eps)/alpha.

        Eq. 10 from https://arxiv.org/abs/2107.00630, with the
        implementation pulled from:
        https://github.com/google-research/google-research/blob/master/diffusion_distillation/diffusion_distillation/dpm.py#L86C1-L90C53
        """
        logsnr_t = broadcast_from_left(context["logsnr_t"], z.shape)
        return torch.sqrt(1.0 + torch.exp(-logsnr_t)) * (
            z - epsilon * torch.rsqrt(1.0 + torch.exp(logsnr_t))
        )

    def predict_x_from_v(self, z, v, context) -> torch.Tensor:
        # From section 4 of https://arxiv.org/abs/2202.00512, the
        # v-parameterization of the score network yields:
        #   x_hat = alpha_t*z_t - sigma_t * v_hat
        logsnr_t = broadcast_from_left(context["logsnr_t"], z.shape)
        alpha_t = torch.sqrt(torch.nn.functional.sigmoid(logsnr_t))
        sigma_t = torch.sqrt(torch.nn.functional.sigmoid(-logsnr_t))
        x_hat = alpha_t * z - sigma_t * v
        return x_hat

    def predict_v_from_x_and_epsilon(self, x, epsilon, t) -> torch.Tensor:
        t_idx = (t * self.num_timesteps).to(torch.long)
        alpha_t = extract(self.alphas, t_idx, x.shape)
        sigma_t = extract(self.sqrt_sigma2, t_idx, x.shape)
        return alpha_t * epsilon - sigma_t * x

    def predict_epsilon_from_x(self, z, x, context) -> torch.Tensor:
        """eps = (z - alpha*x)/sigma."""
        logsnr_t = broadcast_from_left(context["logsnr_t"], z.shape)
        return torch.sqrt(1.0 + torch.exp(logsnr_t)) * (
            z - x * torch.rsqrt(1.0 + torch.exp(-logsnr_t))
        )

    def update_with_all_losses(self, ts, losses):
        """Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        Args:
            ts: Tensor batch of int timesteps.
            losses: Tensor batch of float losses, one per timestep.
        """
        return


class DiscreteRectifiedFlowNoiseScheduler(torch.nn.Module):
    def __init__(self, steps: int, max_time: float, **kwargs):
        super().__init__()
        self._epsilon = 1e-3
        self._max_time = max_time
        self._steps = steps

        distribution = "uniform-clipped"
        if "distribution" in kwargs:
            distribution = kwargs["distribution"]
        assert distribution in ["uniform", "uniform-clipped", "logit-normal"]

        if distribution == "uniform":
            self._sample_t = torch.rand
            self._epsilon = 0.0
        elif distribution == "uniform-clipped":
            self._sample_t = torch.rand
        else:
            assert distribution == "logit-normal"
            print("Using logit-normal rectified flow scheduler.")
            loc = 0.0
            scale = 1.0
            self.distribution = LogisticNormal(
                torch.tensor([loc]), torch.tensor([scale])
            )
            self._sample_t = lambda batch_size, device: self.distribution.sample(
                (batch_size,)
            )[:, 0].to(device)
            self._epsilon = 0.0

    def sample_random_times(
        self, batch_size, device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Rectified flow time is in the range (eps, 1.0 - eps)
        t = (
            self._sample_t(batch_size, device=device) * (self._max_time - self._epsilon)
            + self._epsilon
        )
        return t, torch.ones_like(t)

    def continuous(self) -> bool:
        return False

    def variance_fixed_large(self, context, shape) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplemented()

    def q_posterior(
        self, x_start, x_t, context
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplemented()

    def q_sample(self, x_start, t, noise=None) -> torch.Tensor:
        assert noise is not None

        # t=1.0 should be noise, t=0.0 should be x_start.
        t_expanded = broadcast_from_left(t, shape=x_start.shape)
        perturbed_data = t_expanded * x_start + (1.0 - t_expanded) * noise
        return perturbed_data

    def predict_x_from_epsilon(self, z, epsilon, context) -> torch.Tensor:
        raise NotImplemented()

    def predict_x_from_v(self, z, v, context) -> torch.Tensor:
        raise NotImplemented()

    def predict_v_from_x_and_epsilon(self, x, epsilon, t) -> torch.Tensor:
        raise NotImplemented()

    def predict_epsilon_from_x(self, z, x, context) -> torch.Tensor:
        raise NotImplemented()

    def steps(self) -> int:
        return self._steps

    def update_with_all_losses(self, ts, losses):
        """Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        Args:
            ts: Tensor batch of int timesteps.
            losses: Tensor batch of float losses, one per timestep.
        """
        return
