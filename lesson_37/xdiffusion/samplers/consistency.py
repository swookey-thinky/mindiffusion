import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Optional

from xdiffusion.samplers.base import ReverseProcessSampler
from xdiffusion.diffusion import DiffusionModel
from xdiffusion.utils import append_zero, append_dims


class OneStepConsistencySampler(ReverseProcessSampler):
    def __init__(
        self, sigma_min: float, sigma_max: float, rho: float, clip_denoised: bool
    ):
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._rho = rho
        self._clip_denoised = clip_denoised

    @torch.no_grad()
    def p_sample_loop(
        self,
        diffusion_model: DiffusionModel,
        latents: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ):
        x_T = latents * self._sigma_max
        context = {
            "sigmas": get_sigmas_karras(
                40, self._sigma_min, self._sigma_max, self._rho, device=latents.device
            )
        }

        x_0 = self.p_sample(
            x_T,
            context=context,
            unconditional_context=None,
            diffusion_model=diffusion_model,
        )
        return x_0.clamp(-1, 1)

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        context: Dict,
        unconditional_context: Optional[Dict],
        diffusion_model: DiffusionModel,
        guidance_fn=None,
        classifier_free_guidance: Optional[float] = None,
    ):
        """Reverse process single step.

        Samples x_{t-1} given x_t - the joint distribution p_theta(x_{t-1}|x_t).

        Args:
            x: Tensor batch of the distribution at time t.
            t: Tensor batch of the current timestep.
            low_res_context: Low resolution context, for cascaded models.
            y: Tensor batch of class labels if they exist
            guidance_fn: Optional guidance function using the gradients of a classifier
                to guide diffusion.
            classifier_free_guidance: Classifier free guidance value

        Returns:
            Tensor batch of the distribution at timestep t-1.
        """

        def denoiser(x_t, sigma):
            denoised = (
                diffusion_model._score_network(x_t, sigma)
                if diffusion_model._score_network_ema is None
                else diffusion_model._score_network_ema(x_t, sigma)
            )
            if self._clip_denoised:
                denoised = denoised.clamp(-1, 1)
            return denoised

        x_0 = sample_onestep(
            denoiser,
            x,
            context["sigmas"],
        )
        return x_0


class GeneralizedConsistencySampler(ReverseProcessSampler):
    def __init__(
        self,
        steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        clip_denoised: bool,
        sampler: str,
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        multistep=None,
    ):
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._rho = rho
        self._clip_denoised = clip_denoised
        self._steps = steps
        self._sampler = sampler
        self._s_churn = s_churn
        self._s_tmin = s_tmin
        self._s_tmax = s_tmax
        self._s_noise = s_noise
        self._multistep = multistep

    @torch.no_grad()
    def p_sample_loop(
        self,
        diffusion_model: DiffusionModel,
        latents: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ):
        x_T = latents * self._sigma_max

        if self._sampler == "progdist":
            sigmas = get_sigmas_karras(
                self._steps + 1,
                self._sigma_min,
                self._sigma_max,
                self._rho,
                device=latents.device,
            )
        else:
            sigmas = get_sigmas_karras(
                self._steps,
                self._sigma_min,
                self._sigma_max,
                self._rho,
                device=latents.device,
            )

        sample_fn = {
            "heun": sample_heun,
            "dpm": sample_dpm,
            "ancestral": sample_euler_ancestral,
            "onestep": sample_onestep,
            "progdist": sample_progdist,
            "euler": sample_euler,
            "multistep": stochastic_iterative_sampler,
        }[self._sampler]

        if self._sampler in ["heun", "dpm"]:
            sampler_args = dict(
                s_churn=self._s_churn,
                s_tmin=self._s_tmin,
                s_tmax=self._s_tmax,
                s_noise=self._s_noise,
            )
        elif self._sampler == "multistep":
            sampler_args = dict(
                ts=self._multistep,
                t_min=self._sigma_min,
                t_max=self._sigma_max,
                rho=self._rho,
                steps=self._steps,
            )
        else:
            sampler_args = {}

        def denoiser(x_t, sigma):
            denoised = (
                diffusion_model._score_network(x_t, sigma)
                if diffusion_model._score_network_ema is None
                else diffusion_model._score_network_ema(x_t, sigma)
            )
            if self._clip_denoised:
                denoised = denoised.clamp(-1, 1)
            return denoised

        x_0 = sample_fn(
            denoiser,
            x_T,
            sigmas,
            progress=True,
            **sampler_args,
        )
        return x_0.clamp(-1, 1)

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        context: Dict,
        unconditional_context: Optional[Dict],
        diffusion_model: DiffusionModel,
        guidance_fn=None,
        classifier_free_guidance: Optional[float] = None,
    ):
        """Reverse process single step.

        Samples x_{t-1} given x_t - the joint distribution p_theta(x_{t-1}|x_t).

        Args:
            x: Tensor batch of the distribution at time t.
            t: Tensor batch of the current timestep.
            low_res_context: Low resolution context, for cascaded models.
            y: Tensor batch of class labels if they exist
            guidance_fn: Optional guidance function using the gradients of a classifier
                to guide diffusion.
            classifier_free_guidance: Classifier free guidance value

        Returns:
            Tensor batch of the distribution at timestep t-1.
        """
        raise NotImplementedError()


@torch.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    progress=False,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, progress=False):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        indices = tqdm(indices)

    for i in indices:
        denoised = model(x, sigmas[i] * s_in)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        x = x + torch.randn_like(x) * sigma_up
    return x


@torch.no_grad()
def sample_midpoint_ancestral(model, x, ts, progress=False):
    """Ancestral sampling with midpoint method steps."""
    s_in = x.new_ones([x.shape[0]])
    step_size = 1 / len(ts)
    if progress:
        ts = tqdm(ts)

    for tn in ts:
        dn = model(x, tn * s_in)
        dn_2 = model(x + (step_size / 2) * dn, (tn + step_size / 2) * s_in)
        x = x + step_size * dn_2
    return x


@torch.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


@torch.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    progress=False,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        dt = sigmas[i + 1] - sigma
        x = x + d * dt
    return x


@torch.no_grad()
def sample_dpm(
    denoiser,
    x,
    sigmas,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """A sampler inspired by DPM-Solver-2 and Algorithm 2 from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    indices = range(len(sigmas) - 1)
    if progress:
        indices = tqdm(indices)

    for i in indices:
        gamma = (
            min(s_churn / (len(sigmas) - 1), 2**0.5 - 1)
            if s_tmin <= sigmas[i] <= s_tmax
            else 0.0
        )
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        # Midpoint method, where the midpoint is chosen according to a rho=3 Karras schedule
        sigma_mid = ((sigma_hat ** (1 / 3) + sigmas[i + 1] ** (1 / 3)) / 2) ** 3
        dt_1 = sigma_mid - sigma_hat
        dt_2 = sigmas[i + 1] - sigma_hat
        x_2 = x + d * dt_1
        denoised_2 = denoiser(x_2, sigma_mid * s_in)
        d_2 = to_d(x_2, sigma_mid, denoised_2)
        x = x + d_2 * dt_2
    return x


@torch.no_grad()
def sample_onestep(
    distiller,
    x,
    sigmas,
    progress=False,
):
    """Single-step generation from a distilled model."""
    s_in = x.new_ones([x.shape[0]])
    return distiller(x, sigmas[0] * s_in)


@torch.no_grad()
def stochastic_iterative_sampler(
    distiller,
    x,
    sigmas,
    ts,
    progress=False,
    t_min=0.002,
    t_max=80.0,
    rho=7.0,
    steps=40,
):
    t_max_rho = t_max ** (1 / rho)
    t_min_rho = t_min ** (1 / rho)
    s_in = x.new_ones([x.shape[0]])

    for i in tqdm(range(len(ts) - 1)):
        t = (t_max_rho + ts[i] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        x0 = distiller(x, t * s_in)
        next_t = (t_max_rho + ts[i + 1] / (steps - 1) * (t_min_rho - t_max_rho)) ** rho
        next_t = np.clip(next_t, t_min, t_max)
        x = x0 + torch.randn_like(x) * np.sqrt(next_t**2 - t_min**2)

    return x


@torch.no_grad()
def sample_progdist(
    denoiser,
    x,
    sigmas,
    progress=False,
):
    s_in = x.new_ones([x.shape[0]])
    sigmas = sigmas[:-1]  # skip the zero sigma

    indices = range(len(sigmas) - 1)
    if progress:
        indices = tqdm(indices)

    for i in indices:
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        dt = sigmas[i + 1] - sigma
        x = x + d * dt

    return x


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)
