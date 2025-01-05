import numpy as np
import torch
from tqdm import tqdm
from typing import Dict, Optional

from xdiffusion.samplers.base import ReverseProcessSampler
from xdiffusion.diffusion import DiffusionModel


class StochasticSampler(ReverseProcessSampler):
    """EDM sampler (Algoritm 2 from the paper)."""

    def __init__(
        self,
        num_steps: int = 18,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
    ):
        super().__init__()

        self._num_steps = num_steps
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._rho = rho
        self._S_churn = S_churn
        self._S_min = S_min
        self._S_max = S_max
        self._S_noise = S_noise

    @torch.no_grad()
    def p_sample_loop(
        self,
        diffusion_model: DiffusionModel,
        latents: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ):
        score_network = diffusion_model._score_network

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(self._sigma_min, score_network.sigma_min)
        sigma_max = min(self._sigma_max, score_network.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(
            self._num_steps, dtype=torch.float64, device=latents.device
        )
        t_steps = (
            sigma_max ** (1 / self._rho)
            + step_indices
            / (self._num_steps - 1)
            * (sigma_min ** (1 / self._rho) - sigma_max ** (1 / self._rho))
        ) ** self._rho
        t_steps = torch.cat(
            [score_network.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )  # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(
            tqdm(zip(t_steps[:-1], t_steps[1:]), total=self._num_steps, leave=False)
        ):  # 0, ..., N-1
            x_cur = x_next
            x_next = self.p_sample(
                x=x_cur,
                context={
                    "class_labels": class_labels,
                    "step": i,
                    "t_cur": t_cur,
                    "t_next": t_next,
                },
                unconditional_context={},
                diffusion_model=diffusion_model,
            )

        return x_next

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
        t_cur = context["t_cur"]
        t_next = context["t_next"]
        class_labels = context["class_labels"]
        i = context["step"]
        score_network = diffusion_model._score_network
        x_cur = x

        # Increase noise temporarily.
        gamma = (
            min(self._S_churn / self._num_steps, np.sqrt(2) - 1)
            if self._S_min <= t_cur <= self._S_max
            else 0
        )
        t_hat = score_network.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (
            t_hat**2 - t_cur**2
        ).sqrt() * self._S_noise * torch.randn_like(x_cur)

        # Euler step.
        denoised = score_network(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < self._num_steps - 1:
            denoised = score_network(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        return x_next


class GeneralizedStochasticSampler(ReverseProcessSampler):
    """Generalized sampler representing all of the techniques in the paper."""

    def __init__(
        self,
        num_steps: int = 18,
        sigma_min: Optional[float] = None,
        sigma_max: Optional[float] = None,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        solver: str = "euler",
        discretization: str = "vp",
        schedule: str = "vp",
        scaling: str = "vp",
        epsilon_s: float = 1e-3,
        C_1: float = 0.001,
        C_2: float = 0.008,
        M: int = 1000,
        alpha: float = 1,
    ):
        super().__init__()

        self._num_steps = num_steps
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self._rho = rho
        self._S_churn = S_churn
        self._S_min = S_min
        self._S_max = S_max
        self._S_noise = S_noise
        self._solver = solver
        self._discretization = discretization
        self._schedule = schedule
        self._scaling = scaling
        self._epsilon_s = epsilon_s
        self._C_1 = C_1
        self._C_2 = C_2
        self._M = M
        self._alpha = alpha

    @torch.no_grad()
    def p_sample_loop(
        self,
        diffusion_model: DiffusionModel,
        latents: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ):
        score_network = diffusion_model._score_network

        assert self._solver in ["euler", "heun"]
        assert self._discretization in ["vp", "ve", "iddpm", "edm"]
        assert self._schedule in ["vp", "ve", "linear"]
        assert self._scaling in ["vp", "none"]

        # Helper functions for VP & VE noise level schedules.
        vp_sigma = (
            lambda beta_d, beta_min: lambda t: (
                np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1
            )
            ** 0.5
        )
        vp_sigma_deriv = (
            lambda beta_d, beta_min: lambda t: 0.5
            * (beta_min + beta_d * t)
            * (sigma(t) + 1 / sigma(t))
        )
        vp_sigma_inv = (
            lambda beta_d, beta_min: lambda sigma: (
                (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
            )
            / beta_d
        )
        ve_sigma = lambda t: t.sqrt()
        ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
        ve_sigma_inv = lambda sigma: sigma**2

        # Select default noise level range based on the specified time step discretization.
        sigma_min = self._sigma_min
        sigma_max = self._sigma_max
        if sigma_min is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=self._epsilon_s)
            sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
                self._discretization
            ]
        if sigma_max is None:
            vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
            sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80}[
                self._discretization
            ]

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, score_network.sigma_min)
        sigma_max = min(sigma_max, score_network.sigma_max)

        # Compute corresponding betas for VP.
        vp_beta_d = (
            2
            * (np.log(sigma_min**2 + 1) / self._epsilon_s - np.log(sigma_max**2 + 1))
            / (self._epsilon_s - 1)
        )
        vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

        # Define time steps in terms of noise level.
        step_indices = torch.arange(
            self._num_steps, dtype=torch.float64, device=latents.device
        )
        if self._discretization == "vp":
            orig_t_steps = 1 + step_indices / (self._num_steps - 1) * (
                self._epsilon_s - 1
            )
            sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
        elif self._discretization == "ve":
            orig_t_steps = (sigma_max**2) * (
                (sigma_min**2 / sigma_max**2) ** (step_indices / (self._num_steps - 1))
            )
            sigma_steps = ve_sigma(orig_t_steps)
        elif self._discretization == "iddpm":
            u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
            alpha_bar = (
                lambda j: (0.5 * np.pi * j / self._M / (self._C_2 + 1)).sin() ** 2
            )
            for j in torch.arange(self._M, 0, -1, device=latents.device):  # M, ..., 1
                u[j - 1] = (
                    (u[j] ** 2 + 1)
                    / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=self._C_1)
                    - 1
                ).sqrt()
            u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
            sigma_steps = u_filtered[
                ((len(u_filtered) - 1) / (self._num_steps - 1) * step_indices)
                .round()
                .to(torch.int64)
            ]
        else:
            assert self._discretization == "edm"
            sigma_steps = (
                sigma_max ** (1 / self._rho)
                + step_indices
                / (self._num_steps - 1)
                * (sigma_min ** (1 / self._rho) - sigma_max ** (1 / self._rho))
            ) ** self._rho

        # Define noise level schedule.
        if self._schedule == "vp":
            sigma = vp_sigma(vp_beta_d, vp_beta_min)
            sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
            sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
        elif self._schedule == "ve":
            sigma = ve_sigma
            sigma_deriv = ve_sigma_deriv
            sigma_inv = ve_sigma_inv
        else:
            assert self._schedule == "linear"
            sigma = lambda t: t
            sigma_deriv = lambda t: 1
            sigma_inv = lambda sigma: sigma

        # Define scaling schedule.
        if self._scaling == "vp":
            s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
            s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
        else:
            assert self._scaling == "none"
            s = lambda t: 1
            s_deriv = lambda t: 0

        # Compute final time steps based on the corresponding noise levels.
        t_steps = sigma_inv(score_network.round_sigma(sigma_steps))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

        # Main sampling loop.
        t_next = t_steps[0]
        x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
        for i, (t_cur, t_next) in enumerate(
            tqdm(zip(t_steps[:-1], t_steps[1:]), total=len(t_steps) - 1, leave=False)
        ):  # 0, ..., N-1
            x_cur = x_next
            x_next = self.p_sample(
                x=x_cur,
                context={
                    "class_labels": class_labels,
                    "step": i,
                    "t_cur": t_cur,
                    "t_next": t_next,
                    "sigma": sigma,
                    "sigma_deriv": sigma_deriv,
                    "sigma_inv": sigma_inv,
                    "s": s,
                    "s_deriv": s_deriv,
                },
                unconditional_context={},
                diffusion_model=diffusion_model,
            )
        return x_next

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
        t_cur = context["t_cur"]
        t_next = context["t_next"]
        class_labels = context["class_labels"]
        i = context["step"]
        sigma = context["sigma"]
        sigma_deriv = context["sigma_deriv"]
        sigma_inv = context["sigma_inv"]
        s = context["s"]
        s_deriv = context["s_deriv"]
        score_network = diffusion_model._score_network
        x_cur = x

        # Increase noise temporarily.
        gamma = (
            min(self._S_churn / self._num_steps, np.sqrt(2) - 1)
            if self._S_min <= sigma(t_cur) <= self._S_max
            else 0
        )
        t_hat = sigma_inv(
            score_network.round_sigma(sigma(t_cur) + gamma * sigma(t_cur))
        )
        x_hat = s(t_hat) / s(t_cur) * x_cur + (
            sigma(t_hat) ** 2 - sigma(t_cur) ** 2
        ).clip(min=0).sqrt() * s(t_hat) * self._S_noise * torch.randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = score_network(x_hat / s(t_hat), sigma(t_hat), class_labels).to(
            torch.float64
        )
        d_cur = (
            sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
        ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + self._alpha * h * d_cur
        t_prime = t_hat + self._alpha * h

        # Apply 2nd order correction.
        if self._solver == "euler" or i == self._num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert self._solver == "heun"
            denoised = score_network(
                x_prime / s(t_prime), sigma(t_prime), class_labels
            ).to(torch.float64)
            d_prime = (
                sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)
            ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * (
                (1 - 1 / (2 * self._alpha)) * d_cur + 1 / (2 * self._alpha) * d_prime
            )

        return x_next
