"""Rectified Flow ancestral sampler"""

import numpy as np
import torch
from typing import Dict, Optional

from xdiffusion.diffusion import DiffusionModel
from xdiffusion.samplers.base import ReverseProcessSampler


class AncestralSampler(ReverseProcessSampler):
    def __init__(self, **kwargs):
        super().__init__()

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
        sde = diffusion_model.sde()
        assert sde is not None

        # Uniform sampling over t
        # TODO: Add different sampling schedules
        dt = 1.0 / sde.N
        # default: 1e-3
        eps = 1e-3

        # Time flows from 0 -> 1, in the reverse ODE formulation, but in the
        # diffusion formulations, time flows from 1 -> 0 (which is was timestep_idx is based on).
        timestep_idx = context["timestep_idx"]
        timestep_idx = sde.N - (timestep_idx + 1)

        # num_t, t is in the range [eps,1-eps]
        num_t = timestep_idx / sde.N * (sde.T - eps) + eps
        t = torch.ones(x.shape[0], device=x.device) * num_t
        context["timestep"] = t

        # Euler-Maruyama solver for SDE dZ_t = v_θ(Z_t, t)*dt + sigma(t)*dW_t,
        # where sigma(t) = sqrt(Beta_t), in the score SDE notation.
        pred = diffusion_model.predict_score(x, context=context)

        # TODO: Implement CFG for the rectified flow sampler
        # convert to diffusion models if sampling.sigma_variance > 0.0
        # while perserving the marginal probability.
        sigma_t = sde.sigma_t(num_t)

        # Following Eq. (6) in ScoreSDE, the reverse process step for the forward
        # SDE is given by:
        # dx = [f(x, t) - g(t)**2 * grad_x(log(p_t(x)))]dt + g(t)*dW
        # with f(x,t) = v_θ and g(t) = sigma(t)
        #
        # For the rectified flow sampler, sigma_t ends up being 0, and
        # since pred is just the predicted velocity, this becomes x -> x + pred * dt.
        pred_sigma = pred + (sigma_t**2) / (
            2 * (sde.noise_scale() ** 2) * ((1.0 - num_t) ** 2)
        ) * (0.5 * num_t * (1.0 - num_t) * pred - 0.5 * (2.0 - num_t) * x)

        x = (
            x
            + pred_sigma * dt
            + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(x.device)
        )
        return x
