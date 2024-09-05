import abc
import numpy as np
import torch
from typing import Dict, Optional

from xdiffusion.sde.base import SDE
from xdiffusion.sde.vpsde import VPSDE
from xdiffusion.sde.subvpsde import subVPSDE
from xdiffusion.samplers.base import ReverseProcessSampler
from xdiffusion.diffusion import DiffusionModel
from xdiffusion.utils import DotConfig, instantiate_partial_from_config


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(
        self, sde: SDE, diffusion_model: DiffusionModel, probability_flow: bool = False
    ):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(diffusion_model.predict_score, probability_flow)
        self.score_fn = diffusion_model.predict_score

    @abc.abstractmethod
    def update(self, x, context):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde: SDE, diffusion_model: DiffusionModel, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = diffusion_model.predict_score
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update(self, x, context):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class PredictorCorrectorSampler(ReverseProcessSampler):
    def __init__(self, config: DotConfig, sde: SDE, diffusion_model: DiffusionModel):
        self._predictor = instantiate_partial_from_config(config.predictor.to_dict())(
            sde=sde, diffusion_model=diffusion_model
        )
        self._corrector = instantiate_partial_from_config(config.corrector.to_dict())(
            sde=sde, diffusion_model=diffusion_model
        )

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
        x, x_mean = self._corrector.update(x, context)
        x, x_mean = self._predictor.update(x, context)
        return x_mean if context["denoise_final"] else x


class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(
        self, sde: SDE, diffusion_model: DiffusionModel, probability_flow=False
    ):
        super().__init__(sde, diffusion_model, probability_flow)
        if not isinstance(sde, VPSDE):  # and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported."
            )
        assert (
            not probability_flow
        ), "Probability flow not supported by ancestral sampling"

    def vpsde_update(self, x, context):
        sde = self.sde
        t = context["timestep"]
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, context)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1.0 - beta)[
            :, None, None, None
        ]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update(self, x, context):
        if isinstance(self.sde, VPSDE):
            return self.vpsde_update(x, context)


class EulerMaruyamaPredictor(Predictor):
    """Euler-Maruyama predictor."""

    def __init__(
        self, sde: SDE, diffusion_model: DiffusionModel, probability_flow=False
    ):
        super().__init__(sde, diffusion_model, probability_flow)

    def update(self, x, context):
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, context)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


class ReverseDiffusionPredictor(Predictor):
    def __init__(
        self, sde: SDE, diffusion_model: DiffusionModel, probability_flow=False
    ):
        super().__init__(sde, diffusion_model, probability_flow)

    def update(self, x, context):
        f, G = self.rsde.discretize(x, context)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


class LangevinCorrector(Corrector):
    def __init__(self, sde: SDE, diffusion_model: DiffusionModel, snr, n_steps):
        super().__init__(sde, diffusion_model, snr, n_steps)

    def update(self, x, context):
        t = context["timestep"]
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for _ in range(n_steps):
            grad = score_fn(x, context)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, **kwargs):
        pass

    def update(self, x, context):
        return x, x
