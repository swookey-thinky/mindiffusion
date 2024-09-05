"""DDIM sampling, from DDPM."""

import torch
from typing import Dict, List, Optional

from xdiffusion.diffusion import PredictionType, DiffusionModel
from xdiffusion.samplers.base import ReverseProcessSampler
from xdiffusion.utils import broadcast_from_left, dynamic_thresholding


class DDIMSampler(ReverseProcessSampler):
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
        clip_denoised: bool = True,
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
        shape, dtype = x.shape, x.dtype
        logsnr_t = context["logsnr_t"]
        logsnr_s = context["logsnr_s"]

        epsilon_theta, variance, log_variance = self._pred_epsilon(
            x=x,
            context=context,
            diffusion_model=diffusion_model,
        )

        # If we are using classifier free guidance, then calculate the unconditional
        # epsilon as well.
        cfg = (
            classifier_free_guidance
            if classifier_free_guidance is not None
            else diffusion_model.classifier_free_guidance()
        )

        if cfg >= 0.0 and unconditional_context is not None:
            # Unconditionally sample the model
            uncond_epsilon_theta, uncond_variance, uncond_log_variance = (
                self._pred_epsilon(
                    x=x,
                    context=unconditional_context,
                )
            )
            w = cfg
            epsilon_theta = uncond_epsilon_theta + w * (
                epsilon_theta - uncond_epsilon_theta
            )

            # It's not clear from the paper how to guide learned sigma models,
            # but we will treat them the same as the epsilon-param models
            variance = uncond_variance + w * (variance - uncond_variance)
            log_variance = uncond_log_variance + w * (
                log_variance - uncond_log_variance
            )

        _maybe_clip = lambda x_: (torch.clamp(x_, -1.0, 1.0) if clip_denoised else x_)
        if diffusion_model.prediction_type() == PredictionType.EPSILON:
            pred_xstart = diffusion_model.noise_scheduler().predict_x_from_epsilon(
                z=x, context=context, epsilon=epsilon_theta
            )
            pred_epsilon = epsilon_theta
        elif diffusion_model.prediction_type() == PredictionType.V:
            # V-prediction, from https://arxiv.org/abs/2202.00512
            pred_xstart = diffusion_model.noise_scheduler().predict_x_from_v(
                z=x, context=context, v=epsilon_theta
            )
            pred_epsilon = diffusion_model.noise_scheduler().predict_epsilon_from_x(
                z=x, x=pred_xstart, context=context
            )
        else:
            raise NotImplemented(
                f"Prediction type {diffusion_model.prediction_type()} is not implemented."
            )

        if (
            "dynamic_thresholding" in diffusion_model.config().diffusion
            and diffusion_model.config().diffusion.dynamic_thresholding.enable
        ):
            if clip_denoised:
                pred_xstart = dynamic_thresholding(
                    pred_xstart,
                    p=diffusion_model.config().diffusion.dynamic_thresholding.p,
                    c=diffusion_model.config().diffusion.dynamic_thresholding.c,
                )
        else:
            pred_xstart = _maybe_clip(pred_xstart)

        x_pred_t = pred_xstart
        eps_pred_t = pred_epsilon
        stdv_s = broadcast_from_left(
            torch.sqrt(torch.nn.functional.sigmoid(-logsnr_s)), eps_pred_t.shape
        )
        alpha_s = broadcast_from_left(
            torch.sqrt(torch.nn.functional.sigmoid(logsnr_s)), x_pred_t.shape
        )
        z_s_pred = alpha_s * x_pred_t + stdv_s * eps_pred_t

        timestep_idx = context["timestep_idx"]
        return torch.where(torch.tensor(timestep_idx == 0), x_pred_t, z_s_pred)

    def _pred_epsilon(
        self,
        x,
        context: Dict,
        diffusion_model: DiffusionModel,
        epsilon_v_param: Optional[List[torch.Tensor]] = None,
    ):
        """Predict the parameterized noise from the inputs.

        Args:
            x: The input images (or other data) at time t
            t: The current timestep to predict
            low_res_context: Low resolution context, for cascaded models.
            epsilon_v_param: If not None, a tensor batch of the model output at t.
                Used to freeze the epsilon path to prevent gradients flowing back during
                VLB loss calculation.
            y: Tensor batch of class labels if they exist
            image_embeddings: Tensor batch of image embeddings
            text_embeddngs: Tensor batch of text_embeddings

        Returns:
            Tuple of:
                epsilon prediction, variance, log variance
        """
        x = diffusion_model.process_input(x=x, context=context)

        model_output = (
            diffusion_model.predict_score(x, context=context)
            if epsilon_v_param is None
            else epsilon_v_param
        )

        if diffusion_model.is_learned_sigma():
            # The v_param model calculates both the reparameterized
            # mean and variance of the distribution.
            epsilon_theta, v_param = model_output

            # Calculate the learned variance from the model output.
            # We parameterize v_param as the log of the variance.
            log_variance = v_param
            variance = torch.exp(log_variance)
        else:
            epsilon_theta = model_output

            # The predicted variance is fixed. For an epsilon
            # only model, we use the "fixedlarge" estimate of
            # the variance.
            variance, log_variance = (
                diffusion_model.noise_scheduler().variance_fixed_large(context, x.shape)
            )
        return epsilon_theta, variance, log_variance
