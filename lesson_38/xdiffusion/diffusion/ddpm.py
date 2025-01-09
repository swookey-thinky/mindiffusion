"""DDPM Based Diffusion Model.

Base implementation of a DDPM diffusion model. Support the following improvements
from different papers:

"""

import accelerate
from einops import reduce
import numpy as np
import time
import torch
from torchinfo import summary
from tqdm.autonotebook import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

from xdiffusion.diffusion import DiffusionModel, PredictionType
from xdiffusion.samplers.ancestral import AncestralSampler
from xdiffusion.samplers.base import ReverseProcessSampler
from xdiffusion.sde.base import SDE
from xdiffusion.scheduler import NoiseScheduler
from xdiffusion.utils import (
    broadcast_from_left,
    discretized_gaussian_log_likelihood,
    fix_torchinfo_for_str,
    freeze,
    instantiate_from_config,
    instantiate_partial_from_config,
    normal_kl,
    normalize_to_neg_one_to_one,
    prob_mask_like,
    unnormalize_to_zero_to_one,
    get_constant_schedule_with_warmup,
    get_obj_from_str,
    DotConfig,
)


class GaussianDiffusion_DDPM(DiffusionModel):
    """Core DDPM Diffusion algorithm, with classifier free guidance.

    This is the same core algorithm as DDPM, IDDPM, and Guided Diffusion,
    with a few changes listed above to support DaLL*E 2.
    """

    def __init__(self, config: DotConfig, vae: Optional[torch.nn.Module] = None):
        super().__init__()
        self._config = config
        self._latent_scale_factor = -1.0

        if config.diffusion.parameterization == "epsilon":
            self._prediction_type = PredictionType.EPSILON
        elif config.diffusion.parameterization == "v":
            self._prediction_type = PredictionType.V
        elif config.diffusion.parameterization == "rectified_flow":
            self._prediction_type = PredictionType.RECTIFIED_FLOW
        else:
            raise NotImplemented(
                f"Parameterization {config.diffusion.parameterization} not implemented."
            )

        self._score_network = instantiate_from_config(
            config.diffusion.score_network, use_config_struct=True
        )

        self._is_learned_sigma = config.diffusion.score_network.params.is_learned_sigma
        self._is_class_conditional = (
            (config.diffusion.score_network.params.is_class_conditional)
            if "is_class_conditional" in config.diffusion.score_network.params.to_dict()
            else False
        )
        self._num_classes = config.data.num_classes
        self._unconditional_guidance_probability = (
            config.diffusion.classifier_free_guidance.unconditional_guidance_probability
        )
        self._classifier_free_guidance = (
            config.diffusion.classifier_free_guidance.classifier_free_guidance
        )
        self._noise_scheduler: NoiseScheduler = instantiate_from_config(
            config.diffusion.noise_scheduler.to_dict()
        )

        self._context_preprocessors = torch.nn.ModuleList(
            [instantiate_from_config(c) for c in config.diffusion.context_preprocessing]
        )

        self._input_preprocessor = instantiate_from_config(
            config.diffusion.input_preprocessing.to_dict()
        )
        self._unconditional_context = instantiate_from_config(
            config.diffusion.classifier_free_guidance.unconditional_context.to_dict()
        )

        # Instantiate the sampler
        self._reverse_process_sampler: ReverseProcessSampler = instantiate_from_config(
            config.diffusion.sampling.to_dict()
        )

        # Get the SDE associated with this diffusion model, if it exists.
        if "sde" in config.diffusion.to_dict():
            self._sde = instantiate_from_config(config.diffusion.sde.to_dict())
        else:
            self._sde = None

        if "normalize" in config.data:
            self._normalize = get_obj_from_str(config.data.normalize)
        else:
            self._normalize = normalize_to_neg_one_to_one

        if "unnormalize" in config.data:
            self._unnormalize = get_obj_from_str(config.data.unnormalize)
        else:
            self._unnormalize = unnormalize_to_zero_to_one

        # If we are a latent diffusion model, then create the latent encoder
        self._latent_encoder = vae
        if vae is None and "latent_encoder" in self._config.diffusion:
            print("Loading uninitialized latent encoder.")
            self._latent_encoder = instantiate_from_config(
                config.diffusion.latent_encoder.to_dict()
            )
        if self._latent_encoder is not None:
            freeze(self._latent_encoder)

    def models(self) -> List[DiffusionModel]:
        return [self]

    def sde(self) -> Optional[SDE]:
        """Gets the SDE associated with this diffusion model, if it exists."""
        return self._sde

    def config(self) -> DotConfig:
        return self._config

    def update_ema(
        self, step: int, total_steps: int, scale_fn: Callable[[int], Tuple[float, int]]
    ):
        # EMA not supported yet
        return

    def forward(self, images: torch.FloatTensor, context: Dict, **kwargs):
        return self.loss_on_batch(images=images, context=context)

    def loss_on_batch(self, images, context: Dict, **kwargs) -> Dict:
        """Calculates the reverse process loss on a batch of images.

        Args:
            images: Tensor batch of images, of shape [B, C, H, W]
            prompts: List of prompts to use for conditioning, of length B
            clip_embedder: The CLIP model to use for image and text embeddings.
            low_resolution_images: Tensor batch of low resolution images, if this is
                part of a cascade.
            y: Tensor batch of class labels, if they exist.

        Returns:
            Dictionary of loss values, of which the "loss" entry will
            be the training loss.
        """
        start_time = time.perf_counter()
        B = images.shape[0]
        device = images.device
        context = context.copy()

        # The images are normalized into the range (-1, 1),
        # from Section 3.3:
        # "We assume that image data consists of integers in {0, 1, . . . , 255} scaled linearly to [−1, 1]."
        x_0 = self._normalize(images)

        # Encode the normalized images into latents to send to the score network
        if self._latent_encoder is not None:
            z_0 = self._latent_encoder.encode_to_latents(x_0)

            if self._latent_scale_factor == -1.0:
                del self._latent_scale_factor
                self.register_buffer(
                    "_latent_scale_factor", 1.0 / z_0.detach().flatten().std()
                )
                print(f"Latent scale factor: {self._latent_scale_factor}")

            # Scale the latents
            z_0 = z_0 * self._latent_scale_factor
        else:
            z_0 = x_0

        # Line 3, calculate the random timesteps for the training batch.
        # Use importance sampling here if desired.
        t, weights = self._noise_scheduler.sample_random_times(
            batch_size=B, device=device
        )
        if self._noise_scheduler.continuous():
            # Add the logsnr to the context if we are continuous
            context["logsnr_t"] = self._noise_scheduler.logsnr(t)
        context["timestep"] = t

        # Line 4, sample from a Gaussian with mean 0 and unit variance.
        # This is the epsilon prediction target.
        epsilon = torch.randn_like(z_0)

        # Calculate forward process q_t
        x_t = self._noise_scheduler.q_sample(x_start=z_0, t=t, noise=epsilon)

        # If there is a mask in the context, then we need to only select
        # the elements of x_t that are unmasked, the others remain x_0.
        # A zero is the mask means use x_0, otherwise use x_t
        if "video_mask" in context:
            # video_mask is (B,T)
            x_t = torch.where(context["video_mask"][:, None, :, None, None], x_t, z_0)

        # Perform classifier free guidance over the context. This means
        # jointly train a conditional and unconditional model.
        if self._unconditional_guidance_probability > 0.0:
            # Merge the unconditional context with the conditional context
            unconditional_context = self._unconditional_context(context)
            conditional_context = context.copy()
            cfg_mask = prob_mask_like(
                shape=(B,), prob=self._unconditional_guidance_probability, device=device
            )

            for key in self._config.diffusion.classifier_free_guidance.signals:
                # Lists and tensors needs to be merged differently.
                conditional_context_signal = conditional_context[key]
                unconditional_context_signal = unconditional_context[key]

                if isinstance(conditional_context_signal, list):
                    assert len(conditional_context_signal) == B
                    assert len(unconditional_context_signal) == B

                    updated_context_signal = [
                        (
                            unconditional_context_signal[idx]
                            if cfg_mask[idx]
                            else conditional_context_signal[idx]
                        )
                        for idx in range(B)
                    ]
                else:
                    # The context is a tensor type
                    while len(cfg_mask.shape) != len(conditional_context_signal.shape):
                        cfg_mask = cfg_mask.unsqueeze(-1)

                    updated_context_signal = torch.where(
                        cfg_mask,
                        unconditional_context_signal,
                        conditional_context_signal,
                    )
                context[key] = updated_context_signal
        # Preprocess any of the context before it hits the score network.
        # For example, if we have prompts, then convert them
        # into text tokens or text embeddings.
        for preprocessor in self._context_preprocessors:
            context = preprocessor(context, device)

        # Process the input
        x_t = self._input_preprocessor(
            x=x_t, context=context, noise_scheduler=self._noise_scheduler
        )

        # Line 5, predict eps_theta given t. Add the two parameters we calculated
        # earlier together, and run the UNet with that input at time t.
        model_output = self._score_network(
            x_t,
            context=context,
        )

        if self._is_learned_sigma:
            # If we are a learned sigma model, the model is predicting
            # both epsilon, and the re-parameterized estimate of the variance.
            model_prediction, learned_variance = model_output
        else:
            model_prediction = model_output

        if self._prediction_type == PredictionType.EPSILON:
            prediction_target = epsilon
        elif self._prediction_type == PredictionType.V:
            prediction_target = self._noise_scheduler.predict_v_from_x_and_epsilon(
                x=z_0, epsilon=epsilon, t=t
            )
        elif self._prediction_type == PredictionType.RECTIFIED_FLOW:
            prediction_target = z_0 - epsilon
        else:
            raise NotImplemented(
                f"Prediction type {self._prediction_type} not implemented."
            )

        # Line 5, calculate MSE of epsilon, epsilon_theta (predicted_eps)
        mse_loss = torch.nn.functional.mse_loss(
            model_prediction, prediction_target, reduction="none"
        )
        mse_loss = reduce(mse_loss, "b ... -> b", "mean")

        # If we are a learned_variance model, calculate the variational bound term
        # to the total loss, using the estimated variance prediction.
        vb_loss = torch.zeros_like(mse_loss)
        if self._is_learned_sigma:
            # Stop the gradients from epsilon from flowing into the VB loss term
            frozen_out = [model_prediction.detach(), learned_variance]
            vb_loss = self._vb_bits_per_dim(
                epsilon_v_param=frozen_out,
                x_0=z_0,
                x_t=x_t,
                context=context,
                clip_denoised=False,
            )

            # Rescale the variational bound loss  by 1000 for equivalence with
            # initial implementation. Without a factor of 1/1000, the VB term
            # hurts the MSE term.
            lamb = 1e-3
            vb_loss *= lamb
        total_loss = mse_loss + vb_loss

        self._noise_scheduler.update_with_all_losses(t, total_loss.detach())
        total_loss = total_loss * weights

        end_time = time.perf_counter()
        latency = end_time - start_time

        return {
            "loss": total_loss.mean(),
            "mse_loss": mse_loss.mean(),
            "vb_loss": vb_loss.mean(),
        }

    def distillation_loss_on_batch(
        self,
        images,
        N,
        context: Dict,
        teacher_diffusion_model: Self,
    ) -> Dict:
        """Calculates the reverse process loss on a batch of images.

        Args:
            images: Tensor batch of images, of shape [B, C, H, W]
            prompts: List of prompts to use for conditioning, of length B
            clip_embedder: The CLIP model to use for image and text embeddings.
            low_resolution_images: Tensor batch of low resolution images, if this is
                part of a cascade.
            y: Tensor batch of class labels, if they exist.

        Returns:
            Dictionary of loss values, of which the "loss" entry will
            be the training loss.
        """
        B, _, H, W = images.shape
        device = images.device
        context = context.copy()
        assert self._latent_encoder is None

        # The images are normalized into the range (-1, 1),
        # from Section 3.3:
        # "We assume that image data consists of integers in {0, 1, . . . , 255} scaled linearly to [−1, 1]."
        x_0 = self._normalize(images)

        # Line 3, calculate the random timesteps for the training batch.
        # Use importance sampling here if desired.
        assert self._noise_scheduler.continuous()

        # Sample random times
        # t = i/N, i ∼ Cat[1, 2, . . . , N]
        t = torch.randint(
            0,
            N,
            (B,),
            device=device,
            dtype=torch.float32,
        )
        t = t / N

        # Add the logsnr to the context if we are continuous
        context["logsnr_t"] = self._noise_scheduler.logsnr(t)
        context["timestep"] = t

        # Line 4, sample from a Gaussian with mean 0 and unit variance.
        # This is the epsilon prediction target.
        epsilon = torch.randn_like(x_0)

        # Perform classifier free guidance over the context. This means
        # jointly train a conditional and unconditional model.
        if self._unconditional_guidance_probability > 0.0:
            # Merge the unconditional context with the conditional context
            unconditional_context = self._unconditional_context(context)
            conditional_context = context.copy()
            cfg_mask = prob_mask_like(
                shape=(B,), prob=self._unconditional_guidance_probability, device=device
            )

            for key in self._config.diffusion.classifier_free_guidance.signals:
                # Lists and tensors needs to be merged differently.
                conditional_context_signal = conditional_context[key]
                unconditional_context_signal = unconditional_context[key]

                if isinstance(conditional_context_signal, list):
                    assert len(conditional_context_signal) == B
                    assert len(unconditional_context_signal) == B

                    updated_context_signal = [
                        (
                            unconditional_context_signal[idx]
                            if cfg_mask[idx]
                            else conditional_context_signal[idx]
                        )
                        for idx in range(B)
                    ]
                else:
                    # The context is a tensor type
                    updated_context_signal = torch.where(
                        cfg_mask,
                        unconditional_context_signal,
                        conditional_context_signal,
                    )
                context[key] = updated_context_signal

        # Preprocess any of the context before it hits the score network.
        # For example, if we have prompts, then convert them
        # into text tokens or text embeddings.
        for preprocessor in self._context_preprocessors:
            context = preprocessor(context, device)

        # Process the input
        z_t = self._noise_scheduler.q_sample(x_start=x_0, t=t, noise=epsilon)
        z_t = self._input_preprocessor(
            x=z_t, context=context, noise_scheduler=self._noise_scheduler
        )

        # 2 steps of DDIM with teacher
        assert not self._is_learned_sigma, "Learned sigma not implemented yet."
        with torch.no_grad():
            teacher_score_z_t = teacher_diffusion_model._score_network(
                z_t, context=context
            )
            x_pred = self._noise_scheduler.predict_x_from_v(
                z=z_t, v=teacher_score_z_t, context=context
            )
            eps_pred = self._noise_scheduler.predict_epsilon_from_x(
                z=z_t, x=x_pred, context=context
            )

        u = t
        logsnr = self._noise_scheduler.logsnr(u)

        u_mid = u - 0.5 / N
        logsnr_mid = self._noise_scheduler.logsnr(u_mid)
        stdv_mid = broadcast_from_left(
            torch.sqrt(torch.nn.functional.sigmoid(-logsnr_mid)), shape=z_t.shape
        )
        a_mid = broadcast_from_left(
            torch.sqrt(torch.nn.functional.sigmoid(logsnr_mid)), shape=z_t.shape
        )
        z_mid = a_mid * x_pred + stdv_mid * eps_pred

        with torch.no_grad():
            context_tp = context.copy()
            context_tp["logsnr_t"] = logsnr_mid
            context_tp["timestep"] = u_mid
            teacher_score_z_tp = teacher_diffusion_model._score_network(
                z_mid, context=context_tp
            )
            x_pred = self._noise_scheduler.predict_x_from_v(
                z=z_t, v=teacher_score_z_tp, context=context
            )
            eps_pred = self._noise_scheduler.predict_epsilon_from_x(
                z=z_t, x=x_pred, context=context
            )

        u_s = u - 1.0 / N
        logsnr_s = self._noise_scheduler.logsnr(u_s)
        stdv_s = broadcast_from_left(
            torch.sqrt(torch.nn.functional.sigmoid(-logsnr_s)), shape=z_t.shape
        )
        a_s = broadcast_from_left(
            torch.sqrt(torch.nn.functional.sigmoid(logsnr_s)), shape=z_t.shape
        )
        z_teacher = a_s * x_pred + stdv_s * eps_pred

        # get x-target implied by z_teacher (!= x_pred)
        a_t = broadcast_from_left(
            torch.sqrt(torch.nn.functional.sigmoid(logsnr)), shape=z_t.shape
        )
        stdv_frac = broadcast_from_left(
            torch.exp(
                0.5
                * (
                    torch.nn.functional.softplus(logsnr)
                    - torch.nn.functional.softplus(logsnr_s)
                )
            ),
            shape=z_t.shape,
        )
        x_target = (z_teacher - stdv_frac * z_t) / (a_s - stdv_frac * a_t)
        x_target = torch.where(
            broadcast_from_left(t == 0, x_pred.shape), x_pred, x_target
        )
        eps_target = self._noise_scheduler.predict_epsilon_from_x(
            z=z_t, x=x_target, context=context
        )
        v_target = self._noise_scheduler.predict_v_from_x_and_epsilon(
            x=x_target, epsilon=eps_target, t=t
        )

        # denoising loss
        x_hat_score = self._score_network(z_t, context=context)

        model_x = self._noise_scheduler.predict_x_from_v(
            z=z_t, v=x_hat_score, context=context
        )
        model_eps = self._noise_scheduler.predict_epsilon_from_x(
            z=z_t, x=model_x, context=context
        )
        model_v = x_hat_score

        def meanflat(x):
            return x.mean(axis=tuple(range(1, len(x.shape))))

        x_mse = meanflat(torch.square(model_x - x_target))
        eps_mse = meanflat(torch.square(model_eps - eps_target))
        v_mse = meanflat(torch.square(model_v - v_target))

        mean_loss_weight_type = "snr"  # SNR+1 weighting
        if mean_loss_weight_type == "constant":  # constant weight on x_mse
            loss = x_mse
        elif mean_loss_weight_type == "snr":  # SNR * x_mse = eps_mse
            loss = eps_mse
        elif mean_loss_weight_type == "snr_trunc":  # x_mse * max(SNR, 1)
            loss = torch.maximum(x_mse, eps_mse)
        elif mean_loss_weight_type == "v_mse":
            loss = v_mse
        return {"loss": loss.mean()}

    def sample(
        self,
        context: Optional[Dict] = None,
        num_samples: int = 16,
        guidance_fn: Optional[Callable] = None,
        classifier_free_guidance: Optional[float] = None,
        num_sampling_steps: Optional[int] = None,
        sampler: Optional[ReverseProcessSampler] = None,
        initial_noise: Optional[torch.Tensor] = None,
        context_preprocessor: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Unconditionally/conditionally sample from the diffusion model.

        Args:
            image_embeddings: Tensor batch of image embeddings
            text_embedding: Tensor batch of text embeddings
            low_res_context: Tensor batch of low-res conditioning, if a cascade
            num_samples: The number of samples to generate
            guidance_fn: Classifier guidance function.
            classes: Tensor batch of class labels, if class conditional
            classifier_free_guidance: Optional classifier free guidance value.

        Returns:
            Tensor batch of samples from the model.
        """
        # The output shape of the data.
        s = self._config.diffusion.sampling.output_spatial_size
        output_spatial_size = [s[0], s[1]] if isinstance(s, list) else [s, s]

        if "output_frames" in self._config.diffusion.sampling.to_dict():
            shape = (
                num_samples,
                self._config.diffusion.sampling.output_channels,
                self._config.diffusion.sampling.output_frames,
                output_spatial_size[0],
                output_spatial_size[1],
            )
        else:
            shape = (
                num_samples,
                self._config.diffusion.sampling.output_channels,
                output_spatial_size[0],
                output_spatial_size[1],
            )
        device = next(self.parameters()).device
        self.eval()

        # Generate the unconditional context for classifier free guidance
        if classifier_free_guidance is not None:
            unconditional_context = self._unconditional_context(context)
            if context_preprocessor is not None:
                unconditional_context = context_preprocessor(
                    unconditional_context, device
                )
            for preprocessor in self._context_preprocessors:
                unconditional_context = preprocessor(unconditional_context, device)
        else:
            unconditional_context = None

        # If we are a super resolution model, then add the gaussian conditioning
        # sampling augmentation level if it exists
        if "super_resolution" in self._config and context is not None:
            if "sampling_augmentation_level" in self._config.super_resolution:
                context["augmentation_level"] = (
                    self._config.super_resolution.sampling_augmentation_level
                )

        # First do any sampling specific context preprocessors
        if context_preprocessor is not None:
            context = context_preprocessor(context, device)

        # Preprocess any of the context before it hits the score network.
        # For example, if we have prompts, then convert them
        # into text tokens or text embeddings.
        for preprocessor in self._context_preprocessors:
            context = preprocessor(context, device)

        sampling_steps = (
            num_sampling_steps
            if num_sampling_steps is not None
            else self._noise_scheduler.steps()
        )

        latent_samples, intermediate_outputs = self._p_sample_loop(
            shape,
            context=context,
            unconditional_context=unconditional_context,
            guidance_fn=guidance_fn,
            classifier_free_guidance=classifier_free_guidance,
            num_sampling_steps=sampling_steps,
            sampler=sampler,
            initial_noise=initial_noise,
        )
        latents = latent_samples

        # Decode the samples from the latent space
        if self._latent_encoder is not None:
            latents = self._latent_encoder.decode_from_latents(
                latents / self._latent_scale_factor
            )

        # And denormalize the samples from the latent space
        samples = self._unnormalize(latents)
        self.train()
        return samples, intermediate_outputs

    def get_classifier_guidance(
        self, classifier: torch.nn.Module, classifier_scale: float
    ):
        """Gets the function for calculating the classifier guidance.

        The classifier guidance function computes the gradient of a conditional
        log probability with respect to x. In particular, guidance_fn computes
        grad(log(p(y|x))), and we want to condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """

        def guidance_fn(x, t, y=None):
            assert y is not None
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return torch.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

        return guidance_fn

    @torch.no_grad()
    def print_model_summary(self):
        batch_size = 4
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        B = batch_size
        C = self._config.data.num_channels

        s = self._config.data.image_size
        image_size = [s[0], s[1]] if isinstance(s, list) else [s, s]

        H = image_size[0]
        W = image_size[1]
        is_video = (
            True if "input_number_of_frames" in self._config.data.to_dict() else False
        )
        F = (
            self._config.data.input_number_of_frames
            if "input_number_of_frames" in self._config.data.to_dict()
            else 1
        )

        summary_context = {
            "timestep": (
                torch.rand(size=(batch_size,), dtype=dtype, device=device)
                if self._noise_scheduler.continuous()
                else torch.randint(0, 10, size=(batch_size,), device=device)
            ),
            "logsnr_t": torch.rand(size=(batch_size,), dtype=dtype, device=device),
            "text_prompts": [""] * batch_size,
            "text_embeddings": torch.rand(
                size=(batch_size, 300, 2304), dtype=dtype, device=device
            ),
            "classes": torch.randint(
                0, self._config.data.num_classes, size=(batch_size,), device=device
            ),
            # Video specific context, ignored for image
            "x0": torch.zeros(B, C, F, H, W, dtype=dtype),
            "frame_indices": torch.tile(
                torch.arange(end=F, device=device)[None, ...],
                (B, 1),
            ),
            "observed_mask": torch.zeros(
                size=(B, C, F, 1, 1), dtype=dtype, device=device
            ),
            "latent_mask": torch.ones(size=(B, C, F, 1, 1), dtype=dtype, device=device),
            "video_mask": torch.ones(size=(B, F), dtype=torch.bool, device=device),
        }

        if "super_resolution" in self._config:
            summary_context[self._config.super_resolution.conditioning_key] = (
                torch.rand(
                    batch_size,
                    self._config.data.num_channels,
                    self._config.super_resolution.low_resolution_size,
                    self._config.super_resolution.low_resolution_size,
                    device=device,
                    dtype=dtype,
                )
            )
            summary_context["augmentation_timestep"] = torch.randint(
                0, 10, size=(batch_size,), device=device
            )

        # Preprocess the context
        for preprocessor in self._context_preprocessors:
            summary_context = preprocessor(summary_context, device=device)

        # Monkey path torch summary to deal with str inputs from text prompts
        s = self._config.diffusion.score_network.params.input_spatial_size
        input_spatial_size = [s[0], s[1]] if isinstance(s, list) else [s, s]
        fix_torchinfo_for_str()
        print(
            summary(
                self._score_network.to(device),
                input_data=[
                    (
                        torch.rand(
                            batch_size,
                            self._config.diffusion.score_network.params.input_channels,
                            self._config.diffusion.score_network.params.input_number_of_frames,
                            input_spatial_size[0],
                            input_spatial_size[1],
                            device=device,
                            dtype=dtype,
                        )
                        if is_video
                        else torch.rand(
                            batch_size,
                            self._config.diffusion.score_network.params.input_channels,
                            input_spatial_size[0],
                            input_spatial_size[1],
                            device=device,
                            dtype=dtype,
                        )
                    ),
                    summary_context,
                ],
            )
        )

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        # Load the state dict for the score network
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if hasattr(self._score_network, "load_model_weights"):
            state_dict = checkpoint["model_state_dict"]

            # Only preserve the score network keys
            score_network_state_pairs = []
            namespace = "_score_network."
            for k, v in state_dict.items():
                if k.startswith(namespace):
                    k = k[len(namespace) :]
                    score_network_state_pairs.append((k, v))
            self._score_network.load_model_weights(dict(score_network_state_pairs))
        else:
            missing_keys, unexpected_keys = self.load_state_dict(
                checkpoint["model_state_dict"], strict=strict
            )
            for k in missing_keys:
                assert "temporal" in k or "motion_module" in k, k

    def configure_optimizers(self, learning_rate: float) -> List[torch.optim.Optimizer]:

        if "optimizer" in self._config:
            return [
                instantiate_partial_from_config(self._config.optimizer.to_dict())(
                    self.parameters()
                )
            ]
        else:
            return [
                torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.99))
            ]

    def configure_learning_rate_schedule(
        self, optimizers: List[torch.optim.Optimizer]
    ) -> List[torch.optim.lr_scheduler._LRScheduler]:
        if "learning_rate_schedule" in self._config:
            return [
                get_constant_schedule_with_warmup(
                    optimizers[0],
                    **self._config.learning_rate_schedule.params.to_dict(),
                )
            ]
        else:
            return [
                get_constant_schedule_with_warmup(optimizers[0], num_warmup_steps=0)
            ]

    def process_input(self, x: torch.Tensor, context: Dict) -> torch.Tensor:
        return self._input_preprocessor(
            x=x, context=context, noise_scheduler=self._noise_scheduler
        )

    def predict_score(
        self, x: torch.Tensor, context: Dict
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._score_network(x, context=context)

    def is_learned_sigma(self) -> bool:
        return self._is_learned_sigma

    def noise_scheduler(self) -> NoiseScheduler:
        return self._noise_scheduler

    def classifier_free_guidance(self) -> float:
        return self._classifier_free_guidance

    def prediction_type(self) -> PredictionType:
        return self._prediction_type

    def _p_sample_loop(
        self,
        shape,
        context: Dict,
        unconditional_context: Optional[Dict],
        num_sampling_steps: int,
        guidance_fn=None,
        classifier_free_guidance: Optional[float] = None,
        sampler: Optional[ReverseProcessSampler] = None,
        initial_noise: Optional[torch.Tensor] = None,
        save_intermediate_outputs: bool = False,
    ):
        """Defines Algorithm 2 sampling using notation from DDPM implementation.

        Iteratively denoises (reverse diffusion) the probability density function p(x).
        Defines the unconditional sampling loop.

        Args:
            shape: The batched shape of the output images.
            low_res_context: Low resolution context, for cascaded models.
            guidance_fn: Optional guidance function using the gradients of a classifier
                to guide diffusion.
            classes: Tensor batch of class labels if they exist
            classifier_free_guidance: Classifier free guidance value
            image_embeddings: Tensor batch of image embeddings
            text_embeddngs: Tensor batch of text_embeddings

        Returns:
            Tensor batch of unconditional/conditional samples from the reverse distribution.
        """
        # Use the device that the current model is on.
        # Assumes all of the parameters are on the same device
        device = next(self.parameters()).device

        # Initial image is pure noise
        if initial_noise is not None:
            x_t = initial_noise
        else:
            x_t = torch.randn(shape, device=device)

        intermediate_outputs = []

        if save_intermediate_outputs:
            intermediate_outputs.append(self._unnormalize(x_t))

        sampler = sampler if sampler is not None else self._reverse_process_sampler
        for timestep_idx in tqdm(
            reversed(range(0, num_sampling_steps)),
            desc="sampling loop time step",
            total=num_sampling_steps,
            leave=False,
        ):
            # Some of the score network preprocessors can update the
            # context at each call, so we need to use a new dictionary
            # at each step to preserve the original context.
            context_for_timestep = context.copy()
            if unconditional_context is not None:
                unconditional_context_for_timestep = unconditional_context.copy()
            else:
                unconditional_context_for_timestep = None

            t = torch.tensor([timestep_idx] * shape[0], device=device)
            context_for_timestep["timestep_idx"] = timestep_idx
            if self._noise_scheduler.continuous():
                context_for_timestep["logsnr_s"] = self._noise_scheduler.logsnr(
                    t / num_sampling_steps
                )

                t_plus_1 = t + 1
                context_for_timestep["logsnr_t"] = self._noise_scheduler.logsnr(
                    t_plus_1 / num_sampling_steps
                )

                if unconditional_context_for_timestep is not None:
                    unconditional_context_for_timestep["timestep_idx"] = timestep_idx
                    unconditional_context_for_timestep["logsnr_s"] = (
                        self._noise_scheduler.logsnr(t / num_sampling_steps)
                    )
                    unconditional_context_for_timestep["logsnr_t"] = (
                        self._noise_scheduler.logsnr(t_plus_1 / num_sampling_steps)
                    )
                t = t / num_sampling_steps
            context_for_timestep["timestep"] = t

            if unconditional_context_for_timestep is not None:
                unconditional_context_for_timestep["timestep"] = t

            # If there is a video mask, make to sure add in the unmasked
            # frames from the original conditioning. A mask value of 1 means generate,
            # a mask value of 0 means use the conditioning
            if "video_mask" in context:
                assert "x0" in context
                x_t = torch.where(
                    context["video_mask"][:, None, :, None, None], x_t, context["x0"]
                )

            x_t_minus_1 = sampler.p_sample(
                x_t,
                context=context_for_timestep,
                unconditional_context=unconditional_context_for_timestep,
                diffusion_model=self,
                guidance_fn=guidance_fn,
                classifier_free_guidance=classifier_free_guidance,
            )
            x_t = x_t_minus_1

            if "video_mask" in context:
                x_t = torch.where(
                    context["video_mask"][:, None, :, None, None], x_t, context["x0"]
                )

            if save_intermediate_outputs:
                intermediate_outputs.append(self._unnormalize(x_t))

        return x_t, intermediate_outputs

    def _vb_bits_per_dim(
        self, x_0, x_t, context: Dict, epsilon_v_param, clip_denoised: bool
    ):
        """Get a term for the variational lower-bound.

        Used in the loss calculation for the estimate of the reverse process
        variance. The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        Args:
            x_0: Tensor batch of the distribution at time 0
            x_t: Tensor batch of the distribution at time t
            t: Tensor batch of the current timestep.
            low_res_context: Low resolution context, for cascaded models.
            y: Tensor batch of class labels if they exist
            clip_denoised: if True, clip the denoised signal into [-1, 1].
            epsilon_v_param: If not None, the reverse process model prediction
                at time t.
            image_embeddings: Tensor batch of image embeddings
            text_embeddngs: Tensor batch of text_embeddings

        Returns:
            Tensor batch of negative log likelihoods (for the first timestep)
            or KL divergence for subsequent timesteps.
        """
        true_mean, _, true_log_variance_clipped = self._noise_scheduler.q_posterior(
            x_start=x_0, x_t=x_t, context=context
        )

        # When calculating the variational lower bound, we disable
        # classifier free guidance.
        ancestral_sampler = AncestralSampler()
        mean, _, log_variance = ancestral_sampler.p_mean_variance(
            x_t,
            context=context,
            diffusion_model=self,
            unconditional_context=None,
            clip_denoised=clip_denoised,
            epsilon_v_param=epsilon_v_param,
            classifier_free_guidance=0.0,
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, mean, log_variance)

        # Take the mean over the non-batch dimensions
        kl = kl.mean(dim=list(range(1, len(kl.shape)))) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_0, means=mean, log_scales=0.5 * log_variance
        )
        assert decoder_nll.shape == x_0.shape
        decoder_nll = decoder_nll.mean(
            dim=list(range(1, len(decoder_nll.shape)))
        ) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((context["timestep"] == 0), decoder_nll, kl)
        return output
