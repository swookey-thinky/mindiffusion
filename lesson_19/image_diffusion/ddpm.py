"""DDPM Based Diffusion Model.

Base implementation of a DDPM diffusion model. Support the following improvements
from different papers:

"""

from einops import reduce
import numpy as np
import torch
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple

from image_diffusion.diffusion import DiffusionModel
from image_diffusion.scheduler import NoiseScheduler
from image_diffusion.utils import (
    discretized_gaussian_log_likelihood,
    instantiate_from_config,
    instantiate_partial_from_config,
    normal_kl,
    normalize_to_neg_one_to_one,
    prob_mask_like,
    unnormalize_to_zero_to_one,
    DotConfig,
)


class GaussianDiffusion_DDPM(DiffusionModel):
    """Core DDPM Diffusion algorithm, with classifier free guidance.

    This is the same core algorithm as DDPM, IDDPM, and Guided Diffusion,
    with a few changes listed above to support DaLL*E 2.
    """

    def __init__(self, config: DotConfig):
        super().__init__()
        self._score_network = instantiate_from_config(
            config.diffusion.score_network, use_config_struct=True
        )

        self._config = config
        self._is_learned_sigma = config.diffusion.score_network.params.is_learned_sigma
        self._is_class_conditional = (
            config.diffusion.score_network.params.is_class_conditional
        )
        self._num_classes = config.data.num_classes
        self._unconditional_guidance_probability = (
            config.diffusion.classifier_free_guidance.unconditional_guidance_probability
        )
        self._classifier_free_guidance = (
            config.diffusion.classifier_free_guidance.classifier_free_guidance
        )
        self._noise_scheduler = NoiseScheduler(
            beta_schedule=config.diffusion.noise_scheduler.schedule,
            timesteps=config.diffusion.noise_scheduler.num_scales,
            loss_type=config.diffusion.noise_scheduler.loss_type,
        )
        self._importance_sampler = instantiate_from_config(
            config.diffusion.noise_scheduler.sampler.to_dict()
        )

        # TODO: Add the full list
        self._context_preprocessor = instantiate_from_config(
            config.diffusion.context_preprocessing[0]
        )
        self._input_preprocessor = instantiate_from_config(
            config.diffusion.input_preprocessing.to_dict()
        )
        self._unconditional_context = instantiate_from_config(
            config.diffusion.classifier_free_guidance.unconditional_context.to_dict()
        )

    def models(self) -> List[DiffusionModel]:
        return [self]

    def config(self) -> DotConfig:
        return self._config

    def loss_on_batch(
        self,
        images,
        context: Dict,
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

        # The images are normalized into the range (-1, 1),
        # from Section 3.3:
        # "We assume that image data consists of integers in {0, 1, . . . , 255} scaled linearly to [−1, 1]."
        x_0 = normalize_to_neg_one_to_one(images)

        # Line 3, calculate the random timesteps for the training batch.
        # Use importance sampling here if desired.
        t, weights = self._importance_sampler.sample(batch_size=B, device=device)
        context["timestep"] = t

        # Line 4, sample from a Gaussian with mean 0 and unit variance.
        # This is the epsilon prediction target.
        epsilon = torch.randn_like(x_0)

        # Calculate forward process q_t
        x_t = self._noise_scheduler.q_sample(x_start=x_0, t=t, noise=epsilon)

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
        context = self._context_preprocessor(context, device)

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
            epsilon_theta, learned_variance = model_output
        else:
            epsilon_theta = model_output

        # Line 5, calculate MSE of epsilon, epsilon_theta (predicted_eps)
        mse_loss = torch.nn.functional.mse_loss(
            epsilon_theta, epsilon, reduction="none"
        )
        mse_loss = reduce(mse_loss, "b ... -> b", "mean")

        # If we are a learned_variance model, calculate the variational bound term
        # to the total loss, using the estimated variance prediction.
        vb_loss = torch.zeros_like(mse_loss)
        if self._is_learned_sigma:
            # Stop the gradients from epsilon from flowing into the VB loss term
            frozen_out = [epsilon_theta.detach(), learned_variance]
            vb_loss = self._vb_bits_per_dim(
                epsilon_v_param=frozen_out,
                x_0=x_0,
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

        self._importance_sampler.update_with_all_losses(t, total_loss.detach())
        total_loss = total_loss * weights

        return {
            "loss": total_loss.mean(),
            "mse_loss": mse_loss.mean(),
            "vb_loss": vb_loss.mean(),
        }

    def sample(
        self,
        context: Optional[Dict] = None,
        num_samples: int = 16,
        guidance_fn: Optional[Callable] = None,
        classifier_free_guidance: Optional[float] = None,
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
        shape = (
            num_samples,
            self._config.diffusion.sampling.params.output_channels,
            self._config.diffusion.sampling.params.output_spatial_size,
            self._config.diffusion.sampling.params.output_spatial_size,
        )
        device = next(self.parameters()).device
        self.eval()

        # Generate the unconditional context for classifier free guidance
        if classifier_free_guidance is not None:
            unconditional_context = self._unconditional_context(context)
            unconditional_context = self._context_preprocessor(
                unconditional_context, device
            )
        else:
            unconditional_context = None

        # If we are a super resolution model, then add the gaussian conditioning
        # sampling augmentation level if it exists
        if "super_resolution" in self._config and context is not None:
            if "sampling_augmentation_level" in self._config.super_resolution:
                context["sampling_augmentation_level"] = (
                    self._config.super_resolution.sampling_augmentation_level
                )

        # Preprocess any of the context before it hits the score network.
        # For example, if we have prompts, then convert them
        # into text tokens or text embeddings.
        context = self._context_preprocessor(context, device)

        latent_samples = self._p_sample_loop(
            shape,
            context=context,
            unconditional_context=unconditional_context,
            guidance_fn=guidance_fn,
            classifier_free_guidance=classifier_free_guidance,
        )

        if self._is_class_conditional:
            latents, _ = latent_samples
        else:
            latents = latent_samples

        # Decode the samples from the latent space
        samples = unnormalize_to_zero_to_one(latents)
        self.train()
        return samples, None

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

    def print_model_summary(self):
        batch_size = 128

        summary_context = {
            "timestep": torch.randint(0, 10, size=(batch_size,)),
            "text_prompts": [""] * batch_size,
        }

        if "super_resolution" in self._config:
            summary_context[self._config.super_resolution.conditioning_key] = (
                torch.rand(
                    batch_size,
                    self._config.data.num_channels,
                    self._config.super_resolution.low_resolution_spatial_size,
                    self._config.super_resolution.low_resolution_spatial_size,
                )
            )
            summary_context["augmentation_timestep"] = torch.randint(
                0, 10, size=(batch_size,)
            )

        # Preprocess the context
        summary_context = self._context_preprocessor(summary_context, device="cpu")

        # Remove the text prompts since they cause issues with torch summary.
        del summary_context["text_prompts"]
        summary(
            self._score_network,
            input_data=[
                torch.rand(
                    batch_size,
                    self._config.diffusion.score_network.params.input_channels,
                    self._config.diffusion.score_network.params.input_spatial_size,
                    self._config.diffusion.score_network.params.input_spatial_size,
                ),
                summary_context,
            ],
        )

    def configure_optimizers(self, learning_rate: float) -> List[torch.optim.Optimizer]:
        return [
            torch.optim.Adam(self.parameters(), lr=learning_rate, betas=(0.9, 0.99))
        ]

    def _pred_epsilon(
        self,
        x,
        context: Dict,
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
        x = self._input_preprocessor(
            x=x, context=context, noise_scheduler=self._noise_scheduler
        )

        model_output = (
            self._score_network(x, context=context)
            if epsilon_v_param is None
            else epsilon_v_param
        )

        if self._is_learned_sigma:
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
            variance, log_variance = self._noise_scheduler.variance_fixed_large(
                context["timestep"], x.shape
            )
        return epsilon_theta, variance, log_variance

    def _p_mean_variance(
        self,
        x,
        context: Dict,
        unconditional_context: Optional[Dict],
        clip_denoised=True,
        epsilon_v_param: Optional[List[torch.Tensor]] = None,
        classifier_free_guidance: Optional[float] = None,
    ):
        """Calculates the mean and the variance of the reverse process distribution.

        Applies the model to get $p(x_{t-1} | x_t)$, an estimate of the reverse
        diffusion process.

        Args:
            x: Tensor batch of the distribution at time t.
            t: Tensor batch of timesteps.
            low_res_context: Low resolution context, for cascaded models.
            clip_denoised: if True, clip the denoised signal into [-1, 1].
            epsilon_v_param: If not None, a tensor batch of the model output at t.
                Used to freeze the epsilon path to prevent gradients flowing back during
                VLB loss calculation.
            y: Tensor batch of class labels if they exist
            classifier_free_guidance: Classifier free guidance value
            image_embeddings: Tensor batch of image embeddings
            text_embeddngs: Tensor batch of text_embeddings

        Returns:
            A tuple of the following values:
                mean: Tensor batch of the reverse distribution mean.
                variance: Tensor batch of the reverse distribution variance.
                log_variance: Tensor batch of the log of the reverse distribution variance.
        """
        B, C = x.shape[:2]

        assert context["timestep"].shape == (B,)

        epsilon_theta, variance, log_variance = self._pred_epsilon(
            x=x,
            context=context,
            epsilon_v_param=epsilon_v_param,
        )

        # If we are using classifier free guidance, then calculate the unconditional
        # epsilon as well.
        cfg = (
            classifier_free_guidance
            if classifier_free_guidance is not None
            else self._classifier_free_guidance
        )

        if cfg >= 0.0 and unconditional_context is not None:
            # Unconditionally sample the model
            uncond_epsilon_theta, uncond_variance, uncond_log_variance = (
                self._pred_epsilon(
                    x=x,
                    context=unconditional_context,
                    epsilon_v_param=epsilon_v_param,
                )
            )
            w = cfg
            epsilon_theta = uncond_epsilon_theta + w * (
                epsilon_theta - uncond_epsilon_theta
            )

            # It's not clear from the paper how to guide v-param models, but we will treat
            # them the same as the epsilon-param models
            variance = uncond_variance + w * (variance - uncond_variance)
            log_variance = uncond_log_variance + w * (
                log_variance - uncond_log_variance
            )

        _maybe_clip = lambda x_: (torch.clamp(x_, -1.0, 1.0) if clip_denoised else x_)

        # Epsilon prediction (mean reparameterization)
        pred_xstart = self._noise_scheduler.predict_xstart_from_epsilon(
            x_t=x, t=context["timestep"], epsilon=epsilon_theta
        )

        if (
            "dynamic_thresholding" in self._config.diffusion
            and self._config.diffusion.dynamic_thresholding.enable
        ):
            if clip_denoised:
                pred_xstart = self._dynamic_thresholding(
                    pred_xstart,
                    p=self._config.diffusion.dynamic_thresholding.p,
                    c=self._config.diffusion.dynamic_thresholding.c,
                )
        else:
            pred_xstart = _maybe_clip(pred_xstart)

            # Set the mean of the reverse process equal to the mean of the forward process
        # posterior.
        model_mean, _, _ = self._noise_scheduler.q_posterior(
            x_start=pred_xstart, x_t=x, t=context["timestep"]
        )
        return model_mean, variance, log_variance

    def _p_sample_loop(
        self,
        shape,
        context: Dict,
        unconditional_context: Optional[Dict],
        guidance_fn=None,
        classifier_free_guidance: Optional[float] = None,
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
        x_t = torch.randn(shape, device=device)

        for t in tqdm(
            reversed(range(0, self._noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self._noise_scheduler.num_timesteps,
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

            t = torch.tensor([t] * shape[0], device=device)
            context_for_timestep["timestep"] = t
            if unconditional_context_for_timestep is not None:
                unconditional_context_for_timestep["timestep"] = t

            x_t_minus_1 = self._p_sample(
                x_t,
                context=context_for_timestep,
                unconditional_context=unconditional_context_for_timestep,
                guidance_fn=guidance_fn,
                classifier_free_guidance=classifier_free_guidance,
            )
            x_t = x_t_minus_1

        return x_t

    def _p_sample(
        self,
        x,
        context: Dict,
        unconditional_context: Optional[Dict],
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
            image_embeddings: Tensor batch of image embeddings
            text_embeddngs: Tensor batch of text_embeddings

        Returns:
            Tensor batch of the distribution at timestep t-1.
        """
        B, _, _, _ = x.shape
        device = x.device

        with torch.no_grad():
            model_mean, model_variance, model_log_variance = self._p_mean_variance(
                x=x,
                context=context,
                unconditional_context=unconditional_context,
                clip_denoised=True,
                classifier_free_guidance=classifier_free_guidance,
            )

        # No noise if t = 0
        t = context["timestep"]
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        if guidance_fn is not None:
            model_mean = self._guidance_mean(
                guidance_fn, model_mean, model_variance, x, t, context["classes"]
            )

        pred_img = (
            model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        )
        return pred_img

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
            x_start=x_0, x_t=x_t, t=context["timestep"]
        )

        # When calculating the variational lower bound, we disable
        # classifier free guidance. Not sure what the expected
        # behavior should be though, since this is during training not inference.
        mean, _, log_variance = self._p_mean_variance(
            x_t,
            context=context,
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

    def _guidance_mean(self, guidance_fn, p_mean, p_var, x, t, y):
        """Classifier guidance for the mean estimate.

        Compute the mean for the previous step, given a function guidance_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, guidance_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).

        Args:
            guidance_fn: Computes the gradient of a conditional log probability with
                respect to x
            p_mean: The predicted mean of the distribution
            p_var: The predicted variance of the distribution
            x: Distribution at x
            t: Timestep
            y: Class labels

        Returns:
            The mean of the distribution guided by the gradient.
        """
        gradient = guidance_fn(x, t, y)
        new_mean = p_mean.float() + p_var * gradient.float()
        return new_mean

    def _dynamic_thresholding(self, x, p=0.995, c=1.7):
        """
        Dynamic thresholding, a diffusion sampling technique from Imagen (https://arxiv.org/abs/2205.11487)
        to leverage high guidance weights and generating more photorealistic and detailed images
        than previously was possible based on x.clamp(-1, 1) vanilla clipping or static thresholding

        p — percentile determine relative value for clipping threshold for dynamic compression,
            helps prevent oversaturation recommend values [0.96 — 0.99]

        c — absolute hard clipping of value for clipping threshold for dynamic compression,
            helps prevent undersaturation and low contrast issues; recommend values [1.5 — 2.]
        """
        x_shapes = x.shape
        s = torch.quantile(x.abs().reshape(x_shapes[0], -1), p, dim=-1)
        s = torch.clamp(s, min=1, max=c)
        x_compressed = torch.clip(x.reshape(x_shapes[0], -1).T, -s, s) / s
        x_compressed = x_compressed.T.reshape(x_shapes)
        return x_compressed
