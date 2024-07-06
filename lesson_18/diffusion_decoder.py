"""Diffusion Decoder from DaLL*E 2 (unCLIP).

Diffusion model which calculates P(x | zi, y) which produces a sample x from the
original distribution conditioned on CLIP image embeddings z_i and text captions y.

For the decoder, they use the GLIDE model (with classifier free guidance and text conditioning)
in learned sigma mode (v-param), with the text encoder from GLIDE (which appeared to just
be a BPE based text encoder).

We have retained the text conditioning from GLIDE, as mentioned in the DaLL*E 2 paper, even
though they should that it didn't help too much and the image embeddings were sufficient
for the conditioning.

In order to generate high resolution imagery, DaLL*E 2 uses a cascade of 1 base stage and 2
super resolution stages. The decoder here defines the base stage, and because the dataset
is so small, we leave the super resolution stages to future work. You can follow Lesson 12 -
Cascaded Diffusion Models, for an example of how to do it.
"""

from einops import reduce
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from typing import Callable, Dict, List, Optional

from clip import FrozenCLIPEmbedder
from importance_sampling import ImportanceSampler, UniformSampler
from score_network import MNistUnet
from utils import (
    discretized_gaussian_log_likelihood,
    normal_kl,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    DotConfig,
)
from diffusion_scheduler import NoiseScheduler


class GaussianDiffusion_DaLLE2_Decoder(torch.nn.Module):
    """Core GLIDE Diffusion algorithm, with classifier free guidance.

    This is the same core algorithm as DDPM, IDDPM, and Guided Diffusion,
    with a few changes listed above to support DaLL*E 2.
    """

    def __init__(self, config: DotConfig):
        super().__init__()
        self._score_network = MNistUnet(config)
        self._num_timesteps = config.model.num_scales
        self._is_v_param = config.model.is_v_param
        self._is_class_conditional = config.model.is_class_conditional
        self._num_classes = config.data.num_classes
        self._unconditional_guidance_probability = (
            config.model.unconditional_guidance_probability
        )
        self._classifier_free_guidance = config.model.classifier_free_guidance
        self._config = config
        self._noise_scheduler = NoiseScheduler(
            beta_schedule="cosine" if self._is_v_param else "linear",
            timesteps=self._num_timesteps,
            loss_type="l2",
        )

        if config.model.is_v_param:
            self._importance_sampler = ImportanceSampler(
                num_timesteps=config.model.num_scales
            )
        else:
            self._importance_sampler = UniformSampler(
                num_timesteps=config.model.num_scales
            )
        self._null_image_embeddings = torch.nn.Parameter(
            torch.randn(1, config.model.context_size)
        )
        self._null_text_embeddings = torch.nn.Parameter(
            torch.randn(1, config.model.context_size)
        )

    def loss_on_batch(
        self,
        images,
        prompts: List[str],
        clip_embedder: FrozenCLIPEmbedder,
        low_resolution_images: Optional[torch.Tensor] = None,
        y=None,
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

        # If we have a CLIP embedder, convert the images and text into embeddings.
        # The image and text embeddings are shape: (B, config.diffusion_decoder.context_size)
        image_embeddings, text_embeddings, _ = clip_embedder.encode(images, prompts)

        # The images are normalized into the range (-1, 1),
        # from Section 3.3:
        # "We assume that image data consists of integers in {0, 1, . . . , 255} scaled linearly to [−1, 1]."
        x_0 = normalize_to_neg_one_to_one(images)

        if low_resolution_images is not None:
            low_res_x_0 = normalize_to_neg_one_to_one(
                transforms.functional.resize(
                    low_resolution_images,
                    size=(
                        self._config.model.input_spatial_size,
                        self._config.model.input_spatial_size,
                    ),
                    antialias=True,
                )
            )

            # Apply gaussian conditioning augmentation if configured
            if self._config.model.gaussian_conditioning_augmentation:
                # Use non-truncating GCA. First sample s.
                s = torch.randint(0, self._num_timesteps, (B,), device=device).long()
                low_res_x_0 = self._noise_scheduler.q_sample(low_res_x_0, s)
        else:
            low_res_x_0 = None

        # Line 3, calculate the random timesteps for the training batch.
        # Use importance sampling here if desired.
        t, weights = self._importance_sampler.sample(batch_size=B, device=device)

        # Line 4, sample from a Gaussian with mean 0 and unit variance.
        # This is the epsilon prediction target.
        epsilon = torch.randn_like(x_0)

        # Calculate forward process q_t
        x_t = self._noise_scheduler.q_sample(x_start=x_0, t=t, noise=epsilon)

        # Drop some of the class labels so that we can perform unconditional
        # guidance.
        if y is not None:
            # Make room for the NULL token in the class labels
            conditional_y = y + 1
            # Create the NULL tokens
            unconditional_y = torch.zeros_like(y)
            # Sample the unconditional tokens
            unconditional_probability = torch.rand(size=y.shape, device=y.device)
            y = torch.where(
                unconditional_probability <= self._unconditional_guidance_probability,
                unconditional_y,
                conditional_y,
            )

        # Drop image and text embeddings to perform classifier free guidance.
        unconditional_probability = torch.rand(size=(B, 1), device=device)
        text_embeddings = torch.where(
            unconditional_probability <= self._unconditional_guidance_probability,
            self._null_text_embeddings,
            text_embeddings,
        )
        image_embeddings = torch.where(
            unconditional_probability <= self._unconditional_guidance_probability,
            self._null_image_embeddings,
            image_embeddings,
        )

        # Line 5, predict eps_theta given t. Add the two parameters we calculated
        # earlier together, and run the UNet with that input at time t.
        model_output = self._score_network(
            torch.cat([x_t, low_res_x_0], dim=1) if low_res_x_0 is not None else x_t,
            t=t,
            y=y,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
        )

        if self._is_v_param:
            # If we are a v-param model, the model is predicting
            # both epsilon, and the re-parameterized estimate of the variance.
            epsilon_theta, v_param = model_output
        else:
            epsilon_theta = model_output

        # Line 5, calculate MSE of epsilon, epsilon_theta (predicted_eps)
        mse_loss = torch.nn.functional.mse_loss(
            epsilon_theta, epsilon, reduction="none"
        )
        mse_loss = reduce(mse_loss, "b ... -> b", "mean")

        # If we are a v-param model, calculate the variational bound term
        # to the total loss, using the estimated variance prediction.
        vb_loss = torch.zeros_like(mse_loss)
        if self._is_v_param:
            # Stop the gradients from epsilon from flowing into the VB loss term
            frozen_out = [epsilon_theta.detach(), v_param]
            vb_loss = self._vb_bits_per_dim(
                epsilon_v_param=frozen_out,
                x_0=x_0,
                x_t=x_t,
                t=t,
                low_res_context=low_res_x_0,
                y=y,
                clip_denoised=False,
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
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
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        low_res_context: Optional[torch.Tensor] = None,
        num_samples: int = 16,
        guidance_fn: Optional[Callable] = None,
        classes: Optional[torch.Tensor] = None,
        classifier_free_guidance: Optional[float] = None,
    ):
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
            self._config.model.output_channels,
            self._config.model.input_spatial_size,
            self._config.model.input_spatial_size,
        )
        device = next(self.parameters()).device
        self.eval()

        # Generate latent samples
        # The additional context is the low resolution images
        if low_res_context is not None:
            low_res_x_0 = normalize_to_neg_one_to_one(
                transforms.functional.resize(
                    low_res_context,
                    size=(
                        self._config.model.input_spatial_size,
                        self._config.model.input_spatial_size,
                    ),
                    antialias=True,
                )
            )

            # Apply gaussian conditioning augmentation if configured
            if self._config.model.gaussian_conditioning_augmentation:
                # Use non-truncating GCA. First sample s.
                s = torch.randint(
                    0, self._num_timesteps, (num_samples,), device=device
                ).long()
                low_res_x_0 = self._noise_scheduler.q_sample(low_res_x_0, s)

        else:
            low_res_x_0 = None

        latent_samples = self._p_sample_loop(
            shape,
            low_res_x_0,
            guidance_fn,
            classes=classes,
            classifier_free_guidance=classifier_free_guidance,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )

        if self._is_class_conditional:
            latents, _ = latent_samples
        else:
            latents = latent_samples

        # Decode the samples from the latent space
        samples = unnormalize_to_zero_to_one(latents)
        self.train()
        return samples

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

    def _pred_epsilon(
        self,
        x,
        t,
        low_res_context: Optional[torch.Tensor],
        epsilon_v_param: Optional[List[torch.Tensor]] = None,
        y=None,
        image_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
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
        model_output = (
            self._score_network(
                (
                    torch.cat([x, low_res_context], dim=1)
                    if low_res_context is not None
                    else x
                ),
                t=t,
                y=y,
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
            )
            if epsilon_v_param is None
            else epsilon_v_param
        )

        if self._is_v_param:
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
                t, x.shape
            )
        return epsilon_theta, variance, log_variance

    def _p_mean_variance(
        self,
        x,
        t,
        low_res_context: Optional[torch.Tensor],
        clip_denoised=True,
        epsilon_v_param: Optional[List[torch.Tensor]] = None,
        y=None,
        classifier_free_guidance: Optional[float] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
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
        assert t.shape == (B,)

        epsilon_theta, variance, log_variance = self._pred_epsilon(
            x=x,
            t=t,
            low_res_context=low_res_context,
            epsilon_v_param=epsilon_v_param,
            y=y,
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
        )

        # If we are using classifier free guidance, then calculate the unconditional
        # epsilon as well.
        cfg = (
            classifier_free_guidance
            if classifier_free_guidance is not None
            else self._classifier_free_guidance
        )

        if cfg >= 0.0:
            assert image_embeddings is not None and text_embeddings is not None

            # Unconditionally sample the model
            uncond_epsilon_theta, uncond_variance, uncond_log_variance = (
                self._pred_epsilon(
                    x=x,
                    t=t,
                    low_res_context=low_res_context,
                    epsilon_v_param=epsilon_v_param,
                    image_embeddings=torch.zeros_like(image_embeddings),
                    text_embeddings=torch.zeros_like(text_embeddings),
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
        pred_xstart = _maybe_clip(
            self._noise_scheduler.predict_xstart_from_epsilon(
                x_t=x, t=t, epsilon=epsilon_theta
            )
        )
        # Set the mean of the reverse process equal to the mean of the forward process
        # posterior.
        model_mean, _, _ = self._noise_scheduler.q_posterior(
            x_start=pred_xstart, x_t=x, t=t
        )
        # print(pred_xstart, model_mean)
        return model_mean, variance, log_variance

    def _p_sample_loop(
        self,
        shape,
        low_res_context: Optional[torch.Tensor],
        guidance_fn=None,
        classes=None,
        classifier_free_guidance: Optional[float] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
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

        # If this is a class conditional model, setup the classes
        # to generate.
        y = None
        if self._is_class_conditional:
            if classes is None:
                # Use the NULL token for unconditional generation
                classes = torch.zeros(
                    size=(x_t.shape[0],), device=device, dtype=torch.int32
                )
                y = classes
            else:
                # Make room for the NULL token
                y = classes + 1

        for t in tqdm(
            reversed(range(0, self._num_timesteps)),
            desc="sampling loop time step",
            total=self._num_timesteps,
            leave=False,
        ):
            t = torch.tensor([t] * shape[0], device=device)
            x_t_minus_1 = self._p_sample(
                x_t,
                t=t,
                low_res_context=low_res_context,
                y=y,
                guidance_fn=guidance_fn,
                classifier_free_guidance=classifier_free_guidance,
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
            )
            x_t = x_t_minus_1

        if self._is_class_conditional:
            return x_t, classes
        return x_t

    def _p_sample(
        self,
        x,
        t,
        low_res_context: Optional[torch.Tensor],
        y,
        guidance_fn=None,
        classifier_free_guidance: Optional[float] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
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
                t=t,
                low_res_context=low_res_context,
                clip_denoised=True,
                y=y,
                classifier_free_guidance=classifier_free_guidance,
                image_embeddings=image_embeddings,
                text_embeddings=text_embeddings,
            )

        # No noise if t = 0
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        if guidance_fn is not None:
            model_mean = self._guidance_mean(
                guidance_fn, model_mean, model_variance, x, t, y
            )

        pred_img = (
            model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        )
        return pred_img

    def _vb_bits_per_dim(
        self,
        x_0,
        x_t,
        t,
        low_res_context: Optional[torch.Tensor],
        y,
        clip_denoised=True,
        epsilon_v_param: Optional[List[torch.Tensor]] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
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
            x_start=x_0, x_t=x_t, t=t
        )
        mean, _, log_variance = self._p_mean_variance(
            x_t,
            t,
            clip_denoised=clip_denoised,
            epsilon_v_param=epsilon_v_param,
            y=y,
            low_res_context=low_res_context,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
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
        output = torch.where((t == 0), decoder_nll, kl)
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
