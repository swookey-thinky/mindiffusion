"""Forward and reverse diffusion processes.

This package implements the forward and reverse diffusion processes from
the paper "Diffusion Models Beat GANs on Image Synthesis"
(https://arxiv.org/abs/2105.05233).

Based on authors original implementation here:
https://github.com/openai/guided-diffusion/tree/main

This contains several improvements over the original DDPM paper and the authors
previous work Improved DDPM.

1.) Slightly different score network architecture:
    a.) Multi-resolution attention
    b.) BigGAN residual blocks for up and down sampling
    c.) Increasing the number of attention heads
2.) Class conditioning built into the model
3.) Classifier guided sampling
4.) Adaptive group normalization at the timestep embedding stage
"""

from einops import reduce
import numpy as np
import torch
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Type

from importance_sampling import ImportanceSampler, UniformSampler
from utils import (
    cosine_beta_schedule,
    discretized_gaussian_log_likelihood,
    extract,
    linear_beta_schedule,
    normal_kl,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    DotConfig,
)


class GaussianDiffusion_GuidedDiffusion(torch.nn.Module):
    """Core Guided Diffusion DDPM algorithm.

    Implements both DDPM and Improved/Guided DDPM training and sampling algorithms.
    """

    def __init__(self, score_network_type: Type, config: DotConfig):
        super().__init__()
        self._score_network = score_network_type(config)
        self._num_timesteps = config.model.num_scales
        self._is_v_param = config.model.is_v_param
        self._is_class_conditional = config.model.is_class_conditional
        self._num_classes = config.data.num_classes

        if config.model.is_v_param:
            self._importance_sampler = ImportanceSampler(
                num_timesteps=config.model.num_scales
            )
        else:
            self._importance_sampler = UniformSampler(
                num_timesteps=config.model.num_scales
            )

        # Helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        # Fixed linear schedule for epsilon Beta
        if self._is_v_param:
            betas = cosine_beta_schedule(self._num_timesteps).to(torch.float32)
        else:
            betas = linear_beta_schedule(self._num_timesteps).to(torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.nn.functional.pad(
            alphas_cumprod[:-1], (1, 0), value=1.0
        )

        # The first parameter is the square root of the cumulative product
        # of alpha at time t, times x0.
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

        # The second parameter is the square root of 1 minus the cumulative product
        # of alpha at time t, times epsilon.
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        register_buffer("betas", betas)
        register_buffer("alphas", alphas)
        register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recip_alphas_cumprod_minus_one", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        register_buffer("posterior_variance", posterior_variance)
        register_buffer(
            "posterior_log_variance_clipped", posterior_log_variance_clipped
        )
        register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    def loss_on_batch(self, images, y=None) -> Dict:
        """Calculates the reverse process loss on a batch of images.

        Args:
            image: Tensor batch of images, of shape [B, C, H, W]
            y: Class labels, if they exist.

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

        # Line 4, sample from a Gaussian with mean 0 and unit variance.
        # This is the epsilon prediction target.
        epsilon = torch.randn_like(x_0)

        # Calculate forward process q_t
        x_t = self._q_sample(x_0=x_0, t=t, noise=epsilon)

        # Line 5, predict eps_theta given t. Add the two parameters we calculated
        # earlier together, and run the UNet with that input at time t.
        model_output = self._score_network(x_t, t, y)
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
                y=y,
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
        image_size: int,
        num_channels: int,
        batch_size: int = 16,
        guidance_fn: Optional[Callable] = None,
        classes: Optional[torch.Tensor] = None,
    ):
        """Unconditionally/conditionally sample from the diffusion model."""
        # The output shape of the data.
        shape = (batch_size, num_channels, image_size, image_size)
        self.eval()
        samples = self._p_sample_loop(shape, guidance_fn, classes=classes)
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

    def _q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ):
        """Forward process for DDPM.

        Noise the initial sample x_0 to the timestep t, calculating $q(x_t | x_0)$.

        Args:
            x_0: Tensor batch of original samples at time 0
            t: Tensor batch of timesteps to noise to.
            noise: Optional fixed noise to add.

        Returns:
            Tensor batch of noised samples at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    def _q_posterior_mean_variance(self, x_0, x_t, t):
        """Compute the mean and variance of the diffusion posterior.

        Calculates $q(x_{t-1} | x_t, x_0)$

        Args:
            x_0: The initial starting state (or predicted starting state) of the distribution.
            x_t: The distribution at time t.
            t: The timestep to calculate the posterior.

        Returns:
            Tuple of:
                mean: Tensor batch of the mean of the posterior
                variance: Tensor batch of the variance of the posterior
                log_variance: Tensor batch of the log of the posterior variance, clipped.
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_0.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _p_mean_variance(
        self,
        x,
        t,
        clip_denoised=True,
        epsilon_v_param: Optional[List[torch.Tensor]] = None,
        y=None,
    ):
        """Calculates the mean and the variance of the reverse process distribution.

        Applies the model to get $p(x_{t-1} | x_t)$, an estimate of the reverse
        diffusion process.

        Args:
            x: Tensor batch of the distribution at time t.
            t: Tensor batch of timesteps.
            clip_denoised: if True, clip the denoised signal into [-1, 1].
            epsilon_v_param: If not None, a tensor batch of the model output at t.
                Used to freeze the epsilon path to prevent gradients flowing back during
                VLB loss calculation.

        Returns:
            A tuple of the following values:
                mean: Tensor batch of the reverse distribution mean.
                variance: Tensor batch of the reverse distribution variance.
                log_variance: Tensor batch of the log of the reverse distribution variance.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = (
            self._score_network(x, t, y) if epsilon_v_param is None else epsilon_v_param
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

            variance = extract(variance, t, x.shape)
            log_variance = extract(log_variance, t, x.shape)

        _maybe_clip = lambda x_: (torch.clamp(x_, -1.0, 1.0) if clip_denoised else x_)

        # Epsilon prediction (mean reparameterization)
        pred_xstart = _maybe_clip(
            self._predict_start_from_eps(x_t=x, t=t, epsilon=epsilon_theta)
        )

        # Set the mean of the reverse process equal to the mean of the forward process
        # posterior.
        model_mean, _, _ = self._q_posterior_mean_variance(x_0=pred_xstart, x_t=x, t=t)
        return model_mean, variance, log_variance

    def _p_sample_loop(self, shape, guidance_fn=None, classes=None):
        """Defines Algorithm 2 sampling using notation from DDPM implementation.

        Iteratively denoises (reverse diffusion) the probability density function p(x).
        Defines the unconditional sampling loop.

        Args:
            shape: The batched shape of the output images.
            guidance_fn: Optional guidance function using the gradients of a classifier
                to guide diffusion.

        Returns:
            Tensor batch of unconditional samples from the reverse distribution.
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
                classes = torch.randint(
                    low=0,
                    high=self._num_classes,
                    size=(x_t.shape[0],),
                    device=device,
                )
            y = classes

        for t in tqdm(
            reversed(range(0, self._num_timesteps)),
            desc="sampling loop time step",
            total=self._num_timesteps,
            leave=False,
        ):
            t = torch.tensor([t] * shape[0], device=device)
            x_t_minus_1 = self._p_sample(x_t, t, y, guidance_fn)
            x_t = x_t_minus_1

        ret = unnormalize_to_zero_to_one(x_t)

        if self._is_class_conditional:
            return ret, classes
        return ret

    def _p_sample(self, x, t, y, guidance_fn=None):
        """Reverse process single step.

        Samples x_{t-1} given x_t - the joint distribution p_theta(x_{t-1}|x_t).

        Args:
            x: Tensor batch of the distribution at time t.
            t: Tensor batch of the current timestep.
            y: Tensor batch of optional class labels.
            guidance_fn: Optional guidance function using the gradients of a classifier
                to guide diffusion.


        Returns:
            Tensor batch of the distribution at timestep t-1.
        """
        B, _, _, _ = x.shape
        device = x.device

        with torch.no_grad():
            model_mean, model_variance, model_log_variance = self._p_mean_variance(
                x=x, t=t, clip_denoised=True, y=y
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

    def _predict_start_from_eps(self, x_t, t, epsilon):
        """Predict the original image from the re-parameterized mean, epsilon.

        Original implementation here: https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py#L170

        Args:
            x_t: Tensor batch of the distribution at time t.
            t: Tensor batch of the current timestep.
            epsilon: Tensor batch of the re-parameterized estimate of the mean at time t

        Returns:
            Tensor batch of the predicted x_0 given epsilon.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recip_alphas_cumprod_minus_one, t, x_t.shape) * epsilon
        )

    def _vb_bits_per_dim(
        self,
        x_0,
        x_t,
        t,
        y,
        clip_denoised=True,
        epsilon_v_param: Optional[List[torch.Tensor]] = None,
    ):
        """Get a term for the variational lower-bound.

        Used in the loss calculation for the estimate of the reverse process
        variance. The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        Args:
            x_0: Tensor batch of the distribution at time 0
            x_t: Tensor batch of the distribution at time t
            t: Tensor batch of the current timestep.
            clip_denoised: if True, clip the denoised signal into [-1, 1].
            epsilon_v_param: If not None, the reverse process model prediction
                at time t.
        Returns:
            Tensor batch of negative log likelihoods (for the first timestep)
            or KL divergence for subsequent timesteps.
        """
        true_mean, _, true_log_variance_clipped = self._q_posterior_mean_variance(
            x_0=x_0, x_t=x_t, t=t
        )
        mean, _, log_variance = self._p_mean_variance(
            x_t, t, clip_denoised=clip_denoised, epsilon_v_param=epsilon_v_param, y=y
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
        """
        gradient = guidance_fn(x, t, y)
        new_mean = p_mean.float() + p_var * gradient.float()
        return new_mean
