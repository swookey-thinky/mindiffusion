"""Diffusion Prior from DaLL*E 2 (unCLIP).

Diffusion model which calculates P(z_i | y), a CLIP image embedding z_i given a
text caption y.

The diffusion prior is a decoder-only Transformer based network with a
causal attention mask on a sequence consisting of, in order:
the encoded text, the CLIP text embedding, an embedding for the diffusion timestep,
the noised CLIP image embedding, and a final embedding whose output from the Transformer is used to
predict the unnoised CLIP image embedding. Instead of an epsilon prediction network the
authors found it better to predict the unnoised z_i directly.

I'm unsure why we need both the text encoding AND the text embedding, but that's
what the paper mentions so we replicate it here. I suspect this is redundant, and later
research has better approaches to text conditioning (and DaLL*E 3 changes the architecture
again), so we don't attempt to ablate it here. It's probably overkill for this dataset
anyways.
"""

from einops import repeat
from einops.layers.torch import Rearrange
import torch
from tqdm import tqdm
from typing import List, Optional

from clip import FrozenCLIPEmbedder
from diffusion_scheduler import NoiseScheduler
from transformer import Transformer
from utils import DotConfig


class GaussianDiffusion_DaLLE2_Prior(torch.nn.Module):
    """Diffusion based prior network for DaLL*E 2.

    This is the same core algorithm as DDPM, IDDPM, and Guided Diffusion,
    with a few changes listed above.
    """

    def __init__(self, config: DotConfig):
        super().__init__()

        self._config = config

        # Create the transformer that is used to calculate the unnoised
        # CLIP image embeddings
        self._score_network = Transformer(
            context_size=config.diffusion_prior.model.transformer.context_size,
            attention_channels=config.diffusion_prior.model.transformer.width,
            layers=config.diffusion_prior.model.transformer.num_layers,
            attention_heads=config.diffusion_prior.model.transformer.attention_heads,
            is_causal=True,
        )

        # Create the time, text, and image embeddings for the transformer
        self._text_embeddings = torch.nn.Sequential(
            (
                torch.nn.Linear(
                    config.diffusion_prior.model.transformer.context_size,
                    config.diffusion_prior.model.transformer.context_size
                    * config.diffusion_prior.model.num_text_embeddings,
                )
                if config.diffusion_prior.model.num_text_embeddings > 1
                else torch.nn.Identity()
            ),
            Rearrange(
                "b (n d) -> b n d", n=config.diffusion_prior.model.num_text_embeddings
            ),
        )
        self._time_embeddings = torch.nn.Sequential(
            torch.nn.Embedding(
                config.diffusion_prior.model.num_timesteps,
                config.diffusion_prior.model.transformer.context_size
                * config.diffusion_prior.model.num_time_embeddings,
            ),
            Rearrange(
                "b (n d) -> b n d", n=config.diffusion_prior.model.num_time_embeddings
            ),
        )

        self._image_embeddings = torch.nn.Sequential(
            (
                torch.nn.Linear(
                    config.diffusion_prior.model.transformer.context_size,
                    config.diffusion_prior.model.transformer.context_size
                    * config.diffusion_prior.model.num_image_embeddings,
                )
                if config.diffusion_prior.model.num_image_embeddings > 1
                else torch.nn.Identity()
            ),
            Rearrange(
                "b (n d) -> b n d", n=config.diffusion_prior.model.num_image_embeddings
            ),
        )

        # A final embedding whose output from the Transformer is used to
        # predict the unnoised CLIP image embedding.
        self._learned_query = torch.nn.Parameter(
            torch.randn(config.diffusion_prior.model.transformer.context_size)
        )

        # The noise scheduler to use for diffusion
        self._noise_scheduler = NoiseScheduler(
            beta_schedule="cosine",
            timesteps=config.diffusion_prior.model.num_timesteps,
            loss_type="l2",
        )

    def loss_on_batch(
        self,
        images: torch.Tensor,
        prompts: List[str],
        clip_embedder: FrozenCLIPEmbedder,
    ):
        """Calculates the loss on a batch of data.

        Args:
            images: Tensor batch of image data
            prompts: List of prompts, length(prompts) == images.shape[0]
            clip_embedder: A CLIP model to calculate text and image embeddings.

        Returns:
            A dictionary of loss values, one of which will be keyed by "loss"
            and used for the optimizer step.
        """
        B, C, H, W = images.shape
        device = images.device

        # Convert the prompts and images to the target image/text embeddings
        with torch.no_grad():
            target_image_embeddings, target_text_embeddings, target_text_encodings = (
                clip_embedder.encode(images, prompts)
            )
            target_image_embeddings = target_image_embeddings.to(device)
            target_text_encodings = target_text_encodings.to(device)
            target_text_embeddings = target_text_embeddings.to(device)

        # Create random training timesteps
        timesteps = self._noise_scheduler.sample_random_times(batch=B)

        text_embeddings = self._text_embeddings(target_text_embeddings)
        time_embeddings = self._time_embeddings(timesteps)
        learned_queries = repeat(self._learned_query, "d -> b 1 d", b=B)

        # Noise the image embeddings according to the timesteps
        image_embeddings = target_image_embeddings
        noisy_image_embeddings = self._noise_scheduler.q_sample(
            x_start=image_embeddings, t=timesteps
        )
        noisy_image_embeddings = self._image_embeddings(noisy_image_embeddings)

        # in section 2.2, last paragraph
        # "... consisting of encoded text, CLIP text embedding, diffusion timestep embedding,
        # noised CLIP image embedding, final embedding for prediction"
        tokens = torch.cat(
            (
                target_text_encodings,
                text_embeddings,
                time_embeddings,
                noisy_image_embeddings,
                learned_queries,
            ),
            dim=-2,
        )

        # Predict the next tokens in the transformer
        transformer_out = self._score_network(tokens)
        predicted_image_embeddings = transformer_out[..., -1, :]

        loss = self._noise_scheduler.loss_fn(
            predicted_image_embeddings, target_image_embeddings
        )
        return {"loss": loss}

    def sample(
        self,
        prompts: List[str],
        num_samples: int = 16,
        clip_embedder: Optional[FrozenCLIPEmbedder] = None,
    ):
        """Unconditionally/conditionally sample from the diffusion model.

        Args:
            prompts: List of prompts, length(prompts) == num_samples
            num_samples: The number of samples to generate.
            clip_embedder: A CLIP model to calculate text embeddings.

        Returns:
            Tensor batch of predicted image embeddings.
        """
        # The output shape of the data. This is the shape of the
        # image embeddings
        shape = (
            num_samples,
            self._config.diffusion_prior.model.transformer.context_size,
        )
        device = next(self.parameters()).device
        self.eval()

        with torch.no_grad():
            text_embeddings, text_encodings = clip_embedder.encode_text(prompts)
            text_encodings = text_encodings.to(device)
            text_embeddings = text_embeddings.to(device)

            image_embeddings = self._p_sample_loop(
                shape,
                text_embeddings=text_embeddings,
                text_encodings=text_encodings,
            )
        self.train()
        return image_embeddings, text_embeddings

    def _p_sample_loop(
        self,
        shape,
        text_embeddings: torch.Tensor,
        text_encodings: torch.Tensor,
    ):
        """Defines Algorithm 2 sampling using notation from DDPM implementation.

        Iteratively denoises (reverse diffusion) the probability density function p(x).
        Defines the unconditional sampling loop.

        Args:
            shape: The batched shape of the output images.
            text_embeddings: Tensor batch of text embeddings.
            text_encodings: Tensor batch of text encoding.

        Returns:
            Tensor batch of unconditional samples from the reverse distribution.
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
            t = torch.tensor([t] * shape[0], device=device)
            x_t_minus_1 = self._p_sample(
                x_t,
                t=t,
                text_embeddings=text_embeddings,
                text_encodings=text_encodings,
            )
            x_t = x_t_minus_1

        return x_t

    def _p_sample(
        self,
        x,
        t,
        text_embeddings: torch.Tensor,
        text_encodings: torch.Tensor,
    ):
        """Reverse process single step.

        Samples x_{t-1} given x_t - the joint distribution p_theta(x_{t-1}|x_t).

        Args:
            x: Tensor batch of the distribution at time t.
            t: Tensor batch of the current timestep.
            text_embeddings: Tensor batch of text embeddings.
            text_encodings: Tensor batch of text encoding.

        Returns:
            Tensor batch of the distribution at timestep t-1.
        """
        with torch.no_grad():
            model_mean, _, model_log_variance = self._p_mean_variance(
                x=x,
                t=t,
                text_embeddings=text_embeddings,
                text_encodings=text_encodings,
            )

        # No noise if t = 0
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))

        pred_img = (
            model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        )
        return pred_img

    def _p_mean_variance(self, x, t, text_embeddings, text_encodings):
        """Calculates the mean/variance of the reverse process distribution.

        Args:
            x: Tensor batch of the distribution at time t.
            t: Tensor batch of the current timestep.
            text_embeddings: Tensor batch of text embeddings.
            text_encodings: Tensor batch of text encoding.

        Returns:
            Tuple of mean, variance, and log_variance of the reverse
            distribution at timestep t-1.
        """
        text_embeddings = self._text_embeddings(text_embeddings)
        time_embeddings = self._time_embeddings(t)
        learned_queries = repeat(self._learned_query, "d -> b 1 d", b=x.shape[0])

        # Noise the image embeddings according to the timesteps
        image_embeddings = self._image_embeddings(x)

        # in section 2.2, last paragraph
        # "... consisting of encoded text, CLIP text embedding, diffusion timestep embedding,
        # noised CLIP image embedding, final embedding for prediction"
        tokens = torch.cat(
            (
                text_encodings,
                text_embeddings,
                time_embeddings,
                image_embeddings,
                learned_queries,
            ),
            dim=-2,
        )

        # Predict the next tokens in the transformer
        transformer_out = self._score_network(tokens)
        predicted_image_embeddings = transformer_out[..., -1, :]

        model_mean, posterior_variance, posterior_log_variance = (
            self._noise_scheduler.q_posterior(
                x_start=predicted_image_embeddings, x_t=x, t=t
            )
        )
        return model_mean, posterior_variance, posterior_log_variance
