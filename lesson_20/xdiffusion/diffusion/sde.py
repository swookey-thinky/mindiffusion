import torch
from torchinfo import summary
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple, Union

from xdiffusion.diffusion import DiffusionModel
from xdiffusion.sde.base import SDE
from xdiffusion.sde.vpsde import VPSDE
from xdiffusion.samplers.base import ReverseProcessSampler
from xdiffusion.utils import (
    DotConfig,
    get_constant_schedule_with_warmup,
    fix_torchinfo_for_str,
    instantiate_from_config,
    instantiate_partial_from_config,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)


class GaussianDiffusion_SDE(DiffusionModel):
    def __init__(self, config: DotConfig):
        super().__init__()
        self._config = config
        self._continuous = config.diffusion.continuous
        self._likelihood_weighting = False

        self._score_network = instantiate_from_config(
            config.diffusion.score_network, use_config_struct=True
        )

        self._input_preprocessor = instantiate_from_config(
            config.diffusion.input_preprocessing.to_dict()
        )

        self._context_preprocessors = torch.nn.ModuleList(
            [instantiate_from_config(c) for c in config.diffusion.context_preprocessing]
        )

        # Get the SDE associated with this diffusion model, if it exists.
        assert "sde" in config.diffusion.to_dict()
        self._sde: SDE = instantiate_from_config(config.diffusion.sde.to_dict())

        # Instantiate the sampler
        self._sampler = instantiate_partial_from_config(
            config.diffusion.sampling, use_config_struct=True
        )(sde=self._sde, diffusion_model=self)

    def models(self) -> List[DiffusionModel]:
        return [self]

    def sde(self) -> Optional[SDE]:
        """Gets the SDE associated with this diffusion model, if it exists."""
        return self._sde

    def config(self) -> DotConfig:
        return self._config

    def print_model_summary(self):
        batch_size = 4
        device = "cuda"

        B = batch_size
        C = self._config.data.num_channels
        H = W = self._config.data.image_size
        is_video = (
            True if "input_number_of_frames" in self._config.data.to_dict() else False
        )
        F = (
            self._config.data.input_number_of_frames
            if "input_number_of_frames" in self._config.data.to_dict()
            else 1
        )

        summary_context = {
            "timestep": (torch.rand(size=(batch_size,), device=device)),
            "text_prompts": [""] * batch_size,
            "classes": torch.randint(
                0, self._config.data.num_classes, size=(batch_size,), device=device
            ),
            # Video specific context, ignored for image
            "x0": torch.zeros(B, C, F, H, W),
            "frame_indices": torch.tile(
                torch.arange(end=F, device=device)[None, ...],
                (B, 1),
            ),
            "observed_mask": torch.zeros(
                size=(B, C, F, 1, 1), dtype=torch.float32, device=device
            ),
            "latent_mask": torch.ones(
                size=(B, C, F, 1, 1), dtype=torch.float32, device=device
            ),
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
                )
            )
            summary_context["augmentation_timestep"] = torch.randint(
                0, 10, size=(batch_size,), device=device
            )

        # Preprocess the context
        for preprocessor in self._context_preprocessors:
            summary_context = preprocessor(summary_context, device=device)

        # Monkey path torch summary to deal with str inputs from text prompts
        fix_torchinfo_for_str()
        summary(
            self._score_network.to(device),
            input_data=[
                (
                    torch.rand(
                        batch_size,
                        self._config.diffusion.score_network.params.input_channels,
                        self._config.diffusion.score_network.params.input_number_of_frames,
                        self._config.diffusion.score_network.params.input_spatial_size,
                        self._config.diffusion.score_network.params.input_spatial_size,
                        device=device,
                    )
                    if is_video
                    else torch.rand(
                        batch_size,
                        self._config.diffusion.score_network.params.input_channels,
                        self._config.diffusion.score_network.params.input_spatial_size,
                        self._config.diffusion.score_network.params.input_spatial_size,
                        device=device,
                    )
                ),
                summary_context,
            ],
        )

    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        # Load the state dict for the score network
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
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
                assert "temporal" in k, k

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

    def predict_score(
        self, x: torch.Tensor, context: Dict
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Predict the score on the input. The timestep when predicting
        # the score is always [0,1]
        context = context.copy()
        t = context["timestep"]
        assert torch.min(t) >= 0.0
        assert torch.max(t) <= 1.0

        if self._continuous:
            # For VP-trained models, t=0 corresponds to the lowest noise level
            # The maximum value of time embedding is assumed to 999 for
            # continuously-trained models.
            labels = t * 999
            context["timestep"] = labels
            score = self._score_network(
                x,
                context=context,
            )
            std = self._sde.marginal_prob(torch.zeros_like(x), t)[1]
        else:
            # For VP-trained models, t=0 corresponds to the lowest noise level
            assert isinstance(self._sde, VPSDE)
            labels = t * (self._sde.N - 1)
            context["timestep"] = labels
            score = self._score_network(
                x,
                context=context,
            )
            std = self._sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]
        score = -score / std[:, None, None, None]
        return score

    def loss_on_batch(self, images: torch.Tensor, context: Dict) -> Dict:
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
        B = images.shape[0]
        device = images.device
        context = context.copy()
        eps = 1e-5

        # The images are normalized into the range (-1, 1).
        x_0 = normalize_to_neg_one_to_one(images)

        # Calculate the random timesteps for the training batch.
        t = torch.rand(x_0.shape[0], device=device) * (self._sde.T - eps) + eps
        context["timestep"] = t

        # Setup z, the "brownian motion" W from the SDE.
        z = torch.randn_like(x_0)

        # Perturb the input data for the loss.
        mean, std = self._sde.marginal_prob(x_0, t)
        x_t = mean + std[:, None, None, None] * z

        # Preprocess any of the context before it hits the score network.
        # For example, if we have prompts, then convert them
        # into text tokens or text embeddings.
        for preprocessor in self._context_preprocessors:
            context = preprocessor(context, device)

        # Process the input
        x_t = self._input_preprocessor(x=x_t, context=context, noise_scheduler=None)

        score = self.predict_score(x_t, context)
        if not self._likelihood_weighting:
            # This is the denoising score matching objective from https://arxiv.org/abs/1907.05600,
            # Eq. 5 with λ(σi) = σ^2
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = self._sde.sde(torch.zeros_like(x_0), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        loss = torch.mean(losses)
        return {"loss": loss}

    def sample(
        self,
        context: Optional[Dict] = None,
        num_samples: int = 16,
        guidance_fn: Optional[Callable] = None,
        classifier_free_guidance: Optional[float] = None,
        num_sampling_steps: Optional[int] = None,
        sampler: Optional[ReverseProcessSampler] = None,
        initial_noise: Optional[torch.Tensor] = None,
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
        shape = (
            num_samples,
            self._config.diffusion.sampling.output_channels,
            self._config.diffusion.sampling.output_spatial_size,
            self._config.diffusion.sampling.output_spatial_size,
        )
        device = next(self.parameters()).device
        self.eval()

        with torch.no_grad():
            # Initial sample
            eps = 1e-3
            x = self._sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(self._sde.T, eps, self._sde.N, device=device)

            for i in tqdm(range(self._sde.N), leave=False):
                t = timesteps[i]
                context = {"timestep": torch.ones(shape[0], device=t.device) * t}

                if i == (self._sde.N - 1):
                    context["denoise_final"] = True
                else:
                    context["denoise_final"] = False

                x = self._sampler.p_sample(
                    x,
                    context,
                    unconditional_context=None,
                    diffusion_model=self,
                )

        samples = unnormalize_to_zero_to_one(x)
        self.train()
        return samples, None
