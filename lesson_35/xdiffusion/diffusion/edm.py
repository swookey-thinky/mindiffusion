"""Diffusion formulation from EDM.

"Elucidating the Design Space of Diffusion-Based Generative Models"
(https://arxiv.org/abs/2206.00364).

Based on code from: https://github.com/NVlabs/edm/tree/main
"""

import copy
import torch
from torchinfo import summary
from typing import Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

from xdiffusion.diffusion import DiffusionModel, PredictionType
from xdiffusion.samplers.base import ReverseProcessSampler
from xdiffusion.scheduler import NoiseScheduler
from xdiffusion.sde.base import SDE
from xdiffusion.utils import (
    DotConfig,
    fix_torchinfo_for_str,
    instantiate_from_config,
    instantiate_partial_from_config,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    get_constant_schedule_with_warmup,
)


class GaussianDiffusion_EDM(DiffusionModel):

    def __init__(self, config: DotConfig):
        super().__init__()
        self._config = config
        # Instantiate the score network
        self._score_network = instantiate_from_config(
            config.diffusion.score_network.to_dict()
        )
        self._context_preprocessors = []
        self._loss = instantiate_from_config(config.diffusion.loss.to_dict())
        self._augment_pipeline = None
        self._sampler = instantiate_from_config(config.diffusion.sampling.to_dict())

    def loss_on_batch(self, images: torch.Tensor, context: Dict) -> Dict:
        # Normalize images
        x = normalize_to_neg_one_to_one(images)
        labels = context["labels"] if "labels" in context else None
        loss = self._loss(
            net=self._score_network,
            images=x,
            labels=labels,
            augment_pipe=self._augment_pipeline,
        )
        return {"loss": loss.mean()}

    def sample(
        self,
        context: Optional[Dict] = None,
        num_samples: int = 16,
        guidance_fn: Optional[Callable] = None,
        classifier_free_guidance: Optional[float] = None,
        sampler: Optional[ReverseProcessSampler] = None,
        num_sampling_steps: Optional[int] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # The output shape of the data.
        if "output_frames" in self._config.diffusion.sampling.to_dict():
            shape = (
                num_samples,
                self._config.diffusion.sampling.output_channels,
                self._config.diffusion.sampling.output_frames,
                self._config.diffusion.sampling.output_spatial_size,
                self._config.diffusion.sampling.output_spatial_size,
            )
        else:
            shape = (
                num_samples,
                self._config.diffusion.sampling.output_channels,
                self._config.diffusion.sampling.output_spatial_size,
                self._config.diffusion.sampling.output_spatial_size,
            )
        device = next(self.parameters()).device
        self.eval()

        x_t = torch.randn(shape, device=device)
        x_0 = self._sampler.p_sample_loop(
            diffusion_model=self, latents=x_t, class_labels=None
        )
        self.train()
        return unnormalize_to_zero_to_one(x_0), None

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
            "logsnr_t": torch.rand(size=(batch_size,), device=device),
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
                        self._config.diffusion.score_network.params.img_channels,
                        self._config.diffusion.score_network.params.input_number_of_frames,
                        self._config.diffusion.score_network.params.img_resolution,
                        self._config.diffusion.score_network.params.img_resolution,
                        device=device,
                    )
                    if is_video
                    else torch.rand(
                        batch_size,
                        self._config.diffusion.score_network.params.img_channels,
                        self._config.diffusion.score_network.params.img_resolution,
                        self._config.diffusion.score_network.params.img_resolution,
                        device=device,
                    )
                ),
                summary_context["logsnr_t"],
            ],
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

    def models(self) -> List[DiffusionModel]:
        return [self]

    def update_ema(
        self, step: int, total_steps: int, scale_fn: Callable[[int], Tuple[float, int]]
    ):
        # EMA not supported yet
        return

    def config(self) -> DotConfig:
        return self._config

    def process_input(self, x: torch.Tensor, context: Dict) -> torch.Tensor:
        raise NotImplementedError()

    def predict_score(
        self, x: torch.Tensor, context: Dict
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Assumes context has "t", which for edm models is batch of sigmas.
        return self._score_network(x, context["t"])

    def is_learned_sigma(self) -> bool:
        raise NotImplementedError()

    def noise_scheduler(self) -> NoiseScheduler:
        raise NotImplementedError()

    def classifier_free_guidance(self) -> float:
        raise NotImplementedError()

    def prediction_type(self) -> PredictionType:
        raise NotImplementedError()

    def sde(self) -> Optional[SDE]:
        raise NotImplementedError()


class VPLoss:
    """
    Loss function corresponding to the variance preserving (VP) formulation
    from the paper "Score-Based Generative Modeling through Stochastic
    Differential Equations".
    """

    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


class VELoss:
    """
    Loss function corresponding to the variance exploding (VE) formulation
    from the paper "Score-Based Generative Modeling through Stochastic
    Differential Equations".
    """

    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLoss:
    """
    Improved loss function proposed in the paper "Elucidating the Design Space
    of Diffusion-Based Generative Models" (EDM).
    """

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
