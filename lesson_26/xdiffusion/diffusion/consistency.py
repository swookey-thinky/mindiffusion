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
    append_dims,
    fix_torchinfo_for_str,
    instantiate_from_config,
    instantiate_partial_from_config,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    get_constant_schedule_with_warmup,
    mean_flat,
)
from xdiffusion.layers.ema import update_ema, create_ema_and_scales_fn


class GaussianDiffusion_ConsistencyModel(DiffusionModel):

    def __init__(self, config: DotConfig):
        super().__init__()
        self._config = config

        # Instantiate the score network
        self._score_network = instantiate_from_config(
            config.diffusion.score_network.to_dict()
        )

        # Initialize the target network, which is the EMA updated
        # version of the score network.
        self._target_model = instantiate_from_config(
            config.diffusion.score_network.to_dict()
        )
        self._target_model.requires_grad_(False)
        self._target_model.train()

        # Initialize the target model parameters from the
        # the score network
        for dst, src in zip(
            self._target_model.parameters(), self._score_network.parameters()
        ):
            dst.data.copy_(src.data)

        # Create the ema rate function for the target network. Will be created on
        # first run.
        self._target_ema_rate = None

        # Create the ema model for the score network
        if "exponential_moving_average" in config.diffusion:
            self._score_network_ema = instantiate_from_config(
                config.diffusion.score_network.to_dict()
            )
            # Initialize the target model parameters from the
            # the score network
            for dst, src in zip(
                self._score_network_ema.parameters(), self._score_network.parameters()
            ):
                dst.data.copy_(src.data)
        else:
            self._score_network_ema = None

        self._context_preprocessors = []
        self._loss: ConsistencyTrainingLoss = instantiate_from_config(
            config.diffusion.loss.to_dict()
        )
        self._augment_pipeline = None
        self._sampler = instantiate_from_config(config.diffusion.sampling.to_dict())

    def loss_on_batch(self, images: torch.Tensor, context: Dict) -> Dict:
        step = context["step"]
        total_steps = context["total_steps"]

        # Normalize images
        x = normalize_to_neg_one_to_one(images)
        labels = context["labels"] if "labels" in context else None

        # From Section 5, "we propose to progressively increase N during training according to a
        # schedule function...". The num_scales result from the ema function does this.
        if self._target_ema_rate is None:
            self._target_ema_rate = create_ema_and_scales_fn(
                target_ema_mode=self._config.diffusion.consistency_model.target_ema.target_ema_mode,
                start_ema=self._config.diffusion.consistency_model.target_ema.start_ema,
                scale_mode=self._config.diffusion.consistency_model.target_ema.scale_mode,
                start_scales=self._config.diffusion.consistency_model.target_ema.start_scales,
                end_scales=self._config.diffusion.consistency_model.target_ema.end_scales,
                total_steps=total_steps,
                distill_steps_per_iter=(
                    self._config.diffusion.consistency_model.target_ema.distill_steps_per_iter
                    if "distill_steps_per_iter"
                    in self._config.diffusion.consistency_model.target_ema
                    else 0
                ),
            )

        _, num_scales = self._target_ema_rate(step)

        loss = self._loss(
            score_network=self._score_network,
            target_network=self._target_model,
            images=x,
            num_scales=num_scales,
            labels=labels,
            augment_pipe=self._augment_pipeline,
        )
        return {"loss": loss.mean()}

    def distillation_loss_on_batch(
        self,
        images: torch.Tensor,
        N: int,
        context: Dict,
        teacher_diffusion_model: DiffusionModel,
    ) -> Dict:
        step = context["step"]
        total_steps = context["total_steps"]

        # Normalize images
        x = normalize_to_neg_one_to_one(images)
        labels = context["labels"] if "labels" in context else None

        # From Section 5, "we propose to progressively increase N during training according to a
        # schedule function...". The num_scales result from the ema function does this.
        if self._target_ema_rate is None:
            self._target_ema_rate = create_ema_and_scales_fn(
                total_steps=total_steps,
                distill_steps_per_iter=(
                    self._config.diffusion.consistency_model.target_ema.distill_steps_per_iter
                    if "distill_steps_per_iter"
                    in self._config.diffusion.consistency_model.target_ema
                    else 0
                ),
                **self._config.diffusion.consistency_model.target_ema.to_dict(),
            )

        _, num_scales = self._target_ema_rate(step)

        loss = self._loss(
            score_network=self._score_network,
            target_network=self._target_model,
            teacher_model=teacher_diffusion_model,
            images=x,
            num_scales=num_scales,
            labels=labels,
            augment_pipe=self._augment_pipeline,
        )
        return {"loss": loss.mean()}

    def update_ema(
        self, step: int, total_steps: int, scale_fn: Callable[[int], Tuple[float, int]]
    ):
        # First update the EMA model
        ema_rate, _ = scale_fn(step)
        if self._score_network_ema is not None:
            update_ema(
                self._score_network_ema.parameters(),
                self._score_network.parameters(),
                rate=ema_rate,
            )

        # Update the target model ema
        if self._target_ema_rate is None:
            self._target_ema_rate = create_ema_and_scales_fn(
                target_ema_mode=self._config.diffusion.consistency_model.target_ema.target_ema_mode,
                start_ema=self._config.diffusion.consistency_model.target_ema.start_ema,
                scale_mode=self._config.diffusion.consistency_model.target_ema.scale_mode,
                start_scales=self._config.diffusion.consistency_model.target_ema.start_scales,
                end_scales=self._config.diffusion.consistency_model.target_ema.end_scales,
                total_steps=total_steps,
                distill_steps_per_iter=(
                    self._config.diffusion.consistency_model.target_ema.distill_steps_per_iter
                    if "distill_steps_per_iter"
                    in self._config.diffusion.consistency_model.target_ema
                    else 0
                ),
            )

        target_ema, _ = self._target_ema_rate(step)
        with torch.no_grad():
            update_ema(
                self._target_model.parameters(),
                self._score_network.parameters(),
                rate=target_ema,
            )

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
        if sampler is not None:
            x_0 = sampler.p_sample_loop(
                diffusion_model=self, latents=x_t, class_labels=None
            )
        else:
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

    def models(self) -> List[DiffusionModel]:
        return [self]

    def config(self) -> DotConfig:
        return self._config

    def process_input(self, x: torch.Tensor, context: Dict) -> torch.Tensor:
        raise NotImplementedError()

    def predict_score(
        self, x: torch.Tensor, context: Dict
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Assumes context has "t", which for consistency models is batch of sigmas.
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


class ConsistencyTrainingLoss:
    """Consistency Training Loss

    Equation 10. from https://arxiv.org/abs/2303.01469
    """

    def __init__(
        self,
        sigma_data=0.5,
        rho: float = 7.0,
        loss_norm: str = "lpips",
        weight_schedule: str = "uniform",
    ):
        self.sigma_data = sigma_data
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.loss_norm = loss_norm

    def __call__(
        self,
        score_network,
        target_network,
        images,
        num_scales,
        labels=None,
        augment_pipe=None,
    ):
        x_start = images
        noise = torch.randn_like(x_start)
        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(score_network, x, t)

        @torch.no_grad()
        def target_denoise_fn(x, t):
            return self.denoise(target_network, x, t)

        @torch.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            denoiser = x0
            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            return samples

        indices = torch.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = score_network.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            score_network.sigma_min ** (1 / self.rho)
            - score_network.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = score_network.sigma_max ** (1 / self.rho) + (indices + 1) / (
            num_scales - 1
        ) * (
            score_network.sigma_min ** (1 / self.rho)
            - score_network.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        distiller = denoise_fn(x_t, t)
        x_t2 = euler_solver(x_t, t, t2, x_start).detach()
        distiller_target = target_denoise_fn(x_t2, t2)
        distiller_target = distiller_target.detach()
        snrs = t**-2
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        if self.loss_norm == "l1":
            diffs = torch.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2-32":
            distiller = torch.nn.functional.interpolate(
                distiller, size=32, mode="bilinear"
            )
            distiller_target = torch.nn.functional.interpolate(
                distiller_target,
                size=32,
                mode="bilinear",
            )
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = torch.nn.functional.interpolate(
                    distiller, size=224, mode="bilinear"
                )
                distiller_target = torch.nn.functional.interpolate(
                    distiller_target, size=224, mode="bilinear"
                )

            loss = (
                self.lpips_loss(
                    (distiller + 1) / 2.0,
                    (distiller_target + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")
        return loss

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        return model(x_t, sigmas)


class ConsistencyDistillationLoss:
    """Consistency Distillation Loss

    Equation 7. from https://arxiv.org/abs/2303.01469
    """

    def __init__(
        self,
        sigma_data=0.5,
        rho: float = 7.0,
        loss_norm: str = "lpips",
        weight_schedule: str = "uniform",
    ):
        self.sigma_data = sigma_data
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.loss_norm = loss_norm

    def __call__(
        self,
        score_network,
        target_network,
        teacher_model: DiffusionModel,
        images,
        num_scales,
        labels=None,
        augment_pipe=None,
    ):
        x_start = images
        noise = torch.randn_like(x_start)
        dims = x_start.ndim

        def denoise_fn(x, t):
            return self.denoise(score_network, x, t)

        @torch.no_grad()
        def target_denoise_fn(x, t):
            return self.denoise(target_network, x, t)

        @torch.no_grad()
        def teacher_denoise_fn(x, t):
            context = {"t": t}
            return teacher_model.predict_score(x, context)

        @torch.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        indices = torch.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        t = score_network.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            score_network.sigma_min ** (1 / self.rho)
            - score_network.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        t2 = score_network.sigma_max ** (1 / self.rho) + (indices + 1) / (
            num_scales - 1
        ) * (
            score_network.sigma_min ** (1 / self.rho)
            - score_network.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        distiller = denoise_fn(x_t, t)
        x_t2 = heun_solver(x_t, t, t2, x_start).detach()

        distiller_target = target_denoise_fn(x_t2, t2)
        distiller_target = distiller_target.detach()
        snrs = t**-2
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        if self.loss_norm == "l1":
            diffs = torch.abs(distiller - distiller_target)
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2-32":
            distiller = torch.nn.functional.interpolate(
                distiller, size=32, mode="bilinear"
            )
            distiller_target = torch.nn.functional.interpolate(
                distiller_target,
                size=32,
                mode="bilinear",
            )
            diffs = (distiller - distiller_target) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if x_start.shape[-1] < 256:
                distiller = torch.nn.functional.interpolate(
                    distiller, size=224, mode="bilinear"
                )
                distiller_target = torch.nn.functional.interpolate(
                    distiller_target, size=224, mode="bilinear"
                )

            loss = (
                self.lpips_loss(
                    (distiller + 1) / 2.0,
                    (distiller_target + 1) / 2.0,
                )
                * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")
        return loss

    def denoise(self, model, x_t, sigmas, **model_kwargs):
        return model(x_t, sigmas)


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings
