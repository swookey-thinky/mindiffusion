import torch
from torchvision import transforms
from typing import Dict

from xdiffusion.utils import normalize_to_neg_one_to_one
from xdiffusion.scheduler import NoiseScheduler
from xdiffusion.layers.embedding import TimestepEmbeddingProjection


class InputPreprocessor(torch.nn.Module):
    """Prepares the low-resolution imagery for input concatenation."""

    def __init__(
        self,
        low_resolution_size: int,
        super_resolution_size: int,
        context_input_key: str,
        apply_gaussian_conditioning_augmentation: bool,
        is_spatial: bool = True,
        is_temporal: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._super_resolution_size = super_resolution_size
        self._low_resolution_size = low_resolution_size
        self._context_input_key = context_input_key
        self._apply_gaussian_conditioning_augmentation = (
            apply_gaussian_conditioning_augmentation
        )
        self._is_spatial = is_spatial
        self._is_temporal = is_temporal
        assert is_temporal ^ is_spatial

        if "temporal_upsampling" in kwargs:
            assert kwargs["temporal_upsampling"].startswith("frameskip")
            self._temporal_upsampling_skip = int(
                kwargs["temporal_upsampling"].split("_")[1]
            )
        elif self._is_temporal:
            assert self._super_resolution_size % self._low_resolution_size == 0
            self._temporal_upsampling_skip = (
                self._super_resolution_size // self._low_resolution_size
            )

    def forward(
        self, x: torch.Tensor, context: Dict, noise_scheduler: NoiseScheduler, **kwargs
    ):
        low_resolution_images = context[self._context_input_key]

        if len(low_resolution_images.shape) == 4:
            # We are spatial only, since this is a batch of images
            assert self._is_spatial
            B, C, H, W = low_resolution_images.shape
        else:
            B, C, F, H, W = low_resolution_images.shape

        if self._is_spatial:
            assert (
                B == x.shape[0]
                and H == self._low_resolution_size
                and W == self._low_resolution_size
            )
        else:
            assert self._is_temporal
            assert B == x.shape[0] and F == self._low_resolution_size

        # Upsample the low resolution imagery to the model output size.
        if self._is_spatial:
            low_res_x_0 = normalize_to_neg_one_to_one(
                transforms.v2.functional.resize(
                    low_resolution_images,
                    size=(
                        self._super_resolution_size,
                        self._super_resolution_size,
                    ),
                    antialias=True,
                )
            )
        else:
            # Temporal super-resolution is performed by "repeating frames or by filling in blank frames".
            # Here, we choose to repeat frames.
            assert self._super_resolution_size % self._low_resolution_size == 0
            low_res_x_0 = normalize_to_neg_one_to_one(
                torch.repeat_interleave(
                    low_resolution_images,
                    repeats=self._temporal_upsampling_skip,
                    dim=2,
                )[:, :, : self._super_resolution_size, :, :]
            )

        # Apply gaussian conditioning augmentation if configured
        if self._apply_gaussian_conditioning_augmentation:
            # Use non-truncating GCA. First sample s.
            if "augmentation_level" in context:
                if noise_scheduler.continuous():
                    s = (
                        torch.ones(
                            size=(B,),
                            dtype=torch.float32,
                            device=low_res_x_0.device,
                        )
                        * context["augmentation_level"]
                    )
                else:
                    s = (
                        torch.ones(
                            size=(B,),
                            dtype=torch.long,
                            device=low_res_x_0.device,
                        )
                        * noise_scheduler.steps()
                        * context["augmentation_level"]
                    ).to(torch.long)
            else:
                if "augmentation_timestep" in context:
                    s = context["augmentation_timestep"]
                else:
                    s, _ = noise_scheduler.sample_random_times(
                        batch_size=B, device=low_res_x_0.device
                    )

            # Now noise the low res images to s
            low_res_x_0 = noise_scheduler.q_sample(low_res_x_0, s)

            # Add the augmentation timesteps to the conditioning
            context["augmentation_timestep"] = s

        # Concatenate the low resolution imagery to the input signal
        return torch.cat([x, low_res_x_0], dim=1)


class GaussianConditioningAugmentationToTimestep(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        time_embedding_mult: int,
        **kwargs,
    ):
        super().__init__()

        self._embedding_projection = TimestepEmbeddingProjection(
            num_features, time_embedding_mult
        )

    def forward(self, context: Dict, **kwargs):
        assert "timestep_embedding" in context
        assert "augmentation_timestep" in context

        projection = self._embedding_projection(context["augmentation_timestep"])
        if torch.isnan(projection).any():
            print(context["augmentation_timestep"])
            assert False

        timestep_embedding = context["timestep_embedding"]
        timestep_embedding = timestep_embedding + projection.to(timestep_embedding)
        context["timestep_embedding"] = timestep_embedding
        return context
