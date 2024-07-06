import torch
from torchvision import transforms
from typing import Dict

from image_diffusion.utils import normalize_to_neg_one_to_one
from image_diffusion.scheduler import NoiseScheduler
from image_diffusion.layers.embedding import TimestepEmbeddingProjection


class InputPreprocessor(torch.nn.Module):
    """Prepares the low-resolution imagery for input concatenation."""

    def __init__(
        self,
        low_resolution_spatial_size: int,
        super_resolution_spatial_size: int,
        context_input_key: str,
        apply_gaussian_conditioning_augmentation: bool,
        **kwargs,
    ):
        super().__init__()
        self._super_resolution_spatial_size = super_resolution_spatial_size
        self._low_resolution_spatial_size = low_resolution_spatial_size
        self._context_input_key = context_input_key
        self._apply_gaussian_conditioning_augmentation = (
            apply_gaussian_conditioning_augmentation
        )

    def forward(
        self, x: torch.Tensor, context: Dict, noise_scheduler: NoiseScheduler, **kwargs
    ):
        low_resolution_images = context[self._context_input_key]
        B, _, H, W = low_resolution_images.shape

        assert (
            B == x.shape[0]
            and H == self._low_resolution_spatial_size
            and W == self._low_resolution_spatial_size
        )

        # Upsample the low resolution imagery to the model output size.
        low_res_x_0 = normalize_to_neg_one_to_one(
            transforms.functional.resize(
                low_resolution_images,
                size=(
                    self._super_resolution_spatial_size,
                    self._super_resolution_spatial_size,
                ),
                antialias=True,
            )
        )

        # Apply gaussian conditioning augmentation if configured
        if self._apply_gaussian_conditioning_augmentation:
            # Use non-truncating GCA. First sample s.
            if "augmentation_level" in context:
                s = (
                    torch.ones(
                        size=(B,), dtype=torch.long, device=noise_scheduler.betas.device
                    )
                    * noise_scheduler.num_timesteps
                    * context["augmentation_level"]
                )
            else:
                if "augmentation_timestep" in context:
                    s = context["augmentation_timestep"]
                else:
                    s = noise_scheduler.sample_random_times(batch_size=B)

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
