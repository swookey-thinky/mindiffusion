from abc import ABC, abstractmethod
import torch


class VariationalAutoEncoder(ABC):
    @abstractmethod
    def encode_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into latents."""

    @abstractmethod
    def decode_from_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latents into images."""

    @abstractmethod
    def eval(self):
        """Moves the VAE to evaluation mode."""

    @abstractmethod
    def train(self):
        """Moves the VAE to train mode."""
