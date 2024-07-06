"""Simple variational autoencoder for MNIST images.

This autoencoder takes an input representation at dimensions 1x32x32
and converts it into a latent space of 1x8x8.
"""

from einops import rearrange
import torch

from autoencoders.base import VariationalAutoEncoder
from utils import DotConfig


class MNISTAutoEncoderKL(torch.nn.Module, VariationalAutoEncoder):
    def __init__(self, config: DotConfig):
        super().__init__()

        self._config = config

        # MNIST data comes in at (1,32,32), which is a
        # a (1, 1024) vector (which is already pretty small).
        # Let's take this down to a (1,8,8) vector, and then back up,
        # to show a simple VAE and how to train it.
        num_features = config.num_features
        input_features = config.input_channels * config.input_resolution**2
        latent_size = config.latent_size
        self._latent_size = latent_size

        layers = []
        channels = list(
            map(lambda x: int(num_features * x), config.channel_multipliers)
        )
        num_layers = len(channels)

        in_ch = input_features
        for layer_idx, ch in enumerate(channels):
            layers.append(torch.nn.Linear(in_features=in_ch, out_features=ch))
            layers.append(torch.nn.Tanh())

            if layer_idx != (num_layers - 1):
                layers.append(torch.nn.Dropout(config.dropout))
            in_ch = ch
        self.encoders = torch.nn.Sequential(*layers)

        self.mean = torch.nn.Linear(
            in_features=ch,
            out_features=config.latent_channels * latent_size**2,
        )
        self.log_variance = torch.nn.Linear(
            in_features=ch,
            out_features=config.latent_channels * latent_size**2,
        )

        layers = []
        in_ch = config.latent_channels * latent_size**2
        for layer_idx, ch in enumerate(reversed(channels)):
            layers.append(torch.nn.Linear(in_features=in_ch, out_features=ch))
            layers.append(torch.nn.Tanh())

            if layer_idx != (num_layers - 1):
                layers.append(torch.nn.Dropout(config.dropout))
            in_ch = ch
        layers.append(torch.nn.Linear(in_features=ch, out_features=input_features))
        layers.append(torch.nn.Tanh())
        self.decoders = torch.nn.Sequential(*layers)

    def encoder(self, x):
        # Re-arrange the input into a single channel
        B, C, H, W = x.shape

        # Run through the encoder modules
        h = self.encoders(rearrange(x, "b c h w -> b (c h w)"))

        mean = self.mean(h)
        log_variance = self.log_variance(h)

        mean = rearrange(
            mean, "b (c h w) -> b c h w", h=self._latent_size, w=self._latent_size
        )
        log_variance = rearrange(
            log_variance,
            "b (c h w) -> b c h w",
            h=self._latent_size,
            w=self._latent_size,
        )

        return mean, log_variance

    def decoder(self, z):
        # Re-arrange the input into a single channel
        B, C, H, W = z.shape

        # Run through the decoder modules
        h = self.decoders(rearrange(z, "b c h w -> b (c h w)"))

        return rearrange(
            h,
            "b (c h w) -> b c h w",
            h=self._config.input_resolution,
            w=self._config.input_resolution,
        )

    def reparameterization(self, mean, var):
        # Sampling epsilon
        epsilon = torch.randn_like(var)
        # reparameterization trick
        z = mean + var * epsilon
        return z

    def encode(self, x):
        """Returns latents of shape (b, 1, 8, 8)"""
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        return z

    def decode(self, z):
        """Decodes latents of shape (B, 1, 8, 8)"""
        x_hat = self.decoder(z)
        return x_hat

    def encode_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into latents."""
        return self.encode(x)

    def decode_from_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latents into images."""
        return self.decode(z)

    def forward(self, x):
        mean, log_var = self.encoder(x)

        # takes exponential function (log var -> var)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        x_hat = self.decoder(z)
        return x_hat, mean, log_var
