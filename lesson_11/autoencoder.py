"""Simple variational autoencoder for MNIST images.

This autoencoder takes an input representation at dimensions 1x32x32
and converts it into a latent space of 1x8x8.
"""

from einops import rearrange
import torch


class MNISTAutoEncoderKL(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # MNIST data comes in at (1,32,32), which is a
        # a (1, 1024) vector (which is already pretty small).
        # Let's take this down to a (1,8, 8) vector, and then back up,
        # to show a simple VAE and how to train it.
        input_channels = 32 * 32
        self.model_channels = 32 * 32
        self.encoders = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    in_features=input_channels, out_features=self.model_channels
                ),  # (1, 1024) -> (1, 1024)
                torch.nn.Linear(
                    in_features=self.model_channels,
                    out_features=self.model_channels // 4,
                ),  # (1, 1024) - > (1, 256)
                torch.nn.Linear(
                    in_features=self.model_channels // 4,
                    out_features=self.model_channels // 16,
                ),  # (1, 256) -> (1, 64)
            ]
        )
        self.mean = torch.nn.Linear(
            in_features=self.model_channels // 16,
            out_features=self.model_channels // 16,
        )
        self.log_variance = torch.nn.Linear(
            in_features=self.model_channels // 16,
            out_features=self.model_channels // 16,
        )

        self.decoders = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    in_features=self.model_channels // 16,
                    out_features=self.model_channels // 4,
                ),  # (1, 64) -> (1, 256)
                torch.nn.Linear(
                    in_features=self.model_channels // 4,
                    out_features=self.model_channels,
                ),  # (1, 256) -> (1, 1024)
                torch.nn.Linear(
                    in_features=self.model_channels, out_features=input_channels
                ),  # (1, 1024) -> (1, 1024)
            ]
        )

    def encoder(self, x):
        # Re-arrange the input into a single channel
        B, C, H, W = x.shape
        h = rearrange(x, "b c h w -> b c (h w)")

        # Run through the encoder modules
        for layer in self.encoders:
            h = torch.tanh(layer(h))
        return self.mean(h), self.log_variance(h)

    def decoder(self, z):
        # Run through the encoder modules
        h = z
        for layer in self.decoders:
            h = torch.tanh(layer(h))
        return h

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
        return rearrange(z, "b c (h w) -> b c h w", h=8, w=8)

    def decode(self, z):
        """Decodes latents of shape (B, 1, 8, 8)"""
        z = rearrange(z, "b c h w -> b c (h w)")
        x_hat = self.decoder(z)
        x_hat = rearrange(x_hat, "b c (h w) -> b c h w", h=32, w=32)
        return x_hat

    def forward(self, x):
        mean, log_var = self.encoder(x)

        # takes exponential function (log var -> var)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))

        x_hat = self.decoder(z)
        x_hat = rearrange(x_hat, "b c (h w) -> b c h w", h=32, w=32)

        return x_hat, mean, log_var
