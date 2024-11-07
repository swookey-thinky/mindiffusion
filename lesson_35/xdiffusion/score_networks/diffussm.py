"""DiffuSSM score network from 'Diffusion Models Without Attention'

https://arxiv.org/abs/2311.18257
"""

from einops import rearrange
import torch
from typing import Dict

from xdiffusion.layers.flux import Modulation, MLPEmbedder
from xdiffusion.layers.utils import timestep_embedding
from xdiffusion.layers.sequence import SequenceModel
from xdiffusion.utils import instantiate_partial_from_config


class DiffusionSSMBlock(torch.nn.Module):
    def __init__(self, config: Dict):
        super().__init__()

        self._input_modulation = Modulation(dim=config.d_model, double=False)
        self._input_norm = torch.nn.LayerNorm(
            config.d_model, elementwise_affine=False, eps=1e-6
        )

        # Embedder for the timestep position embedding
        self._condition_embedder = MLPEmbedder(in_dim=256, hidden_dim=config.d_model)

        # Hourglass module before going into the SSM
        L = config.input_spatial_size**2
        J = L // config.M
        self._hourglass_ratio = config.M
        self._hourglass = torch.nn.Sequential(
            # Downscale
            torch.nn.Conv1d(in_channels=L, out_channels=J, kernel_size=1),
            # MLP
            MLPEmbedder(in_dim=config.d_model, hidden_dim=config.d_model),
            # Upscale
            torch.nn.Conv1d(in_channels=J, out_channels=L, kernel_size=1),
        )

        # Bidirectional SSM (S4D) block
        self._ssm = instantiate_partial_from_config(config.block_config.to_dict())(
            d_input=config.d_model
        )

        # Fuse the SSM output
        self._downscale_left = torch.nn.Conv1d(
            in_channels=L, out_channels=J, kernel_size=1
        )
        self._downscale_right = torch.nn.Conv1d(
            in_channels=L, out_channels=J, kernel_size=1
        )
        self._mlp_left = MLPEmbedder(in_dim=config.d_model, hidden_dim=config.d_model)
        self._mlp_right = MLPEmbedder(in_dim=config.d_model, hidden_dim=config.d_model)
        self._mlp_final = MLPEmbedder(in_dim=config.d_model, hidden_dim=config.d_model)
        self._upscale_final = torch.nn.Conv1d(
            in_channels=J, out_channels=L, kernel_size=1
        )

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor):
        y = self._condition_embedder(time_embed)
        modulation, _ = self._input_modulation(y)

        # Input norm
        h = self._input_norm(x)
        # Scale/shift
        h = (1 + modulation.scale) * h + modulation.shift

        # Hourglass before SSM
        h_ssm, _ = self._ssm(self._hourglass(h))

        # Fuse the output of the SSM and the scaled/shifted residual
        h_fused = self._mlp_left(self._downscale_left(h)) * self._mlp_right(
            self._downscale_right(h_ssm)
        )
        h_fused = self._upscale_final(self._mlp_final(h_fused))

        # Gate the output
        return h + modulation.gate * h_fused


class DiffusionSSM(torch.nn.Module):
    """Diffusion score network with a Transformer backbone."""

    def __init__(
        self,
        config: Dict,
    ):
        super().__init__()

        # Input projection to model dimensions
        self._input_proj = torch.nn.Linear(config.d_input, config.d_model)
        # Output projection to input dimensions
        self._output_proj = torch.nn.Linear(config.d_model, config.d_input)

        # Instantiate the individual layers
        self._layers = torch.nn.ModuleList(
            [DiffusionSSMBlock(config) for _ in range(config.n_layers)]
        )

    def forward(self, x, context: Dict):
        """Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # Grab the timestep and embed it
        timesteps = context["timestep"]
        time_embed = timestep_embedding(timesteps, dim=256)

        # Flatten input x from (B, C, H, W) to (B, H*W, C)
        B, C, H, W = x.shape

        x = rearrange(x, "b c h w -> b (h w) c")

        # Project x to the model dimensions
        h = self._input_proj(x)

        # Run through all of the SSM layers
        for layer in self._layers:
            h = layer(h, time_embed)

        # Project back to the input dimensions
        h = self._output_proj(h)

        # Reshape back to the input shape
        h = rearrange(h, "b (h w) c -> b c h w", h=H, w=W)
        return h
