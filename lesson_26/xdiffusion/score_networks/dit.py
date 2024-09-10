"""Transformer based score network from Scalable Diffusion Models with Transformers."""

import torch
from typing import Dict

from xdiffusion.layers.attention import MultiHeadSelfAttention as Attention
from xdiffusion.layers.embedding import PatchEmbed
from xdiffusion.layers.mlp import Mlp
from xdiffusion.layers.utils import get_2d_sincos_pos_embed
from xdiffusion.utils import (
    instantiate_from_config,
    instantiate_partial_from_config,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(torch.nn.Module):
    """DiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(torch.nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = torch.nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = torch.nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.adaLN_modulation = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(torch.nn.Module):
    """Diffusion score network with a Transformer backbone."""

    def __init__(
        self,
        config: Dict,
    ):
        super().__init__()

        input_spatial_size = config.input_spatial_size
        patch_size = config.patch_size
        input_channels = config.input_channels
        hidden_size = config.hidden_size
        depth = config.depth
        num_heads = config.num_heads
        mlp_ratio = config.mlp_ratio
        learn_sigma = config.is_learned_sigma

        self.learn_sigma = learn_sigma
        self.in_channels = input_channels
        self.out_channels = input_channels * 2 if learn_sigma else input_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        # Spatial embedding for the image.
        self.x_embedder = PatchEmbed(
            input_spatial_size, patch_size, input_channels, hidden_size, bias=True
        )

        # Instantiate all of the projections used in the model
        self._projections = torch.nn.ModuleDict()
        for projection_name in config.conditioning.signals:
            self._projections[projection_name] = instantiate_from_config(
                config.conditioning.projections[projection_name].to_dict()
            )

        # If we have a context transformer, let's create it here. This is
        # a GLIDE style transformer for embedding the text context
        # into both the timestep embedding and the attention layers,
        # and is applied at the top of the network once.
        if isinstance(config.conditioning.context_transformer_head, list):
            self._context_transformers = torch.nn.ModuleList([])
            for c in config.conditioning.context_transformer_head:
                self._context_transformers.append(
                    instantiate_partial_from_config(c)(projections=self._projections)
                )
        else:
            self._context_transformers = torch.nn.ModuleList(
                [
                    instantiate_partial_from_config(
                        config.conditioning.context_transformer_head.to_dict()
                    )(projections=self._projections)
                ]
            )

        # Will use fixed sin-cos position embeddings for the patches.
        num_patches = self.x_embedder.num_patches
        self.pos_embed = torch.nn.Parameter(
            torch.zeros(1, num_patches, hidden_size), requires_grad=False
        )

        # Transformer blocks
        self.blocks = torch.nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            torch.nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            torch.nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        torch.nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        torch.nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        torch.nn.init.constant_(self.final_layer.linear.weight, 0)
        torch.nn.init.constant_(self.final_layer.linear.bias, 0)

        # Run any special initializers
        def _custom_init(module):
            if hasattr(module, "custom_initializer"):
                module.custom_initializer()

        self.apply(_custom_init)

    def unpatchify(self, x):
        """Converts sequence of image patches into spatial image.

        Args:
            x: (N, T, patch_size**2 * C)

        Returns:
            imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, context: Dict):
        """Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # Don't change the original context
        context = context.copy()

        # Transform the context at the top if we have it. This will generate
        # an embedding to combine with the timestep projection, and the embedded
        # context.
        for context_transformer in self._context_transformers:
            context = context_transformer(context, device=x.device)

        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        c = context["timestep_embedding"]  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x
