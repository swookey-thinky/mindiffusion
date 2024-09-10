"""Transformer implementations, including spatial transformers.

The basic transformer implementation was adapted from CLIP ViT:
https://github.com/openai/CLIP/blob/4c0275784d6d9da97ca1f47eaaee31de1867da91/clip/model.py

The spatial transformer implements basic transformer block functionality,
and is based on the original LDM implementation from
https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py#L218
"""

from einops import rearrange
import math
import torch
from typing import Dict

from xdiffusion.layers.attention import (
    SpatialCrossAttention,
    LastChannelCrossAttention,
)
from xdiffusion.context import (
    ContextAdapter,
    NullContextAdapter,
)
from xdiffusion.layers.utils import ContextBlock
from xdiffusion.utils import instantiate_from_config


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class LayerNorm(torch.nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).to(x.dtype)


class GEGLU(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = (
            torch.nn.Sequential(torch.nn.Linear(dim, inner_dim), torch.nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = torch.nn.Sequential(
            project_in, torch.nn.Dropout(dropout), torch.nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class ResidualAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        context_dim: int,
        width: int,
        heads: int,
    ):
        super().__init__()

        self.attn = SpatialCrossAttention(
            in_channels=context_dim, heads=heads, dim_head=width // heads
        )
        self.ln_1 = LayerNorm(width)
        self.mlp = FeedForward(dim=width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SpatialTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
    ):
        super().__init__()

        # Use the last channel cross attention here becuase
        # the input at each layer is (B, (H*W), C), and layernorm
        # needs to operate on C channels.
        self.attn1 = LastChannelCrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = LastChannelCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )  # is self-attn if context is none

        # Layer normalization happens in the last dimension,
        # which is the spaial dimension
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.norm3 = torch.nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class Transformer(torch.nn.Module):
    """Basic transformer adapted from ViT.

    This transformer accepts input data of shape [B, N, context_dim].
    """

    def __init__(
        self,
        context_dim: int,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        self._context_dim = context_dim
        self._width = width
        self._layers = layers
        self.resblocks = torch.nn.ModuleList(
            [
                ResidualAttentionBlock(
                    context_dim=context_dim,
                    width=width,
                    heads=heads,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x


class SpatialTransformer(ContextBlock):
    """Spatial transformer adapted from LDM.

    Transformer block for image-like data. First, project the input (aka embedding)
    and reshape to b, t, d. Then apply standard transformer action.
    Finally, reshape to image.
    """

    def __init__(
        self,
        in_channels,
        attention_heads,
        attention_channels,
        num_layers=1,
        dropout=0.0,
        context_dim=None,
        context_adapter: Dict = {},
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = attention_heads * attention_channels
        self.norm = Normalize(in_channels)

        self.proj_in = torch.nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                SpatialTransformerBlock(
                    inner_dim,
                    attention_heads,
                    attention_channels,
                    dropout=dropout,
                    context_dim=context_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = zero_module(
            torch.nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

        if "target" in context_adapter:
            self._context_adapter = instantiate_from_config(context_adapter)
        else:
            self._context_adapter = NullContextAdapter

    def forward(self, x, context: Dict):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        # If the shape of the context is (B, context_dim),
        # we need to extend it to (B, C, context_dim).
        context = self._context_adapter(context)
        if context is not None:
            if len(context.shape) == 2:
                context = context[:, None, :]

        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class GLIDETransformerWrapper(torch.nn.Module):
    """Wraps a Transformer with additional layers, per GLIDE.

    This is the shared transformer used across layers in GLIDE.
    """

    def __init__(
        self,
        context_dim: int,
        width: int,
        layers: int,
        heads: int,
        final_layer_norm: bool,
        output_projection_dimension: int,
        projections: torch.nn.ModuleDict,
        **kwargs,
    ):
        super().__init__()

        self._context_transformer = Transformer(
            context_dim=context_dim,
            layers=layers,
            width=width,
            heads=heads,
        )
        self._projections = projections

        if final_layer_norm:
            self._final_layer_norm = LayerNorm(normalized_shape=width)
        else:
            self._final_layer_norm = torch.nn.Identity()

        self._positional_embedding = torch.nn.Parameter(
            torch.empty(
                1,
                context_dim,
                dtype=torch.float32,
            )
        )
        self._transformer_projection = torch.nn.Linear(
            width, output_projection_dimension
        )

    def forward(self, context: Dict, **kwargs):
        # Check for text embeddings. If we have them, they supercede text tokens
        if "text_embedding" in context:
            # Use the text embedding directly
            xf_in = context["text_embedding"][:, None, :]
        elif "text_tokens" in context:
            xf_in = self._projections["text_tokens"](context["text_tokens"].long())
        else:
            # The GLIDE transformer wrapper needs either text embeddings or text tokens
            assert False, "GLIDE transformer needs text tokens or embeddings."
        xf_in = xf_in + self._positional_embedding[None]
        xf_out = self._context_transformer(
            xf_in.to(context["timestep_embedding"].dtype)
        )
        xf_out = self._final_layer_norm(xf_out)
        xf_proj = self._transformer_projection(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        # The projection is used for the timestep embedding
        timestep_embedding = context["timestep_embedding"]
        context["timestep_embedding"] = timestep_embedding + xf_proj.to(
            timestep_embedding
        )

        # The transformer output becomes the embedded context.
        context["context_embedding"] = xf_out
        return context
