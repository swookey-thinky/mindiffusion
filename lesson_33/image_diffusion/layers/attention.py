"""Attention layers to use with DDPM.

This package implements multi-head self/cross attention from "Attention Is All You Need".
"""

from einops import rearrange
from functools import partial
import math
import torch
from torch.jit import Final
from typing import Optional, Dict

from image_diffusion.layers.utils import conv_nd, zero_module, ContextBlock
from image_diffusion.conditioning import (
    NullContextAdapter,
)
from image_diffusion.utils import instantiate_from_config


class SpatialCrossAttention(ContextBlock):
    """An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.

    When the context_dim is None or -1, this is equivalent to Multi-Head Self Attention.
    And when heads=1 and dim_head=in_channels, this is equivalent to the self attention.

    The input to this block is of shape (B, C, *spatial)
    """

    def __init__(
        self,
        in_channels,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        pre_layer_norm: bool = False,
        post_layer_norm: bool = False,
        context_layer_norm: bool = False,
        context_adapter: Dict = {},
    ):
        """Initialize a new instance of SpatialCrossAttention."""
        super().__init__()

        if context_dim == -1:
            context_dim = None
        self._channels = in_channels
        if dim_head == -1:
            self._num_heads = heads
        else:
            assert (
                in_channels % dim_head == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {dim_head}"
            self._num_heads = in_channels // dim_head
        if pre_layer_norm:
            self._norm = ChanLayerNorm(in_channels)
        else:
            self._norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels)
        if post_layer_norm:
            self._final_norm = ChanLayerNorm(in_channels)
        else:
            self._final_norm = torch.nn.Identity()

        if context_layer_norm:
            self._context_layer_norm = ChanLayerNorm(context_dim, dim=-2)
        else:
            self._context_layer_norm = torch.nn.Identity()

        self._qkv = conv_nd(1, in_channels, in_channels * 3, 1)
        self._attention = QKVAttention(self._num_heads)

        if "target" in context_adapter:
            self._context_adapter = instantiate_from_config(context_adapter)
        else:
            self._context_adapter = NullContextAdapter()

        if context_dim is not None:
            self._encoder_kv = conv_nd(1, context_dim, in_channels * 2, 1)
        self._proj_out = zero_module(conv_nd(1, in_channels, in_channels, 1))
        self._dropout = torch.nn.Dropout(dropout)

    def forward(self, x, context: Optional[Dict] = None):
        b, c, *spatial = x.shape
        qkv = self._qkv(self._norm(x).view(b, c, -1))
        if context is not None:
            context = self._context_layer_norm(self._context_adapter(context))
            if context is not None:
                encoder_out = self._encoder_kv(context)
            else:
                encoder_out = None
            h = self._attention(qkv, encoder_out)
        else:
            h = self._attention(qkv)
        h = self._proj_out(h)
        h = h.reshape(b, c, *spatial)
        h = self._final_norm(h)
        return x + self._dropout(h)


class QKVAttention(torch.nn.Module):
    """A module which performs QKV attention."""

    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv, encoder_kv=None):
        """Apply QKV attention.

        Args:
            qkv: an [B x (H * C * 3) x T] tensor of Qs, Ks, and Vs.

        Returns:
            A [B x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        q, k, v = qkv.reshape(bs * self.num_heads, ch * 3, length).split(ch, dim=1)

        if encoder_kv is not None:
            assert encoder_kv.shape[1] == self.num_heads * ch * 2
            ek, ev = encoder_kv.reshape(bs * self.num_heads, ch * 2, -1).split(
                ch, dim=1
            )
            k = torch.cat([ek, k], dim=-1)
            v = torch.cat([ev, v], dim=-1)

        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class LastChannelCrossAttention(torch.nn.Module):
    """Same a SpatialCrossAttention but optimized for (B, *, C) input."""

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = torch.nn.Linear(inner_dim, query_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, context=None):
        h = self.heads
        q = self.to_q(x)

        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = self.dropout(out)
        return out


class AttentionPooling(torch.nn.Module):
    """Implements attention pooling from Imagen."""

    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.dim_per_head = embed_dim // self.num_heads

    def forward(self, x):
        bs, length, width = x.size()

        def shape(x):
            # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
            x = x.view(bs, -1, self.num_heads, self.dim_per_head)
            # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
            x = x.transpose(1, 2)
            # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
            x = x.reshape(bs * self.num_heads, -1, self.dim_per_head)
            # (bs*n_heads, length, dim_per_head) --> (bs*n_heads, dim_per_head, length)
            x = x.transpose(1, 2)
            return x

        class_token = x.mean(dim=1, keepdim=True) + self.positional_embedding.to(
            x.dtype
        )
        x = torch.cat([class_token, x], dim=1)  # (bs, length+1, width)

        # (bs*n_heads, class_token_length, dim_per_head)
        q = shape(self.q_proj(class_token))
        # (bs*n_heads, length+class_token_length, dim_per_head)
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # (bs*n_heads, class_token_length, length+class_token_length):
        scale = 1 / math.sqrt(math.sqrt(self.dim_per_head))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)

        # (bs*n_heads, dim_per_head, class_token_length)
        a = torch.einsum("bts,bcs->bct", weight, v)

        # (bs, length+1, width)
        a = a.reshape(bs, -1, 1).transpose(1, 2)

        return a[:, 0, :]  # cls_token


class LayerNorm(torch.nn.Module):
    """LayerNorm class which supports axis specification."""

    def __init__(self, feats, stable=False, dim=-1):
        super().__init__()
        self.stable = stable
        self.dim = dim

        self.g = torch.nn.Parameter(torch.ones(feats, *((1,) * (-dim - 1))))

    def forward(self, x):
        dtype, dim = x.dtype, self.dim
        if self.stable:
            x = x / x.amax(dim=dim, keepdim=True).detach()

        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=dim, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=dim, keepdim=True)

        return (x - mean) * (var + eps).rsqrt().type(dtype) * self.g.type(dtype)


# Helper to perform LayerNorm over the channel axis for spatial input
# (shape = [B, C, H, W]).
ChanLayerNorm = partial(LayerNorm, dim=-3)


class MultiHeadSelfAttention(torch.nn.Module):
    """Multi-Head Self Attention class used by DiT.

    Based on the transformer attention in vision transformers from here:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L58C1-L106C17

    Optimized for the case where input is of shape (B, L, C), and supports
    using built-in scaled dot product attention. Has no support for cross attention.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_fused_attn: bool = False,
        norm_layer: torch.nn.Module = torch.nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn

        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else torch.nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else torch.nn.Identity()
        self.attn_drop = torch.nn.Dropout(attn_drop)
        self.proj = torch.nn.Linear(dim, dim)
        self.proj_drop = torch.nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            assert hasattr(
                torch.nn.functional, "scaled_dot_product_attention"
            ), "Torch version does not have scaled_dot_product_attention. disable fused attention."
            x = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
