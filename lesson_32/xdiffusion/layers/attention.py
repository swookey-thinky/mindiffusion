"""Attention layers to use with DDPM.

This package implements multi-head self/cross attention from "Attention Is All You Need".
"""

from einops import rearrange
from functools import partial
import math
import torch
from torch.jit import Final
from typing import Optional, Dict

from xdiffusion.context import (
    NullContextAdapter,
)
from xdiffusion.layers.utils import conv_nd, normalization, zero_module, ContextBlock
from xdiffusion.utils import instantiate_from_config


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
        context_projection_input_dim=None,
        context_projection_output_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        is_video: bool = False,
        pre_layer_norm: bool = False,
        post_layer_norm: bool = False,
        context_layer_norm: bool = False,
        disable_self_attention: bool = False,
        context_adapter: Dict = {},
    ):
        """Initialize a new instance of SpatialCrossAttention."""
        super().__init__()
        self._is_video = is_video
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
        self._attention = QKVAttention(
            self._num_heads, disable_self_attention=disable_self_attention
        )

        if "target" in context_adapter:
            self._context_adapter = instantiate_from_config(context_adapter)
        else:
            self._context_adapter = NullContextAdapter()

        if context_dim is not None:
            self._encoder_kv = conv_nd(1, context_dim, in_channels * 2, 1)

        if (
            context_projection_input_dim is not None
            and context_projection_output_dim is not None
        ):
            self._context_proj = torch.nn.Linear(
                context_projection_input_dim, context_projection_output_dim
            )
        else:
            self._context_proj = torch.nn.Identity()
        self._proj_out = zero_module(conv_nd(1, in_channels, in_channels, 1))
        self._dropout = torch.nn.Dropout(dropout)

    def forward(self, x, context: Optional[Dict] = None):
        if self._is_video:
            B, C, F, H, W = x.shape
            x = rearrange(x, "b c f h w -> (b f) c h w")

        b, c, *spatial = x.shape
        qkv = self._qkv(self._norm(x).view(b, c, -1))
        if context is not None:
            context = self._context_layer_norm(self._context_adapter(context))
            if context is not None:
                encoder_out = self._encoder_kv(context)

                # encoder_out is shape (b, c, d), but need
                # to tile the batch dimension to all of the entries
                if self._is_video:
                    # Encoder out (B, C, L) -> (B, C, F, L)
                    encoder_out = encoder_out[:, :, None, :]
                    encoder_out = torch.tile(encoder_out, (1, 1, F, 1))
                    encoder_out = rearrange(encoder_out, "b c f l -> (b f) c l")
                else:
                    # Is it factorized?
                    if b != encoder_out.shape[0]:
                        # The batch sizes are not equal
                        F = b // encoder_out.shape[0]
                        encoder_out = encoder_out[:, :, None, :]
                        encoder_out = torch.tile(encoder_out, (1, 1, F, 1))
                        encoder_out = rearrange(encoder_out, "b c f l -> (b f) c l")
                encoder_out = self._context_proj(encoder_out)
            else:
                encoder_out = None
            h = self._attention(qkv, encoder_out)
        else:
            h = self._attention(qkv)
        h = self._proj_out(h)
        h = h.reshape(b, c, *spatial)
        h = self._final_norm(h)
        ret = x + self._dropout(h)

        if self._is_video:
            ret = rearrange(ret, "(b f) c h w -> b c f h w", b=B, c=C, f=F, h=H, w=W)

        return ret


class QKVAttention(torch.nn.Module):
    """A module which performs QKV attention."""

    def __init__(self, num_heads, disable_self_attention: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self._disable_self_attention = disable_self_attention

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

            # Concatting here adds both self attention and cross attention.
            # Without the cat, it is straight cross attention.
            ek, ev = encoder_kv.reshape(bs * self.num_heads, ch * 2, -1).split(
                ch, dim=1
            )
            if self._disable_self_attention:
                k = ek
                v = ev
            else:
                # Concatting here adds both self attention and cross attention.
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


class TemporalSelfAttention(ContextBlock):
    """An attention block that allows temporal positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.

    The relative position embeddings code was ported from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L1739

    The input to this block is of shape (B, C, *temporal).
    """

    def __init__(
        self,
        in_channels,
        temporal_sequence_length,
        max_relative_position,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        is_video: bool = False,
        pre_layer_norm: bool = False,
        post_layer_norm: bool = False,
        context_layer_norm: bool = False,
        context_adapter: Dict = {},
    ):
        """Initialize a new instance of TemporalSelfAttention."""
        super().__init__()
        self._is_video = is_video
        if context_dim == -1:
            context_dim = None
        self._channels = in_channels
        if dim_head == -1:
            self._num_heads = heads
            self._dim_head = in_channels // heads
        else:
            assert (
                in_channels % dim_head == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {dim_head}"
            self._num_heads = in_channels // dim_head
            self._dim_head = dim_head
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
        self._attention = QKVAttentionWithRelativePosition(
            self._num_heads,
            max_relative_position=max_relative_position,
            dim_head=self._dim_head,
            sequence_length=temporal_sequence_length,
        )

        if "target" in context_adapter:
            self._context_adapter = instantiate_from_config(context_adapter)
        else:
            self._context_adapter = NullContextAdapter()

        if context_dim is not None:
            self._encoder_kv = conv_nd(1, context_dim, in_channels * 2, 1)
        self._proj_out = zero_module(conv_nd(1, in_channels, in_channels, 1))
        self._dropout = torch.nn.Dropout(dropout)

    def forward(self, x, context: Optional[Dict] = None):
        if self._is_video:
            B, C, F, H, W = x.shape
            x = rearrange(x, "b c f h w -> (b h w) c f")
        b, c, *temporal = x.shape

        qkv = self._qkv(self._norm(x).view(b, c, -1))

        if context is not None:
            context = self._context_layer_norm(self._context_adapter(context))
            if context is not None:
                encoder_out = self._encoder_kv(context)

                # encoder_out is shape (b, c, d), but need
                # to tile the batch dimension to all of the entries
                # Encoder out (B, C, L) -> (B, C, F, L)
                encoder_out = torch.tile(encoder_out, (b // encoder_out.shape[0], 1, 1))
            else:
                encoder_out = None
            h = self._attention(qkv, encoder_out)
        else:
            h = self._attention(qkv)

        h = self._proj_out(h)
        h = h.reshape(b, c, *temporal)
        h = self._final_norm(h)
        ret = x + self._dropout(h)

        if self._is_video:
            ret = rearrange(ret, "(b h w) c f -> b c f h w", b=B, c=C, f=F, h=H, w=W)
        return ret


class QKVAttentionWithRelativePosition(torch.nn.Module):
    """A module which performs QKV attention using relative positions.

    Based on the TF implementation here:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L1739
    """

    def __init__(
        self,
        num_heads: int,
        max_relative_position: int,
        dim_head: int,
        sequence_length: int,
    ):
        super().__init__()
        # Depth is the dimension per head
        self.num_heads = num_heads

        # Generates embedding for each relative position of dimension depth.
        depth = dim_head
        self._depth = depth

        initializer_stddev = depth**-0.5
        max_relative_position_unmasked = 2 * max_relative_position - 1
        self._max_relative_position = max_relative_position

        self._k_embeddings_table = torch.nn.Parameter(
            torch.normal(
                mean=torch.zeros(
                    num_heads,
                    max_relative_position_unmasked,
                    depth,
                    dtype=torch.float32,
                ),
                std=torch.ones(
                    num_heads,
                    max_relative_position_unmasked,
                    depth,
                    dtype=torch.float32,
                )
                * initializer_stddev,
            )
        )
        self._v_embeddings_table = torch.nn.Parameter(
            torch.normal(
                mean=torch.zeros(
                    num_heads,
                    max_relative_position_unmasked,
                    depth,
                    dtype=torch.float32,
                ),
                std=torch.ones(
                    num_heads,
                    max_relative_position_unmasked,
                    depth,
                    dtype=torch.float32,
                )
                * initializer_stddev,
            )
        )

    def forward(self, qkv, encoder_kv=None):
        """Apply QKV attention.

        Args:
            qkv: an [B x (H * C * 3) x T] tensor of Qs, Ks, and Vs.

        Returns:
            A [B x (H * C) x T] tensor after attention.
        """
        B, C3, L = qkv.shape

        # num_heads = C // dim_head
        # Want to convert Q,K,V into tensors of shape
        # B, H, L, D, where D = C // H
        # from their current shape of B, C*3, L
        assert C3 % (3 * self.num_heads) == 0
        D = C3 // (3 * self.num_heads)
        q, k, v = qkv.reshape(B, self.num_heads, D * 3, L).split(D, dim=2)

        if encoder_kv is not None:
            ch = D

            # TODO: How do we combine self and cross attention here?
            #       For the self attention dot product below, concatting
            #       the original k,v with the encoder_kv yields the wrong shapes.
            assert encoder_kv.shape[1] == self.num_heads * ch * 2
            k, v = encoder_kv.reshape(B, self.num_heads, ch * 2, -1).split(ch, dim=2)

        # q, k, v are now tensors of shape [B, H, D, L],
        # but we want them to be [B, H, L, D]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        a = dot_product_unmasked_self_attention_relative_v2(
            q,
            k,
            v,
            bias=None,
            key_relative_embeddings=self._k_embeddings_table,
            value_relative_embeddings=self._v_embeddings_table,
            max_relative_position=self._max_relative_position,
        )
        return a.reshape(B, -1, L)


def dot_product_unmasked_self_attention_relative_v2(
    q,
    k,
    v,
    bias,
    key_relative_embeddings,
    value_relative_embeddings,
    max_relative_position=None,
    heads_share_relative_embedding=False,
    add_relative_to_values=False,
):
    """Calculate relative position-aware dot-product self-attention.

    The attention calculation is augmented with learned representations for the
    relative position between each element in q and each element in k and v.

    Args:
      q: a Tensor with shape [batch, heads, length, depth].
      k: a Tensor with shape [batch, heads, length, depth].
      v: a Tensor with shape [batch, heads, length, depth].
      bias: bias Tensor.
      max_relative_position: an integer the max relative embedding considered.
        Changing this invalidates checkpoints.
      dropout_rate: a floating point number.
      dropout_broadcast_dims:  an optional list of integers less than 4
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.
      heads_share_relative_embedding: a boolean indicating wheather to share
        relative embeddings between attention heads.
      add_relative_to_values: a boolean for whether to add relative component to
        values.

    Returns:
      A Tensor.

    Raises:
      ValueError: if max_relative_position is not > 0.
    """
    if not max_relative_position:
        raise ValueError(
            "Max relative position (%s) should be > 0 when using "
            "relative self attention." % (max_relative_position)
        )

    # This calculation only works for self attention.
    # q, k and v must therefore have the same shape.
    assert q.shape == k.shape
    assert q.shape == v.shape

    # [batch, num_heads, query_length, memory_length]
    logits = torch.matmul(q, k.transpose(-2, -1))

    length = q.shape[2]
    k_shape = k.shape
    num_heads = k_shape[1]
    depth_k = k_shape[-1]

    unmasked_rel_logits = matmul_with_relative_keys(
        q, key_relative_embeddings, heads_share_relative_embedding
    )
    unmasked_rel_logits = _relative_position_to_absolute_position_unmasked(
        unmasked_rel_logits
    )
    logits += unmasked_rel_logits

    if bias is not None:
        logits += bias
    weights = torch.nn.functional.softmax(logits, dim=-1)

    ret = torch.matmul(weights, v)
    if add_relative_to_values:
        # Adds the contribution of the weighted relative embeddings to the values.
        # [batch, num_heads, query_length, 2*memory_length-1]
        relative_weights = _absolute_position_to_relative_position_unmasked(weights)
        ret += matmul_with_relative_values(
            relative_weights,
            value_relative_embeddings,
            heads_share_relative_embedding,
        )
    return ret


def matmul_with_relative_keys(x, y, heads_share_relative_embedding):
    if heads_share_relative_embedding:
        ret = torch.einsum("bhld,md->bhlm", x, y)
    else:
        ret = torch.einsum("bhld,hmd->bhlm", x, y)
    return ret


def matmul_with_relative_values(x, y, heads_share_relative_embedding):
    if heads_share_relative_embedding:
        ret = torch.einsum("bhlm,md->bhld", x, y)
    else:
        ret = torch.einsum("bhlm,hmd->bhld", x, y)
    return ret


def _relative_position_to_absolute_position_unmasked(x):
    """Converts tensor from relative to aboslute indexing for local attention.

    Args:
      x: a Tensor of shape [batch (or batch*num_blocks), heads,
                            length, 2 * length - 1]

    Returns:
      A Tensor of shape [batch (or batch*num_blocks), heads, length, length]
    """
    x_shape = x.shape
    batch = x_shape[0]
    heads = x_shape[1]
    length = x_shape[2]

    # Concat columns of pad to shift from relative to absolute indexing.
    col_pad = torch.zeros((batch, heads, length, 1), device=x.device)
    x = torch.concat([x, col_pad], dim=3)

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    flat_x = torch.reshape(x, [batch, heads, length * 2 * length])
    flat_pad = torch.zeros((batch, heads, length - 1), device=x.device)
    flat_x_padded = torch.concat([flat_x, flat_pad], dim=2)

    # Reshape and slice out the padded elements.
    final_x = torch.reshape(flat_x_padded, [batch, heads, length + 1, 2 * length - 1])
    final_x = final_x[:, :, :, length - 1 :]
    final_x = final_x[:, :, :length, :]
    return final_x


def _absolute_position_to_relative_position_unmasked(x):
    """Helper function for dot_product_unmasked_self_attention_relative_v2.

    Rearrange an attention logits or weights Tensor.

    The dimensions of the input represent:
    [batch, heads, query_position, memory_position]

    The dimensions of the output represent:
    [batch, heads, query_position, memory_position - query_position + length - 1]

    Only works with unmasked_attention.

    Args:
      x: a Tensor with shape [batch, heads, length, length]

    Returns:
      a Tensor with shape [batch, heads, length, 2*length-1]
    """
    batch, heads, length, _ = x.shape
    # pad along column
    x = torch.nn.functional.pad(x, (0, length - 1))
    x_flat = torch.reshape(x, [batch, heads, length**2 + length * (length - 1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = torch.nn.functional.pad(x_flat, (length, 0))
    x = torch.reshape(x_flat, [batch, heads, length, 2 * length])
    x = x[:, :, :, 1 : 2 * length - 1]
    return x


class SpatialAndTemporalCrossAttention(ContextBlock):
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
        temporal_sequence_length,
        max_relative_position,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        is_video: bool = False,
        pre_layer_norm: bool = False,
        post_layer_norm: bool = False,
        context_layer_norm: bool = False,
        context_adapter: Dict = {},
    ):
        """Initialize a new instance of SpatialCrossAttention."""
        super().__init__()
        self._is_video = is_video
        if context_dim == -1:
            context_dim = None
        self._channels = in_channels
        assert is_video, "Only supports video input"

        if dim_head == -1:
            self._num_heads = heads
            self._dim_head = in_channels // heads
        else:
            assert (
                in_channels % dim_head == 0
            ), f"q,k,v channels {in_channels} is not divisible by num_head_channels {dim_head}"
            self._num_heads = in_channels // dim_head
            self._dim_head = dim_head
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

        # Now all of the temporal attention layers
        if pre_layer_norm:
            self._norm_temporal = ChanLayerNorm(in_channels)
        else:
            self._norm_temporal = torch.nn.GroupNorm(
                num_groups=32, num_channels=in_channels
            )
        if post_layer_norm:
            self._final_norm_temporal = ChanLayerNorm(in_channels)
        else:
            self._final_norm_temporal = torch.nn.Identity()

        self._qkv_temporal = conv_nd(1, in_channels, in_channels * 3, 1)
        self._attention_temporal = QKVAttentionWithRelativePosition(
            self._num_heads,
            max_relative_position=max_relative_position,
            dim_head=self._dim_head,
            sequence_length=temporal_sequence_length,
        )

        self._proj_out_temporal = zero_module(conv_nd(1, in_channels, in_channels, 1))
        self._dropout_temporal = torch.nn.Dropout(dropout)

    def forward(self, x, context: Optional[Dict] = None):
        # First apply the spatial attention
        spatial_attn = self._spatial_attention(x, context)

        # Then apply the temporal attention
        attn = self._temporal_attention(spatial_attn, context)
        return attn

    def _spatial_attention(self, x, context: Optional[Dict]):
        if self._is_video:
            B, C, F, H, W = x.shape
            x = rearrange(x, "b c f h w -> (b f) c h w")

        b, c, *spatial = x.shape
        qkv = self._qkv(self._norm(x).view(b, c, -1))
        if context is not None:
            context = self._context_layer_norm(self._context_adapter(context))
            if context is not None:
                encoder_out = self._encoder_kv(context)

                # encoder_out is shape (b, c, d), but need
                # to tile the batch dimension to all of the entries
                if self._is_video:
                    # Encoder out (B, C, L) -> (B, C, F, L)
                    encoder_out = encoder_out[:, :, None, :]
                    encoder_out = torch.tile(encoder_out, (1, 1, F, 1))
                    encoder_out = rearrange(encoder_out, "b c f l -> (b f) c l")
            else:
                encoder_out = None
            h = self._attention(qkv, encoder_out)
        else:
            h = self._attention(qkv)
        h = self._proj_out(h)
        h = h.reshape(b, c, *spatial)
        h = self._final_norm(h)
        ret = x + self._dropout(h)

        if self._is_video:
            ret = rearrange(ret, "(b f) c h w -> b c f h w", b=B, c=C, f=F, h=H, w=W)

        return ret

    def _temporal_attention(self, x, context: Optional[Dict]):
        if self._is_video:
            B, C, F, H, W = x.shape
            x = rearrange(x, "b c f h w -> (b h w) c f")
        b, c, *temporal = x.shape

        qkv = self._qkv_temporal(self._norm_temporal(x).view(b, c, -1))
        h = self._attention_temporal(qkv)
        h = self._proj_out_temporal(h)
        h = h.reshape(b, c, *temporal)
        h = self._final_norm_temporal(h)
        ret = x + self._dropout_temporal(h)

        if self._is_video:
            ret = rearrange(ret, "(b h w) c f -> b c f h w", b=B, c=C, f=F, h=H, w=W)
        return ret


class RPENet(torch.nn.Module):
    def __init__(self, channels, num_heads, time_embed_dim):
        super().__init__()
        self.embed_distances = torch.nn.Linear(3, channels)
        self.embed_diffusion_time = torch.nn.Linear(time_embed_dim, channels)
        self.silu = torch.nn.SiLU()
        self.out = torch.nn.Linear(channels, channels)
        self.out.weight.data *= 0.0
        self.out.bias.data *= 0.0
        self.channels = channels
        self.num_heads = num_heads

    def forward(self, temb, relative_distances):
        distance_embs = torch.stack(
            [
                torch.log(1 + (relative_distances).clamp(min=0)),
                torch.log(1 + (-relative_distances).clamp(min=0)),
                (relative_distances == 0).float(),
            ],
            dim=-1,
        )  # BxTxTx3
        B, T, _ = relative_distances.shape
        C = self.channels
        emb = self.embed_diffusion_time(temb).view(B, T, 1, C) + self.embed_distances(
            distance_embs
        )  # B x T x T x C
        return self.out(self.silu(emb)).view(
            *relative_distances.shape, self.num_heads, self.channels // self.num_heads
        )


class RPE(torch.nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py
    def __init__(self, channels, num_heads, time_embed_dim, use_rpe_net=False):
        """This module handles the relative positional encoding.
        Args:
            channels (int): Number of input channels.
            num_heads (int): Number of attention heads.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
        self.use_rpe_net = use_rpe_net
        if use_rpe_net:
            self.rpe_net = RPENet(channels, num_heads, time_embed_dim)
        else:
            self.lookup_table_weight = torch.nn.Parameter(
                torch.zeros(2 * self.beta + 1, self.num_heads, self.head_dim)
            )

    def get_R(self, pairwise_distances, temb):
        if self.use_rpe_net:
            return self.rpe_net(temb, pairwise_distances)
        else:
            return self.lookup_table_weight[pairwise_distances]  # BxTxTxHx(C/H)

    def forward(self, x, pairwise_distances, temb, mode):
        if mode == "qk":
            return self.forward_qk(x, pairwise_distances, temb)
        elif mode == "v":
            return self.forward_v(x, pairwise_distances, temb)
        else:
            raise ValueError(f"Unexpected RPE attention mode: {mode}")

    def forward_qk(self, qk, pairwise_distances, temb):
        # qv is either of q or k and has shape BxDxHxTx(C/H)
        # Output shape should be # BxDxHxTxT
        R = self.get_R(pairwise_distances, temb)
        return torch.einsum(  # See Eq. 16 in https://arxiv.org/pdf/2107.14222.pdf
            "bdhtf,btshf->bdhts", qk, R  # BxDxHxTxT
        )

    def forward_v(self, attn, pairwise_distances, temb):
        # attn has shape BxDxHxTxT
        # Output shape should be # BxDxHxYx(C/H)
        R = self.get_R(pairwise_distances, temb)
        torch.einsum("bdhts,btshf->bdhtf", attn, R)
        return torch.einsum(  # See Eq. 16ish in https://arxiv.org/pdf/2107.14222.pdf
            "bdhts,btshf->bdhtf", attn, R  # BxDxHxTxT
        )


class RPEAttention(torch.nn.Module):
    # Based on https://github.com/microsoft/Cream/blob/6fb89a2f93d6d97d2c7df51d600fe8be37ff0db4/iRPE/DeiT-with-iRPE/rpe_vision_transformer.py#L42
    def __init__(
        self,
        channels,
        num_heads,
        use_checkpoint=False,
        time_embed_dim=None,
        use_rpe_net=None,
        use_rpe_q=True,
        use_rpe_k=True,
        use_rpe_v=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim**-0.5
        self.use_checkpoint = use_checkpoint

        self.qkv = torch.nn.Linear(channels, channels * 3)
        self.proj_out = zero_module(torch.nn.Linear(channels, channels))
        self.norm = normalization(channels)

        if use_rpe_q or use_rpe_k or use_rpe_v:
            assert use_rpe_net is not None

        def make_rpe_func():
            return RPE(
                channels=channels,
                num_heads=num_heads,
                time_embed_dim=time_embed_dim,
                use_rpe_net=use_rpe_net,
            )

        self.rpe_q = make_rpe_func() if use_rpe_q else None
        self.rpe_k = make_rpe_func() if use_rpe_k else None
        self.rpe_v = make_rpe_func() if use_rpe_v else None

    def forward(self, x, temb, frame_indices, attn_mask=None):
        B, D, C, T = x.shape
        x = x.reshape(B * D, C, T)
        x = self.norm(x)
        x = x.view(B, D, C, T)
        x = torch.einsum("BDCT -> BDTC", x)  # just a permutation
        qkv = self.qkv(x).reshape(B, D, T, 3, self.num_heads, C // self.num_heads)
        qkv = torch.einsum("BDTtHF -> tBDHTF", qkv)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        # q, k, v shapes: BxDxHxTx(C/H)
        q *= self.scale
        attn = q @ k.transpose(-2, -1)  # BxDxHxTxT
        if self.rpe_q is not None or self.rpe_k is not None or self.rpe_v is not None:
            pairwise_distances = frame_indices.unsqueeze(-1) - frame_indices.unsqueeze(
                -2
            )  # BxTxT
        # relative position on keys
        if self.rpe_k is not None:
            attn += self.rpe_k(q, pairwise_distances, temb=temb, mode="qk")
        # relative position on queries
        if self.rpe_q is not None:
            attn += self.rpe_q(
                k * self.scale, pairwise_distances, temb=temb, mode="qk"
            ).transpose(-1, -2)

        # softmax where all elements with mask==0 can attend to eachother and all with mask==1
        # can attend to eachother (but elements with mask==0 can't attend to elements with mask==1)
        def softmax(w, attn_mask):
            if attn_mask is not None:
                # Elements with mask==1 can attent to each other
                allowed_interactions = attn_mask.view(B, 1, T) * attn_mask.view(B, T, 1)
                # Elements with mask==0 can attend to each other
                allowed_interactions += (1 - attn_mask.view(B, 1, T)) * (
                    1 - attn_mask.view(B, T, 1)
                )
                # Select all of the 0 entries for the infinity mask.
                # This will send the softmax to zero.
                inf_mask = 1 - allowed_interactions
                inf_mask[inf_mask == 1] = torch.inf

                # Set the weights entries to -infinity where the masks were zero.
                # Those elements will not attend to each other.
                w = w - inf_mask.view(B, 1, 1, T, T)  # BxDxHxTxT
            return torch.softmax(w.float(), dim=-1).type(w.dtype)

        attn = softmax(attn, attn_mask)
        out = attn @ v
        # relative position on values
        if self.rpe_v is not None:
            out += self.rpe_v(attn, pairwise_distances, temb=temb, mode="v")
        out = torch.einsum("BDHTF -> BDTHF", out).reshape(B, D, T, C)
        out = self.proj_out(out)
        x = x + out
        x = torch.einsum("BDTC -> BDCT", x)
        return x


class FactorizedAttentionBlock(torch.nn.Module):
    """Space-time factorized attention, based on Flexible Diffusion Modeling.

    Implements spatial and temporal self/cross attention.
    """

    def __init__(
        self,
        channels,
        num_heads,
        use_rpe_net,
        time_embed_dim=None,
    ):
        super().__init__()
        self.spatial_attention = RPEAttention(
            channels=channels,
            num_heads=num_heads,
            use_rpe_q=False,
            use_rpe_k=False,
            use_rpe_v=False,
        )
        self.temporal_attention = RPEAttention(
            channels=channels,
            num_heads=num_heads,
            time_embed_dim=time_embed_dim,
            use_rpe_net=use_rpe_net,
        )

    def forward(self, x, context: Dict):
        B, C, T, H, W = x.shape

        frame_indices = context["frame_indices"]
        temb = context["timestep_embedding"]
        attn_mask = (context["observed_mask"] + context["latent_mask"]).clip(max=1)

        # First attend over the temporal dimension, so permute to (B, H, W, C, T)
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(B, H * W, C, T)

        x = self.temporal_attention(
            x,
            temb,
            frame_indices,
            attn_mask=attn_mask.flatten(start_dim=2).squeeze(dim=2),  # B x T
        )

        # Now we attend over the spatial dimensions by reshaping the input
        x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2)  # B, T, C, H, W
        x = x.reshape(B, T, C, H * W)
        x = self.spatial_attention(
            x,
            temb,
            frame_indices=None,
        )
        x = x.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)  # B, C, T, H, W
        return x


def attention_with_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positional_embedding: torch.Tensor,
) -> torch.Tensor:
    q, k = apply_rope(q, k, positional_embedding)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
