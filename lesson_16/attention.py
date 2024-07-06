"""Attention layers to use with DDPM.

This package implements multi-head cross attention from "Attention Is All You Need".
"""

from einops import rearrange
import math
import torch

from layers import conv_nd, zero_module, ConditioningBlock


class CrossAttention(ConditioningBlock):
    """Multi-Head Cross-Attention.

    When the context is None, this is equivalent to Multi-Head Self Attention.
    And when heads=1 and dim_head=query_dim, this is equivalent to the Self Attention
    class above.
    """

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


class SelfAttention(torch.nn.Module):
    """One head of self-attention"""

    def __init__(self, input_channels):
        super().__init__()
        self.key = torch.nn.Linear(input_channels, input_channels, bias=False)
        self.query = torch.nn.Linear(input_channels, input_channels, bias=False)
        self.value = torch.nn.Linear(input_channels, input_channels, bias=False)
        self.proj = torch.nn.Linear(input_channels, input_channels, bias=False)
        self.normalize = torch.nn.GroupNorm(num_groups=32, num_channels=input_channels)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.normalize(x)

        # Move channels to the end
        h = torch.permute(h, (0, 2, 3, 1))
        k = self.key(h)  # (B,H,W,C)
        q = self.query(h)  # (B,H,W,C)
        v = self.value(h)  # (B,H,W,C)

        # compute attention scores ("affinities")
        w = torch.einsum("bhwc,bHWc->bhwHW", q, k) * (int(C) ** (-0.5))  # (B,H,W,H,W)
        w = torch.reshape(w, [B, H, W, H * W])  # (B, H, W, H*W)
        w = torch.nn.functional.softmax(w, dim=-1)
        w = torch.reshape(w, [B, H, W, H, W])

        h = torch.einsum("bhwHW,bHWc->bhwc", w, v)
        h = self.proj(h)
        h = torch.permute(h, (0, 3, 1, 2))
        return x + h


class MultiHeadSelfAttention(torch.nn.Module):
    """Multihead Self Attention

        An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(num_heads=num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class MultiHeadCrossAttention(ConditioningBlock):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        encoder_channels=None,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)

        if encoder_channels is not None:
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, context=None):
        b, c, *spatial = x.shape
        qkv = self.qkv(self.norm(x).view(b, c, -1))
        if context is not None:
            encoder_out = self.encoder_kv(context)
            h = self.attention(qkv, encoder_out)
        else:
            h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h.reshape(b, c, *spatial)


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
