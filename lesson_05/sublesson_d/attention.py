"""Attention layers to use with DDPM.

This package implements multi-head cross attention from "Attention Is All You Need".
"""

from einops import rearrange
import torch


class CrossAttention(torch.nn.Module):
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
