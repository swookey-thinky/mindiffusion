"""
Transformer implementation adapted from CLIP ViT:
https://github.com/openai/CLIP/blob/4c0275784d6d9da97ca1f47eaaee31de1867da91/clip/model.py
"""

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import torch


class LayerNorm(torch.nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: torch.Tensor):
        return super().forward(x.float()).to(x.dtype)


class MultiheadAttention(torch.nn.Module):
    def __init__(self, n_ctx, width, heads):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = torch.nn.Linear(width, width * 3)
        self.c_proj = torch.nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads, n_ctx)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


class MLP(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = torch.nn.Linear(width, width * 4)
        self.c_proj = torch.nn.Linear(width * 4, width)
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(torch.nn.Module):
    def __init__(self, n_heads: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class CausalMultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        dropout=0.0,
        causal=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5

        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = torch.nn.Dropout(dropout)

        self.to_q = torch.nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = torch.nn.Linear(dim, dim_head * 2, bias=False)

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, mask=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        q = q * self.scale

        q, k = map(lambda t: t * math.sqrt(self.scale), (q, k))

        # calculate query / key similarities
        sim = torch.einsum("b h i d, b j d -> b h i j", q, k)

        # masking
        max_neg_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            mask = torch.nn.functional.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype=torch.bool, device=device).triu(
                j - i + 1
            )
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention
        attn = sim.softmax(dim=-1, dtype=torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        # aggregate values
        out = torch.einsum("b h i j, b j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class ResidualAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        width: int,
        attention_channels: int,
        attention_heads: int,
        is_causal: bool = False,
    ):
        super().__init__()

        self.attn = CausalMultiHeadAttention(
            dim=width,
            dim_head=attention_channels,
            heads=attention_heads,
            causal=is_causal,
        )
        self.ln_1 = LayerNorm(width)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        context_size: int,
        layers: int,
        attention_channels: int,
        attention_heads: int,
        is_causal: bool = False,
    ):
        super().__init__()

        # The width of the transformer is attention_channels * attention_heads
        width = attention_channels * attention_heads
        self._width = width
        self._context_size = context_size

        self._projection = (
            torch.nn.Linear(in_features=context_size, out_features=width)
            if width != context_size
            else torch.nn.Identity()
        )

        self.resblocks = torch.nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    attention_channels,
                    attention_heads,
                    is_causal=is_causal,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        # Project into the inner dimension if we have it.
        x = self._projection(x)
        for block in self.resblocks:
            x = block(x)
        return x
