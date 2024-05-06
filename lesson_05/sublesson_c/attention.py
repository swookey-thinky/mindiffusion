"""Attention layers to use with DDPM.

In addition to self attention, this package include multi-head cross
attention from "Attention Is All You Need".
"""

from einops import rearrange
import torch


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

        self.to_out = torch.nn.Sequential(
            torch.nn.Linear(inner_dim, query_dim), torch.nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        # x was shape (B, C, H, W) and was reshaped to (B, H*W, C).
        # context needs to be an equivalent shape.
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
        return self.to_out(out)


class AttentionLayer(torch.nn.Module):
    """A simple layer to add both self and cross attention."""

    def __init__(
        self,
        in_channels,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()

        inner_dim = heads * dim_head

        # Projection to convert the input dimensions to the
        # attention dimensions.
        self.input_proj = torch.nn.Conv2d(
            in_channels, inner_dim, kernel_size=1, stride=1, padding=0
        )

        self.self_attn = SelfAttention(input_channels=inner_dim)
        self.cross_attn = CrossAttention(
            query_dim=inner_dim,
            context_dim=context_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        self.output_proj = torch.nn.Conv2d(
            inner_dim, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.normalize = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels)

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x

        x = self.normalize(x)
        x = self.input_proj(x)

        # Self attend with residual
        x = self.self_attn(x) + x

        # Then cross attend with residual
        x = rearrange(x, "b c h w -> b (h w) c")

        # If the shape of the context is (B, context_dim),
        # we need to extend it to (B, C, context_dim).
        if context is not None:
            if len(context.shape) == 2:
                context = context[:, None, :]
        x = self.cross_attn(x, context=context) + x
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        x = self.output_proj(x)
        return x + x_in
