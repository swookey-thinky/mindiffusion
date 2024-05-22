"""Utility layers used in defining a DDPM with Dropout."""

from abc import abstractmethod
import math
import torch


def conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return torch.nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return torch.nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return torch.nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    if dims == 1:
        return torch.nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return torch.nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return torch.nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class TimestepBlock(torch.nn.Module):
    """Basic block which accepts a timestep embedding as the second argument."""

    @abstractmethod
    def forward(self, x, time_emb):
        """Apply the module to `x` given `time_emb` timestep embeddings."""


class TimestepEmbedSequential(torch.nn.Sequential, TimestepBlock):
    """Sequential module for timestep embeddings.

    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, time_emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        return x


class ResnetBlock(TimestepBlock):
    """ResNet block based on WideResNet architecture.

    From DDPM, uses GroupNorm instead of weight normalization and Swish activation.
    """

    def __init__(
        self,
        dim,
        dim_out,
        time_emb_dim=None,
        use_scale_shift_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        self._use_scale_shift_norm = use_scale_shift_norm
        self.timestep_proj = (
            torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(
                    time_emb_dim, 2 * dim_out if use_scale_shift_norm else dim_out
                ),
            )
            if time_emb_dim is not None
            else None
        )

        self.block1 = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=dim),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim, dim_out, 3, padding=1),
        )

        # In the DDPM implementation, dropout was added to the second
        # resnet layer in the block, in front of the final convolution.
        self.block2 = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=dim_out),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity(),
            torch.nn.Conv2d(dim_out, dim_out, 3, padding=1),
        )

        self.residual_proj = torch.nn.Linear(dim, dim_out)
        self.dim_out = dim_out

    def forward(self, x, time_emb=None):
        B, C, H, W = x.shape
        h = self.block1(x)
        # Add in the timstep embedding between blocks 1 and 2
        if time_emb is not None and self.timestep_proj is not None:
            emb_out = self.timestep_proj(time_emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]

            # Scale/Shift of the norm here is from the IDDPM paper,
            # as one of their improvements.
            if self._use_scale_shift_norm:
                out_norm, out_rest = self.block2[0], self.block2[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h += emb_out
                h = self.block2(h)

        # Project the residual channel to the output dimensions
        if C != self.dim_out:
            x = self.residual_proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return h + x


class SinusoidalPositionEmbedding(torch.nn.Module):
    """Implementation of Sinusoidal Position Embedding.

    Originally introduced in the paper "Attention Is All You Need",
    the original tensorflow implementation is here:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L408
    """

    def __init__(self, embedding_dim, theta=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.embedding_dim // 2
        embedding = math.log(self.theta) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=device) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding


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
    """Multiheaded Self Attention.

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
        self.attention = QKVAttention()
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


class QKVAttention(torch.nn.Module):
    """A module which performs QKV attention."""

    def forward(self, qkv):
        """Apply QKV attention.

        Args:
            qkv: an [B x (C * 3) x T] tensor of Qs, Ks, and Vs.

        Returns:
            A [B x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)


class Downsample(torch.nn.Module):
    """A downsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
            downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(torch.nn.Module):
    """An upsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
            upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = torch.nn.functional.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
