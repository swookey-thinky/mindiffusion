"""Utility layers used in defining a Denoising Diffusion Probabilistic Model."""

import math
import torch


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


class Block(torch.nn.Module):
    """
    A convolutional block which makes up the two convolutional
    layers in the ResnetBlock.
    """

    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = torch.nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=dim)
        self.act = torch.nn.SiLU()

    def forward(self, x):
        # The original paper implementation uses norm->swish->projection
        x = self.norm(x)
        x = self.act(x)
        x = self.proj(x)
        return x


class ResnetBlock(torch.nn.Module):
    """ResNet block based on WideResNet architecture.

    From DDPM, uses GroupNorm instead of weight normalization and Swish activation.
    """

    def __init__(self, dim, dim_out, time_emb_dim=None):
        super().__init__()
        self.timestep_proj = (
            torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.residual_proj = torch.nn.Linear(dim, dim_out)
        self.dim_out = dim_out

    def forward(self, x, time_emb=None):
        B, C, H, W = x.shape

        h = self.block1(x)

        # Add in the timstep embedding between blocks 1 and 2
        if time_emb is not None and self.timestep_proj is not None:
            h += self.timestep_proj(time_emb)[:, :, None, None]

        h = self.block2(h)

        # Project the residual channel to the output dimensions
        if C != self.dim_out:
            x = self.residual_proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return h + x


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
