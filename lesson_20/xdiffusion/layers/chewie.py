import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from xdiffusion.layers.utils import RMSNorm
from xdiffusion.layers.flux import Modulation, SelfAttention, attention


class TripleStreamBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Spatial processing
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # Temporal processing
        self.temporal_mod = Modulation(hidden_size, double=True)
        self.temporal_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.temporal_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )
        self.temporal_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # Textual processing
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(
        self, img: Tensor, temporal: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of block.

        Args:
            img: Tensor batch of noised spatial image data (B, L, D)
            temporal: Tensor batch of noised temporal data (B, L', D)
            txt: Tensor batch of text embeddings (B, L, D)
            vec: Tensor batch of timestep (and other) embeddings (B, L, D)
            pe: Tensor batch of positional encodings (B, L, D)
        """
        # Separate moduation for text and image pathways
        img_mod1, img_mod2 = self.img_mod(vec)
        temporal_mod1, temporal_mod2 = self.temporal_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        # QK normalization, per https://arxiv.org/abs/2302.05442
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # Prepare spatial for attention
        temporal_modulated = self.temporal_norm1(temporal)
        temporal_modulated = (
            1 + temporal_mod1.scale
        ) * temporal_modulated + temporal_mod1.shift
        temporal_qkv = self.temporal_attn.qkv(temporal_modulated)
        temporal_q, temporal_k, temporal_v = rearrange(
            temporal_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        # QK normalization, per https://arxiv.org/abs/2302.05442
        temporal_q, temporal_k = self.temporal_attn.norm(
            temporal_q, temporal_k, temporal_v
        )

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        # QK normalization, per https://arxiv.org/abs/2302.05442
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q, temporal_q), dim=2)
        k = torch.cat((txt_k, img_k, temporal_k), dim=2)
        v = torch.cat((txt_v, img_v, temporal_q), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn, temporal_attn = (
            attn[:, : txt.shape[1]],
            attn[:, txt.shape[1] : txt.shape[1] + img.shape[1]],
            attn[:, txt.shape[1] + img.shape[1] :],
        )

        # calculate the img bloks. Note this is NOT using the parallel
        # attention layers from https://arxiv.org/abs/2302.05442, but rather
        # the standard MMDiT calculation from https://arxiv.org/abs/2403.03206.
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the img bloks. Note this is NOT using the parallel
        # attention layers from https://arxiv.org/abs/2302.05442, but rather
        # the standard MMDiT calculation from https://arxiv.org/abs/2403.03206.
        temporal = temporal + temporal_mod1.gate * self.temporal_attn.proj(
            temporal_attn
        )
        temporal = temporal + temporal_mod2.gate * self.temporal_mlp(
            (1 + temporal_mod2.scale) * self.temporal_norm2(img) + temporal_mod2.shift
        )

        # calculate the txt bloks. Note this is NOT using the parallel
        # attention layers from https://arxiv.org/abs/2302.05442, but rather
        # the standard MMDiT calculation from https://arxiv.org/abs/2403.03206.
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, temporal, txt
