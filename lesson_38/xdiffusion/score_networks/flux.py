"""Score Network from Flux

Flux-Dev Params:     FluxParams(in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000, qkv_bias=True, guidance_embed=True)
Flux-Schnell Params: FluxParams(in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072, mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38, axes_dim=[16, 56, 56], theta=10000, qkv_bias=True, guidance_embed=False)
"""

from dataclasses import dataclass

from einops import rearrange, repeat
import torch
from torch import Tensor, nn
from typing import Dict

from xdiffusion.utils import DotConfig
from xdiffusion.layers.flux import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, config: DotConfig):
        super().__init__()

        self.config = config

        params = FluxParams(
            in_channels=config.in_channels,
            vec_in_dim=config.vec_in_dim,
            context_in_dim=config.context_in_dim,
            hidden_size=config.hidden_size,
            mlp_ratio=config.mlp_ratio,
            num_heads=config.num_heads,
            depth=config.depth,
            depth_single_blocks=config.depth_single_blocks,
            axes_dim=config.axes_dim,
            theta=config.theta,
            qkv_bias=config.qkv_bias,
            guidance_embed=config.guidance_embed,
        )
        self.patch_size = config.patch_size
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if params.guidance_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(self, x: Tensor, context: Dict, **kwargs) -> Tensor:
        # Pull out internal entries from the context
        guidance = (
            context["distillation_guidance"]
            if "distillation_guidance" in context
            else None
        )

        # Patch embed the input
        B, C, H, W = x.shape
        img = rearrange(
            x,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        img_ids = torch.zeros(
            H // self.patch_size, W // self.patch_size, 3, device=x.device
        )
        img_ids[..., 1] = (
            img_ids[..., 1]
            + torch.arange(H // self.patch_size, device=x.device)[:, None]
        )
        img_ids[..., 2] = (
            img_ids[..., 2]
            + torch.arange(W // self.patch_size, device=x.device)[None, :]
        )
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=B)

        # txt is the T5 text embeddings
        txt = context["t5_text_embeddings"]
        txt_ids = torch.zeros(B, txt.shape[1], 3, device=x.device)

        # y is the CLIP text embeddings
        y = context["clip_text_embeddings"]

        timesteps = context["timestep"]
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        # Unpatchify the output
        img = rearrange(
            img,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            ph=self.patch_size,
            pw=self.patch_size,
            h=H // self.patch_size,
            w=W // self.patch_size,
        )
        assert img.shape == x.shape
        return img
