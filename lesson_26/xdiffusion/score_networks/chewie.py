"""CHEWIE: Spatio-temporal cascaded transformer score network.
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
class ChewieParams:
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


class Chewie(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, config: DotConfig):
        super().__init__()

        self.config = config

        params = ChewieParams(
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
        # Patch embed the input
        B, C, F, H, W = x.shape
        img = rearrange(
            x,
            "b c f (h ph) (w pw) -> b (f h w) (c ph pw)",
            ph=self.patch_size,
            pw=self.patch_size,
        )
        img_ids = torch.zeros(
            F, H // self.patch_size, W // self.patch_size, 3, device=x.device
        )

        # Add the position indices for position embedding (using ROPE).
        max_tokens = (H // self.patch_size) * (W // self.patch_size)
        for i in range(F):
            base_idx = i * max_tokens
            img_ids[i, :, :, 1] = (
                img_ids[i, :, :, 1]
                + torch.arange(
                    start=base_idx, end=base_idx + H // self.patch_size, device=x.device
                )[:, None]
            )
            img_ids[i, :, :, 2] = (
                img_ids[i, :, :, 2]
                + torch.arange(
                    start=base_idx, end=base_idx + W // self.patch_size, device=x.device
                )[None, :]
            )
        img_ids = repeat(img_ids, "f h w c -> b (f h w) c", b=B)

        # txt is the T5 text embeddings
        txt = context["t5_text_embeddings"]
        txt_ids = torch.zeros(B, txt.shape[1], 3, device=x.device)

        # y is the CLIP text embeddings
        y = context["clip_text_embeddings"]

        timesteps = context["timestep"]
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        joint_attention_mask = None
        if "is_image_batch" in context and context["is_image_batch"]:
            # The joint training mask tells us which frames are temporally
            # coherent video frames, and which frames are independent
            # images.
            TXT_T = txt.shape[1]
            IMG_T = img.shape[1]
            T = TXT_T + IMG_T
            joint_training_mask = torch.zeros((T, T), dtype=torch.bool, device=x.device)
            # Text tokens can attend amongst themselves
            joint_training_mask[:TXT_T, :TXT_T] = True

            # Make sure the image tokens can attend to the text tokens
            for i in range(IMG_T):
                joint_training_mask[TXT_T + i, :TXT_T] = True

            # Make sure the diagonal is all true (self attend)
            joint_training_mask[range(T), range(T)] = True
            joint_attention_mask = joint_training_mask
            # joint_attention_mask = torch.where(joint_training_mask, 0.0, float("-inf"))

            # Update the image_ids for positional embedding, since each frame is now
            # independent, so each frame will have the same image_id sequence.
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
            img_ids = repeat(img_ids, "h w c -> b (f h w) c", b=B, f=F)

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(
                img=img, txt=txt, vec=vec, pe=pe, attn_mask=joint_attention_mask
            )

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        # Unpatchify the output
        img = rearrange(
            img,
            "b (f h w) (c ph pw) -> b c f (h ph) (w pw)",
            ph=self.patch_size,
            pw=self.patch_size,
            h=H // self.patch_size,
            w=W // self.patch_size,
            f=F,
        )
        assert img.shape == x.shape
        return img
