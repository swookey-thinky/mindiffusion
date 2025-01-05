"""Score Network from for Wide Transformers"""

from einops import rearrange, repeat
import torch
from torch import Tensor, nn
from typing import Dict, Optional

from xdiffusion.utils import DotConfig
from xdiffusion.layers.flux import (
    EmbedND,
    LastLayer,
    MLPEmbedder,
    DoubleStreamBlock,
    timestep_embedding,
)


class WideFormerSingleBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self._transformer_block = DoubleStreamBlock(
            hidden_size, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True
        )

        if in_channels != out_channels:
            self._token_mixer = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            )
        else:
            self._token_mixer = torch.nn.Identity()

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        h = self._token_mixer(img)
        h = self._transformer_block(img=h, txt=txt, vec=vec, pe=pe)
        return h


class WideFormer(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, config: DotConfig):
        super().__init__()

        self.config = config

        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )
        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if config.guidance_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)

        # The total sequence length is the number of text tokens (T5 max length) +
        # the number of image tokens ()
        image_sequence_length = (config.input_spatial_size // config.patch_size) ** 2
        text_sequence_length = config.max_text_tokens
        sequence_length = image_sequence_length + text_sequence_length

        self.transformer_channels = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        WideFormerSingleBlock(
                            self.hidden_size,
                            self.num_heads,
                            mlp_ratio=config.mlp_ratio,
                            in_channels=(
                                image_sequence_length
                                if layer_idx == 0
                                else image_sequence_length * config.transformer_width
                            ),
                            out_channels=image_sequence_length,
                        )
                        for _ in range(config.transformer_width)
                    ]
                )
                for layer_idx in range(config.depth)
            ]
        )
        self.transformer_final = WideFormerSingleBlock(
            self.hidden_size,
            self.num_heads,
            mlp_ratio=config.mlp_ratio,
            in_channels=image_sequence_length * config.transformer_width,
            out_channels=image_sequence_length,
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def forward(self, x: Tensor, context: Dict, **kwargs) -> Tensor:
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

        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # Apply the first layer
        B, L, D = img.shape
        layer_output = []

        for block in self.transformer_channels[0]:
            img_tokens, _ = block(img=img, txt=txt, vec=vec, pe=pe)
            layer_output.append(img_tokens)

        # Now concatenate the layer output and apply each following block
        for layer in self.transformer_channels[1:]:
            # This is a block of N transformers at each layer
            layer_input = layer_output
            layer_output = []

            # The layer input needs to be concatenated together
            layer_input = torch.cat(layer_input, dim=2).view(B, L * len(layer_input), D)
            for block in layer:
                img_tokens, _ = block(img=layer_input, txt=txt, vec=vec, pe=pe)
                layer_output.append(img_tokens)

        # Apply the final transformer layer
        layer_input = torch.cat(layer_output, dim=2).view(B, L * len(layer_output), D)
        img, _ = self.transformer_final(img=layer_input, txt=txt, vec=vec, pe=pe)

        # Parse out the image tokens
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
