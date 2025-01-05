import torch
from typing import Dict, Optional

from xdiffusion.layers.attention_diffusers import Attention, AttnProcessor2_0
from xdiffusion.layers.embedding import PixArtAlphaTextProjection
from xdiffusion.layers.norm import AdaLayerNormSingle, RMSNorm
from xdiffusion.layers.sd3 import PatchEmbed
from xdiffusion.utils import DotConfig


class GLUMBConv(torch.nn.Module):
    """The Mix-FFN block from Sana."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4,
        norm_type: Optional[str] = None,
        residual_connection: bool = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        self.nonlinearity = torch.nn.SiLU()
        self.conv_inverted = torch.nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = torch.nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            1,
            1,
            groups=hidden_channels * 2,
        )
        self.conv_point = torch.nn.Conv2d(
            hidden_channels, out_channels, 1, 1, 0, bias=False
        )

        self.norm = None
        if norm_type == "rms_norm":
            self.norm = RMSNorm(
                out_channels, eps=1e-5, elementwise_affine=True, bias=True
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            residual = hidden_states

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)

        hidden_states = self.conv_point(hidden_states)

        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


class SanaLinearAttnProcessor2_0:
    """Linear attention processor used in Sana."""

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = query.transpose(1, 2).unflatten(1, (attn.heads, -1))
        key = key.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3)
        value = value.transpose(1, 2).unflatten(1, (attn.heads, -1))

        query = torch.nn.functional.relu(query)
        key = torch.nn.functional.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        value = torch.nn.functional.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = torch.matmul(value, key)
        hidden_states = torch.matmul(scores, query)

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + 1e-15)
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if original_dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class SanaTransformerBlock(torch.nn.Module):
    """SANA transformer block from https://arxiv.org/abs/2410.10629v3.

    Standard DiT block with Linear Attention from https://arxiv.org/abs/2006.16236.
    """

    def __init__(
        self,
        dim: int = 2240,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        dropout: float = 0.0,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_out_bias: bool = True,
        mlp_ratio: float = 2.5,
    ) -> None:
        super().__init__()

        # 1. Self Attention
        self.norm1 = torch.nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            processor=SanaLinearAttnProcessor2_0(),
        )
        self.norm2 = torch.nn.LayerNorm(
            dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps
        )

        # 2. Cross Attention
        if cross_attention_dim is not None:
            self.cross_attn = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_cross_attention_heads,
                dim_head=cross_attention_head_dim,
                dropout=dropout,
                bias=True,
                out_bias=attention_out_bias,
                processor=AttnProcessor2_0(),
            )

        # 3. Feed-forward
        self.ff = GLUMBConv(
            dim, dim, mlp_ratio, norm_type=None, residual_connection=False
        )

        self.scale_shift_table = torch.nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        height: int = None,
        width: int = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # 1. Modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # 2. Self Attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states)
        hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention
        if self.cross_attn is not None:
            attn_output = self.cross_attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(
            0, 3, 1, 2
        )
        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states


class SanaScoreNetwork(torch.nn.Module):
    """SANA score network from https://arxiv.org/abs/2410.10629v3."""

    def __init__(self, config: DotConfig):
        super().__init__()

        self.config = config

        sample_size = config.input_spatial_size
        patch_size = config.patch_size
        in_channels = config.in_channels
        out_channels = config.out_channels
        caption_channels = config.caption_channels
        num_attention_heads = config.num_attention_heads
        attention_head_dim = config.attention_head_dim
        num_cross_attention_heads = config.num_cross_attention_heads
        cross_attention_head_dim = config.cross_attention_head_dim
        cross_attention_dim = config.cross_attention_dim
        dropout = config.dropout
        mlp_ratio = config.mlp_ratio
        num_layers = config.num_layers
        inner_dim = num_attention_heads * attention_head_dim

        # Bias in the attention layer
        attention_bias = False
        norm_eps = 1e-6
        norm_elementwise_affine = False

        # Patch embed the input into transformer tokens. The exact patch embedding
        # strategy was not detailed in the technical report so we use the same
        # patch embedding strategy as PixArt-Alpha.
        self.patch_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=None,
            pos_embed_type=None,
        )

        # Timestep embedding. The timestep embedding was not detailed in the technical
        # report either, so we use PixArt-Alpha style embeddings.
        self.time_embed = AdaLayerNormSingle(inner_dim)

        self.caption_projection = PixArtAlphaTextProjection(
            in_features=caption_channels, hidden_size=inner_dim
        )

        # Elementwise_affine creates the learnable scale factor used in Sana
        self.caption_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)

        # The transformer stack
        # 3. Transformer blocks
        self.transformer_blocks = torch.nn.ModuleList(
            [
                SanaTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        # Output blocks
        self.scale_shift_table = torch.nn.Parameter(
            torch.randn(2, inner_dim) / inner_dim**0.5
        )

        self.norm_out = torch.nn.LayerNorm(
            inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = torch.nn.Linear(
            inner_dim, patch_size * patch_size * out_channels
        )

    def forward(self, x: torch.FloatTensor, context: Dict, **kwargs) -> torch.Tensor:
        hidden_states = x
        timestep = context["timestep"]
        encoder_hidden_states = context["text_embeddings"]

        if "text_attention_mask" in context:
            text_attention_masks = context["text_attention_mask"].to(torch.bool)
        else:
            text_attention_masks = None

        # Make sure the encoder hidden states and the hidden states
        # are the same dtype
        encoder_hidden_states = encoder_hidden_states.to(hidden_states.dtype)

        # Process the input
        B, C, H, W = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = H // p, W // p

        # Tokenize the image input via patches
        hidden_states = self.patch_embed(hidden_states)

        # Embed the timestep
        timestep, embedded_timestep = self.time_embed(
            timestep, batch_size=B, hidden_dtype=hidden_states.dtype
        )

        # Project the text captions
        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            B, -1, hidden_states.shape[-1]
        )
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # Run through all of the transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=text_attention_masks,
                timestep=timestep,
                height=post_patch_height,
                width=post_patch_width,
            )

        # Normalize the output
        shift, scale = (
            self.scale_shift_table[None]
            + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)

        # Final modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        hidden_states = hidden_states.reshape(
            B,
            post_patch_height,
            post_patch_width,
            self.config.patch_size,
            self.config.patch_size,
            -1,
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(
            B, -1, post_patch_height * p, post_patch_width * p
        )
        return output
