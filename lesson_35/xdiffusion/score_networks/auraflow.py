"""Score Network for AuraFlow.

From the blog post at: https://blog.fal.ai/auraflow/

Code based on public diffusers implementation at:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/auraflow_transformer_2d.py
"""

import torch
from typing import Dict

from xdiffusion.utils import DotConfig
from xdiffusion.layers.attention_diffusers import Attention
from xdiffusion.layers.embedding import Timesteps, TimestepEmbedding
from xdiffusion.layers.norm import AdaLayerNormZero, FP32LayerNorm


class AuraFlow(torch.nn.Module):
    """Aurflow Transformer Model

    Basically a wide transformer based model with interleaved
    DiT and MMDiT blocks.
    """

    def __init__(self, config: DotConfig):
        super().__init__()

        self.config = config
        sample_size = config.input_spatial_size
        patch_size = config.patch_size
        in_channels = config.input_channels
        out_channels = config.out_channels
        num_mmdit_layers = config.num_mmdit_layers
        num_single_dit_layers = config.num_single_dit_layers
        attention_head_dim = config.attention_head_dim
        num_attention_heads = config.num_attention_heads
        joint_attention_dim = config.joint_attention_dim
        caption_projection_dim = config.caption_projection_dim
        pos_embed_max_size = config.pos_embed_max_size

        default_out_channels = in_channels
        self.out_channels = (
            out_channels if out_channels is not None else default_out_channels
        )
        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )

        self.pos_embed = AuraFlowPatchEmbed(
            height=self.config.input_spatial_size,
            width=self.config.input_spatial_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.input_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.context_embedder = torch.nn.Linear(
            self.config.joint_attention_dim,
            self.config.caption_projection_dim,
            bias=False,
        )
        self.time_step_embed = Timesteps(
            num_channels=256, scale=1000.0, flip_sin_to_cos=True
        )
        self.time_step_proj = TimestepEmbedding(
            in_channels=256, time_embed_dim=self.inner_dim
        )

        self.joint_transformer_blocks = torch.nn.ModuleList(
            [
                AuraFlowJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_mmdit_layers)
            ]
        )
        self.single_transformer_blocks = torch.nn.ModuleList(
            [
                AuraFlowSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for _ in range(self.config.num_single_dit_layers)
            ]
        )
        self.norm_out = AuraFlowPreFinalBlock(self.inner_dim, self.inner_dim)
        self.proj_out = torch.nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=False
        )

        # https://arxiv.org/abs/2309.16588
        # prevents artifacts in the attention maps
        self.register_tokens = torch.nn.Parameter(
            torch.randn(1, 8, self.inner_dim) * 0.02
        )

    def forward(self, x: torch.Tensor, context: Dict, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape

        # Apply patch embedding, timestep embedding, and project the caption embeddings.
        hidden_states = self.pos_embed(
            x
        )  # takes care of adding positional embeddings too.

        # Timestep, in the range [0,1]
        timestep = context["timestep"]
        temb = self.time_step_embed(timestep).to(dtype=next(self.parameters()).dtype)
        temb = self.time_step_proj(temb)

        # txt is the T5 text embeddings
        encoder_hidden_states = context["t5_text_embeddings"]
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_states = torch.cat(
            [
                self.register_tokens.repeat(encoder_hidden_states.size(0), 1, 1),
                encoder_hidden_states,
            ],
            dim=1,
        )

        # MMDiT blocks.
        for _, block in enumerate(self.joint_transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
            )

        # Single DiT blocks that combine the `hidden_states` (image) and `encoder_hidden_states` (text)
        if len(self.single_transformer_blocks) > 0:
            encoder_seq_len = encoder_hidden_states.size(1)
            combined_hidden_states = torch.cat(
                [encoder_hidden_states, hidden_states], dim=1
            )

            for _, block in enumerate(self.single_transformer_blocks):
                combined_hidden_states = block(
                    hidden_states=combined_hidden_states, temb=temb
                )
            hidden_states = combined_hidden_states[:, encoder_seq_len:]

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        out_channels = self.config.out_channels
        height = H // patch_size
        width = W // patch_size

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                out_channels,
                height * patch_size,
                width * patch_size,
            )
        )
        return output


class AuraFlowPatchEmbed(torch.nn.Module):
    """AuraFlow Patch Embedding.

    Does not use convolutions for projection, and uses a learned
    (rather than sinusoidal) position embedding.
    """

    def __init__(
        self,
        height=32,
        width=32,
        patch_size=8,
        in_channels=1,
        embed_dim=768,
        pos_embed_max_size=None,
    ):
        super().__init__()

        self.num_patches = (height // patch_size) * (width // patch_size)
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = torch.nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        self.pos_embed = torch.nn.Parameter(
            torch.randn(1, pos_embed_max_size, embed_dim) * 0.1
        )

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size

    def pe_selection_index_based_on_dim(self, h, w):
        # select subset of positional embedding based on H, W, where H, W is size of latent
        # PE will be viewed as 2d-grid, and H/p x W/p of the PE will be selected
        # because original input are in flattened format, we have to flatten this 2d grid as well.
        h_p, w_p = h // self.patch_size, w // self.patch_size
        original_pe_indexes = torch.arange(self.pos_embed.shape[1])
        h_max, w_max = int(self.pos_embed_max_size**0.5), int(
            self.pos_embed_max_size**0.5
        )
        original_pe_indexes = original_pe_indexes.view(h_max, w_max)
        starth = h_max // 2 - h_p // 2
        endh = starth + h_p
        startw = w_max // 2 - w_p // 2
        endw = startw + w_p
        original_pe_indexes = original_pe_indexes[starth:endh, startw:endw]
        return original_pe_indexes.flatten()

    def forward(self, latent):
        B, C, H, W = latent.size()

        # Patchify the input latents
        latent = latent.view(
            B,
            C,
            H // self.patch_size,
            self.patch_size,
            W // self.patch_size,
            self.patch_size,
        )

        # (B, C, NH, P, NW, P) -> (B, NH, NW, C, P, P)
        # (B, NH, NW, C, P, P) - (B, NH, NW, C*P*P)
        # (B, NH, NW, C*P*P) -> (B, NH*NW, C*P*P) = (B, L, D)
        latent = latent.permute(0, 2, 4, 1, 3, 5).flatten(-3).flatten(1, 2)

        # (B, L, D) -> (B, L, D')
        latent = self.proj(latent)

        # Add the learned positional embeddings
        pe_index = self.pe_selection_index_based_on_dim(H, W)
        return latent + self.pos_embed[:, pe_index]


class AuraFlowSingleTransformerBlock(torch.nn.Module):
    """Similar to `AuraFlowJointTransformerBlock` with a single DiT instead of an MMDiT."""

    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")

        processor = AuraFlowAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="fp32_layer_norm",
            out_dim=dim,
            bias=False,
            out_bias=False,
            processor=processor,
        )

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff = AuraFlowFeedForward(dim, dim * 4)

    def forward(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor):
        residual = hidden_states

        # Norm + Projection.
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )

        # Attention.
        attn_output = self.attn(hidden_states=norm_hidden_states)

        # Process attention outputs for the `hidden_states`.
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(hidden_states)
        hidden_states = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = residual + hidden_states

        return hidden_states


class AuraFlowJointTransformerBlock(torch.nn.Module):
    r"""
    Transformer block for Aura Flow. Similar to SD3 MMDiT. Differences (non-exhaustive):

        * QK Norm in the attention blocks
        * No bias in the attention blocks
        * Most LayerNorms are in FP32

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        is_last (`bool`): Boolean to determine if this is the last block in the model.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim, bias=False, norm_type="fp32_layer_norm")
        self.norm1_context = AdaLayerNormZero(
            dim, bias=False, norm_type="fp32_layer_norm"
        )

        processor = AuraFlowAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            added_proj_bias=False,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="fp32_layer_norm",
            out_dim=dim,
            bias=False,
            out_bias=False,
            processor=processor,
            context_pre_only=False,
        )

        self.norm2 = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff = AuraFlowFeedForward(dim, dim * 4)
        self.norm2_context = FP32LayerNorm(dim, elementwise_affine=False, bias=False)
        self.ff_context = AuraFlowFeedForward(dim, dim * 4)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
    ):
        residual = hidden_states
        residual_context = encoder_hidden_states

        # Norm + Projection.
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            self.norm1_context(encoder_hidden_states, emb=temb)
        )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # Process attention outputs for the `hidden_states`.
        hidden_states = self.norm2(residual + gate_msa.unsqueeze(1) * attn_output)
        hidden_states = hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        hidden_states = gate_mlp.unsqueeze(1) * self.ff(hidden_states)
        hidden_states = residual + hidden_states

        # Process attention outputs for the `encoder_hidden_states`.
        encoder_hidden_states = self.norm2_context(
            residual_context + c_gate_msa.unsqueeze(1) * context_attn_output
        )
        encoder_hidden_states = (
            encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )
        encoder_hidden_states = c_gate_mlp.unsqueeze(1) * self.ff_context(
            encoder_hidden_states
        )
        encoder_hidden_states = residual_context + encoder_hidden_states

        return encoder_hidden_states, hidden_states


class AuraFlowPreFinalBlock(torch.nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()

        self.silu = torch.nn.SiLU()
        self.linear = torch.nn.Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=False
        )

    def forward(
        self, x: torch.Tensor, conditioning_embedding: torch.Tensor
    ) -> torch.Tensor:
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class AuraFlowFeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim=None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        final_hidden_dim = int(2 * hidden_dim / 3)
        final_hidden_dim = find_multiple(final_hidden_dim, 256)

        self.linear_1 = torch.nn.Linear(dim, final_hidden_dim, bias=False)
        self.linear_2 = torch.nn.Linear(dim, final_hidden_dim, bias=False)
        self.out_projection = torch.nn.Linear(final_hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(self.linear_1(x)) * self.linear_2(x)
        x = self.out_projection(x)
        return x


class AuraFlowAttnProcessor2_0:
    """Attention processor used typically in processing Aura Flow."""

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AuraFlowAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to at least 2.1 or above as we use `scale` in `F.scaled_dot_product_attention()`. "
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        # Reshape.
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        # Apply QK norm.
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Concatenate the projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            )
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            )
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            )

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_q(
                    encoder_hidden_states_key_proj
                )

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=1)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Attention.
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, scale=attn.scale, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, encoder_hidden_states.shape[1] :],
                hidden_states[:, : encoder_hidden_states.shape[1]],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
