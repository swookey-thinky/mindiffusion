"""Pyramidal Flow score network.

From the paper "Pyramidal Flow Matching for Efficient Video Generative Modeling"
(https://arxiv.org/pdf/2410.05954).
"""

from einops import rearrange
import torch
from typing import Dict

from xdiffusion.layers.pyramid.embedding import (
    CombinedTimestepConditionEmbeddings,
    EmbedNDRoPE,
    PatchEmbed3D,
)
from xdiffusion.layers.pyramid.norm import AdaLayerNormContinuous
from xdiffusion.layers.pyramid.blocks import JointTransformerBlock
from xdiffusion.utils import DotConfig


class PyramidDiffusionMMDiT(torch.nn.Module):
    def __init__(
        self,
        config: DotConfig,
    ):
        super().__init__()

        self.config = config
        self.out_channels = config.input_channels
        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        assert config.temp_pos_embed_type in ["rope", "sincos"]

        # The input latent embeder, using the name pos_embed to remain the same with SD#
        self.pos_embed = PatchEmbed3D(
            height=config.input_spatial_size,
            width=config.input_spatial_size,
            patch_size=config.patch_size,
            in_channels=config.input_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=config.pos_embed_max_size,  # hard-code for now.
            max_num_frames=config.input_number_of_frames,
            pos_embed_type=config.pos_embed_type,
            temp_pos_embed_type=config.temp_pos_embed_type,
            add_temp_pos_embed=config.add_temp_pos_embed,
            interp_condition_pos=config.interp_condition_pos,
        )

        # The RoPE EMbedding
        if config.pos_embed_type == "rope":
            self.rope_embed = EmbedNDRoPE(self.inner_dim, 10000, axes_dim=[16, 24, 24])
        else:
            self.rope_embed = None

        if config.temp_pos_embed_type == "rope":
            self.temp_rope_embed = EmbedNDRoPE(
                self.inner_dim, 10000, axes_dim=[config.attention_head_dim]
            )
        else:
            self.temp_rope_embed = None

        self.time_text_embed = CombinedTimestepConditionEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
        )
        self.context_embedder = torch.nn.Linear(
            self.config.joint_attention_dim, self.config.caption_projection_dim
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=config.num_attention_heads,
                    attention_head_dim=self.inner_dim,
                    qk_norm=config.qk_norm,
                    context_pre_only=i == config.num_layers - 1,
                    use_flash_attn=config.use_flash_attn,
                )
                for i in range(config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = torch.nn.Linear(
            self.inner_dim,
            config.patch_size * config.patch_size * self.out_channels,
            bias=True,
        )
        self.patch_size = config.patch_size
        self.use_flash_attn = config.use_flash_attn
        self.use_temporal_causal = config.use_temporal_causal
        self.pos_embed_type = config.pos_embed_type
        self.temp_pos_embed_type = config.temp_pos_embed_type
        self.add_temp_pos_embed = config.add_temp_pos_embed

        if self.use_temporal_causal:
            print("Using temporal causal attention")
            assert (
                self.use_flash_attn is False
            ), "The flash attention does not support temporal causal"

        if config.interp_condition_pos:
            print("We interp the position embedding of condition latents")

        # init weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv3d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.pos_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.constant_(self.pos_embed.proj.bias, 0)

        # Initialize all the conditioning to normal init
        torch.nn.init.normal_(
            self.time_text_embed.timestep_embedder.linear_1.weight, std=0.02
        )
        torch.nn.init.normal_(
            self.time_text_embed.timestep_embedder.linear_2.weight, std=0.02
        )
        torch.nn.init.normal_(
            self.time_text_embed.text_embedder.linear_1.weight, std=0.02
        )
        torch.nn.init.normal_(
            self.time_text_embed.text_embedder.linear_2.weight, std=0.02
        )
        torch.nn.init.normal_(self.context_embedder.weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.transformer_blocks:
            torch.nn.init.constant_(block.norm1.linear.weight, 0)
            torch.nn.init.constant_(block.norm1.linear.bias, 0)
            torch.nn.init.constant_(block.norm1_context.linear.weight, 0)
            torch.nn.init.constant_(block.norm1_context.linear.bias, 0)

        # Zero-out output layers:
        torch.nn.init.constant_(self.norm_out.linear.weight, 0)
        torch.nn.init.constant_(self.norm_out.linear.bias, 0)
        torch.nn.init.constant_(self.proj_out.weight, 0)
        torch.nn.init.constant_(self.proj_out.bias, 0)

    @torch.no_grad()
    def _prepare_latent_image_ids(self, batch_size, temp, height, width, device):
        latent_image_ids = torch.zeros(temp, height, width, 3)
        latent_image_ids[..., 0] = (
            latent_image_ids[..., 0] + torch.arange(temp)[:, None, None]
        )
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[None, :, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, None, :]
        )

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1, 1)
        latent_image_ids = rearrange(latent_image_ids, "b t h w c -> b (t h w) c")
        return latent_image_ids.to(device=device)

    @torch.no_grad()
    def _prepare_pyramid_latent_image_ids(
        self, batch_size, temp_list, height_list, width_list, device
    ):
        base_width = width_list[-1]
        base_height = height_list[-1]
        assert base_width == max(width_list)
        assert base_height == max(height_list)

        image_ids_list = []
        for temp, height, width in zip(temp_list, height_list, width_list):
            latent_image_ids = torch.zeros(temp, height, width, 3)

            if height != base_height:
                height_pos = F.interpolate(
                    torch.arange(base_height)[None, None, :].float(),
                    height,
                    mode="linear",
                ).squeeze(0, 1)
            else:
                height_pos = torch.arange(base_height).float()
            if width != base_width:
                width_pos = F.interpolate(
                    torch.arange(base_width)[None, None, :].float(),
                    width,
                    mode="linear",
                ).squeeze(0, 1)
            else:
                width_pos = torch.arange(base_width).float()

            latent_image_ids[..., 0] = (
                latent_image_ids[..., 0] + torch.arange(temp)[:, None, None]
            )
            latent_image_ids[..., 1] = (
                latent_image_ids[..., 1] + height_pos[None, :, None]
            )
            latent_image_ids[..., 2] = (
                latent_image_ids[..., 2] + width_pos[None, None, :]
            )
            latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1, 1)
            latent_image_ids = rearrange(
                latent_image_ids, "b t h w c -> b (t h w) c"
            ).to(device)
            image_ids_list.append(latent_image_ids)

        return image_ids_list

    @torch.no_grad()
    def _prepare_temporal_rope_ids(
        self, batch_size, temp, height, width, device, start_time_stamp=0
    ):
        latent_image_ids = torch.zeros(temp, height, width, 1)
        latent_image_ids[..., 0] = (
            latent_image_ids[..., 0]
            + torch.arange(start_time_stamp, start_time_stamp + temp)[:, None, None]
        )
        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1, 1)
        latent_image_ids = rearrange(latent_image_ids, "b t h w c -> b (t h w) c")
        return latent_image_ids.to(device=device)

    @torch.no_grad()
    def _prepare_pyramid_temporal_rope_ids(self, sample, batch_size, device):
        image_ids_list = []

        for i_b, sample_ in enumerate(sample):
            if not isinstance(sample_, list):
                sample_ = [sample_]

            cur_image_ids = []
            start_time_stamp = 0

            for clip_ in sample_:
                _, _, temp, height, width = clip_.shape
                height = height // self.patch_size
                width = width // self.patch_size
                cur_image_ids.append(
                    self._prepare_temporal_rope_ids(
                        batch_size,
                        temp,
                        height,
                        width,
                        device,
                        start_time_stamp=start_time_stamp,
                    )
                )
                start_time_stamp += temp

            cur_image_ids = torch.cat(cur_image_ids, dim=1)
            image_ids_list.append(cur_image_ids)

        return image_ids_list

    def merge_input(self, sample, encoder_hidden_length, encoder_attention_mask):
        """
        Merge the input video with different resolutions into one sequence

        Args:
            sample: From low resolution to high resolution. At inference, this is a
                    list with a single item, which is itself another list which
                    contains the past conditions as well as the current set of latents
        """
        if isinstance(sample[0], list):
            device = sample[0][-1].device
            pad_batch_size = sample[0][-1].shape[0]
        else:
            device = sample[0].device
            pad_batch_size = sample[0].shape[0]

        num_stages = len(sample)
        height_list = []
        width_list = []
        temp_list = []
        trainable_token_list = []

        for i_b, sample_ in enumerate(sample):
            if isinstance(sample_, list):
                sample_ = sample_[-1]
            _, _, temp, height, width = sample_.shape
            height = height // self.patch_size
            width = width // self.patch_size
            temp_list.append(temp)
            height_list.append(height)
            width_list.append(width)
            trainable_token_list.append(height * width * temp)

        # prepare the RoPE embedding if needed
        if self.pos_embed_type == "rope":
            raise NotImplementedError("Not compatible with video generation now")
        else:
            if self.temp_pos_embed_type == "rope" and self.add_temp_pos_embed:
                image_ids_list = self._prepare_pyramid_temporal_rope_ids(
                    sample, pad_batch_size, device
                )
                text_ids = torch.zeros(
                    pad_batch_size, encoder_attention_mask.shape[1], 1
                ).to(device=device)
                input_ids_list = [
                    torch.cat([text_ids, image_ids], dim=1)
                    for image_ids in image_ids_list
                ]
                image_rotary_emb = [
                    self.temp_rope_embed(input_ids) for input_ids in input_ids_list
                ]  # [bs, seq_len, 1, head_dim // 2, 2, 2]
            else:
                image_rotary_emb = None

        hidden_states = self.pos_embed(
            sample
        )  # hidden states is a list of [b c t h w] b = real_b // num_stages
        hidden_length = []

        for i_b in range(num_stages):
            hidden_length.append(hidden_states[i_b].shape[1])

        # prepare the attention mask
        if self.use_flash_attn:
            attention_mask = None
            indices_list = []
            for i_p, length in enumerate(hidden_length):
                pad_attention_mask = torch.ones(
                    (pad_batch_size, length), dtype=encoder_attention_mask.dtype
                ).to(device)
                pad_attention_mask = torch.cat(
                    [encoder_attention_mask[i_p::num_stages], pad_attention_mask], dim=1
                )

                seqlens_in_batch = pad_attention_mask.sum(dim=-1, dtype=torch.int32)
                indices = torch.nonzero(
                    pad_attention_mask.flatten(), as_tuple=False
                ).flatten()

                indices_list.append(
                    {
                        "indices": indices,
                        "seqlens_in_batch": seqlens_in_batch,
                    }
                )
            encoder_attention_mask = indices_list
        else:
            assert encoder_attention_mask.shape[1] == encoder_hidden_length
            real_batch_size = encoder_attention_mask.shape[0]
            # prepare text ids
            text_ids = (
                torch.arange(1, real_batch_size + 1, dtype=encoder_attention_mask.dtype)
                .unsqueeze(1)
                .repeat(1, encoder_hidden_length)
            )
            text_ids = text_ids.to(device)
            text_ids[encoder_attention_mask == 0] = 0

            # prepare image ids
            image_ids = (
                torch.arange(1, real_batch_size + 1, dtype=encoder_attention_mask.dtype)
                .unsqueeze(1)
                .repeat(1, max(hidden_length))
            )
            image_ids = image_ids.to(device)
            image_ids_list = []
            for i_p, length in enumerate(hidden_length):
                image_ids_list.append(image_ids[i_p::num_stages][:, :length])

            attention_mask = []
            for i_p in range(len(hidden_length)):
                image_ids = image_ids_list[i_p]
                token_ids = torch.cat([text_ids[i_p::num_stages], image_ids], dim=1)
                stage_attention_mask = rearrange(
                    token_ids, "b i -> b 1 i 1"
                ) == rearrange(
                    token_ids, "b j -> b 1 1 j"
                )  # [bs, 1, q_len, k_len]
                if self.use_temporal_causal:
                    input_order_ids = input_ids_list[i_p].squeeze(2)
                    temporal_causal_mask = rearrange(
                        input_order_ids, "b i -> b 1 i 1"
                    ) >= rearrange(input_order_ids, "b j -> b 1 1 j")
                    stage_attention_mask = stage_attention_mask & temporal_causal_mask
                attention_mask.append(stage_attention_mask)

        return (
            hidden_states,
            hidden_length,
            temp_list,
            height_list,
            width_list,
            trainable_token_list,
            encoder_attention_mask,
            attention_mask,
            image_rotary_emb,
        )

    def split_output(
        self,
        batch_hidden_states,
        hidden_length,
        temps,
        heights,
        widths,
        trainable_token_list,
    ):
        # To split the hidden states
        batch_size = batch_hidden_states.shape[0]
        output_hidden_list = []
        batch_hidden_states = torch.split(batch_hidden_states, hidden_length, dim=1)

        for i_p, length in enumerate(hidden_length):
            width, height, temp = widths[i_p], heights[i_p], temps[i_p]
            trainable_token_num = trainable_token_list[i_p]
            hidden_states = batch_hidden_states[i_p]

            # only the trainable token are taking part in loss computation
            hidden_states = hidden_states[:, -trainable_token_num:]

            # unpatchify
            hidden_states = hidden_states.reshape(
                shape=(
                    batch_size,
                    temp,
                    height,
                    width,
                    self.patch_size,
                    self.patch_size,
                    self.out_channels,
                )
            )
            hidden_states = rearrange(
                hidden_states, "b t h w p1 p2 c -> b t (h p1) (w p2) c"
            )
            hidden_states = rearrange(hidden_states, "b t h w c -> b c t h w")
            output_hidden_list.append(hidden_states)

        return output_hidden_list

    def forward(
        self,
        x: torch.FloatTensor,
        context: Dict,
    ):
        """
        Args:
            sample: List of past conditions and latent inputs
            encoder_hidden_states: Prompt embeddings from the text encoder
            encoder_attention_mask: Attention mask from the text encoder
            pooled_projections: Pooled text embeddings from the text encoder
        """
        encoder_hidden_states = context.get("encoder_hidden_states", None)
        encoder_attention_mask = context.get("encoder_attention_mask", None)
        pooled_projections = context.get("pooled_projections", None)
        timestep_ratio = context.get("timestep_ratio", None)
        past_conditions = context.get("past_conditions", None)

        # Get the timestep embedding
        temb = self.time_text_embed(timestep_ratio, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_length = encoder_hidden_states.shape[1]

        # Merge the samples per stage and the input into the format expected below,
        # which is a list
        if past_conditions:
            sample = [past_conditions.append(x)]
        else:
            sample = [[x]]

        # Get the input sequence
        (
            hidden_states,
            hidden_length,
            temps,
            heights,
            widths,
            trainable_token_list,
            encoder_attention_mask,
            attention_mask,
            image_rotary_emb,
        ) = self.merge_input(sample, encoder_hidden_length, encoder_attention_mask)

        # split the long latents if necessary (only in sequence parallel,
        # which is not implemented here).
        hidden_states = torch.cat(hidden_states, dim=1)

        # print(hidden_length)
        for i_b, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                temb=temb,
                attention_mask=attention_mask,
                hidden_length=hidden_length,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, temb, hidden_length=hidden_length)
        hidden_states = self.proj_out(hidden_states)

        output = self.split_output(
            hidden_states, hidden_length, temps, heights, widths, trainable_token_list
        )

        return output
