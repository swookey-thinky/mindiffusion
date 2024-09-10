from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from xdiffusion.layers.embedding import CombinedTimestepTextProjEmbeddings
from xdiffusion.layers.sd3 import MMDiTBlock, AdaLayerNormContinuous, PatchEmbed
from xdiffusion.utils import DotConfig


class SD3Transformer2DModel(torch.nn.Module):
    """
    The Transformer model introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        out_channels (`int`, defaults to 16): Number of output channels.

    """

    def __init__(
        self,
        config: DotConfig,
    ):
        super().__init__()

        self._config = config
        default_out_channels = self._config.in_channels
        self.out_channels = (
            self._config.out_channels
            if "out_channels" in self._config.to_dict()
            else default_out_channels
        )
        self.inner_dim = (
            self._config.num_attention_heads * self._config.attention_head_dim
        )

        self.pos_embed = PatchEmbed(
            height=self._config.sample_size,
            width=self._config.sample_size,
            patch_size=self._config.patch_size,
            in_channels=self._config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=self._config.pos_embed_max_size,
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self._config.pooled_projection_dim,
        )
        self.context_embedder = torch.nn.Linear(
            self._config.joint_attention_dim, self._config.caption_projection_dim
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                MMDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self._config.num_attention_heads,
                    attention_head_dim=self._config.attention_head_dim,
                    context_pre_only=(i == self._config.num_layers - 1),
                )
                for i in range(self._config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=1e-6)
        self.proj_out = torch.nn.Linear(
            self.inner_dim,
            self._config.patch_size * self._config.patch_size * self.out_channels,
            bias=True,
        )

    def forward(
        self, x: torch.FloatTensor, context: Dict, **kwargs
    ) -> Union[torch.FloatTensor]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        hidden_states = x

        encoder_hidden_states = context["text_embeddings"]
        pooled_projections = context["pooled_text_embeddings"]
        timestep = context["timestep"]

        height, width = hidden_states.shape[-2:]
        hidden_states = self.pos_embed(
            hidden_states
        )  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                x=hidden_states,
                c=encoder_hidden_states,
                y=temb,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self._config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )
        return output
