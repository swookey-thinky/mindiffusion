"""Factorized space-time score network from Flexible Diffusion Modeling.


"""

from abc import abstractmethod
import numpy as np
import torch
from typing import Dict

from xdiffusion.layers.resnet import Downsample, Upsample
from xdiffusion.layers.utils import (
    conv_nd,
    zero_module,
    linear,
    normalization,
    timestep_embedding,
)
from xdiffusion.layers.attention import RPEAttention
from xdiffusion.utils import (
    instantiate_from_config,
    instantiate_partial_from_config,
    DotConfig,
)


class TimestepBlock(torch.nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedAttnThingsSequential(torch.nn.Sequential, TimestepBlock):
    """
    A sequential module that passes extra things to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, attn_mask, T=1, frame_indices=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                kwargs = dict(emb=emb)
                kwargs["emb"] = emb
            elif isinstance(layer, FactorizedAttentionBlock):
                kwargs = dict(
                    temb=emb,
                    attn_mask=attn_mask,
                    T=T,
                    frame_indices=frame_indices,
                )
            else:
                kwargs = {}
            x = layer(x, **kwargs)
        return x


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = torch.nn.Sequential(
            normalization(channels),
            torch.nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = torch.nn.Sequential(
            torch.nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = torch.nn.Sequential(
            normalization(self.out_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = torch.nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class FactorizedAttentionBlock(torch.nn.Module):

    def __init__(
        self,
        channels,
        num_heads,
        use_rpe_net,
        time_embed_dim=None,
    ):
        super().__init__()
        self.spatial_attention = RPEAttention(
            channels=channels,
            num_heads=num_heads,
            use_rpe_q=False,
            use_rpe_k=False,
            use_rpe_v=False,
        )
        self.temporal_attention = RPEAttention(
            channels=channels,
            num_heads=num_heads,
            time_embed_dim=time_embed_dim,
            use_rpe_net=use_rpe_net,
        )

    def forward(self, x, attn_mask, temb, T, frame_indices=None):
        BT, C, H, W = x.shape
        B = BT // T
        # reshape to have T in the last dimension because that's what we attend over
        x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1)  # B, H, W, C, T
        x = x.reshape(B, H * W, C, T)
        x = self.temporal_attention(
            x,
            temb,
            frame_indices,
            attn_mask=attn_mask.flatten(start_dim=2).squeeze(dim=2),  # B x T
        )

        # Now we attend over the spatial dimensions by reshaping the input
        x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2)  # B, T, C, H, W
        x = x.reshape(B, T, C, H * W)
        x = self.spatial_attention(
            x,
            temb,
            frame_indices=None,
        )
        x = x.reshape(BT, C, H, W)
        return x


class UNet(torch.nn.Module):
    """
    The full UNet model with attention and timestep embedding. This includes
    " adding an extra input channel which is all ones for observed frames and all
    zeros for latent frames."

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(self, config: DotConfig):
        super().__init__()

        in_channels = config.input_channels
        model_channels = config.model_channels
        out_channels = config.output_channels
        num_res_blocks = config.num_res_blocks
        attention_resolutions = config.attention_resolutions
        dropout = config.dropout
        channel_mult = config.channel_mult
        conv_resample = config.conv_resample
        dims = config.dims
        num_heads = config.num_heads
        num_heads_upsample = config.num_heads_upsample
        use_scale_shift_norm = config.use_scale_shift_norm
        use_rpe_net = config.use_rpe_net
        is_learned_sigma = config.is_learned_sigma

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels + 1
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.use_rpe_net = use_rpe_net
        self.is_learned_sigma = is_learned_sigma

        # Convert the attention resolutions into downsampled
        # factors
        attention_ds = []
        for res in self.attention_resolutions:
            attention_ds.append(config.input_spatial_size // int(res))

        if is_learned_sigma:
            self.out_channels = in_channels * 2

        # Instantiate all of the projections used in the model
        self._projections = torch.nn.ModuleDict()

        for projection_name in config.conditioning.signals:
            self._projections[projection_name] = instantiate_from_config(
                config.conditioning.projections[projection_name].to_dict()
            )

        # If we have a context transformer, let's create it here. This is
        # a GLIDE style transformer for embedding the text context
        # into both the timestep embedding and the attention layers,
        # and is applied at the top of the network once.
        if isinstance(config.conditioning.context_transformer_head, list):
            self._context_transformers = torch.nn.ModuleList([])
            for c in config.conditioning.context_transformer_head:
                self._context_transformers.append(
                    instantiate_partial_from_config(c)(projections=self._projections)
                )
        else:
            self._context_transformers = torch.nn.ModuleList(
                [
                    instantiate_partial_from_config(
                        config.conditioning.context_transformer_head.to_dict()
                    )(projections=self._projections)
                ]
            )

        time_embed_dim = model_channels * 4
        self.time_embed = torch.nn.Sequential(
            linear(model_channels, time_embed_dim),
            torch.nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = torch.nn.ModuleList(
            [
                TimestepEmbedAttnThingsSequential(
                    conv_nd(dims, self.in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_ds:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch,
                            num_heads=num_heads,
                            use_rpe_net=use_rpe_net,
                            time_embed_dim=time_embed_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedAttnThingsSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedAttnThingsSequential(
                        Downsample(ch, conv_resample, dims=dims)
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedAttnThingsSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            FactorizedAttentionBlock(
                ch,
                num_heads=num_heads,
                use_rpe_net=use_rpe_net,
                time_embed_dim=time_embed_dim,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = torch.nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_ds:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            use_rpe_net=use_rpe_net,
                            time_embed_dim=time_embed_dim,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedAttnThingsSequential(*layers))

        self.out = torch.nn.Sequential(
            normalization(ch),
            torch.nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(
        self,
        x,
        context: Dict,
    ):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        B, C, T, H, W = x.shape
        # Don't change the original context
        context = context.copy()

        # Transform the context at the top if we have it.
        for context_transformer in self._context_transformers:
            context = context_transformer(context=context, device=x.device)

        x0 = context["x0"]
        timesteps = context["timestep"]
        frame_indices = context["frame_indices"]
        obs_mask = context["observed_mask"]
        latent_mask = context["latent_mask"]

        # This model wants input in the order (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        x0 = x0.permute(0, 2, 1, 3, 4)
        obs_mask = obs_mask.permute(0, 2, 1, 3, 4)
        latent_mask = latent_mask.permute(0, 2, 1, 3, 4)

        timesteps = timesteps.view(B, 1).expand(B, T)
        attn_mask = (obs_mask + latent_mask).clip(max=1)
        # Add channel to indicate observations. From the paper,
        # "adding an extra input channel which is all ones for observed frames
        # and all zeros for latent frames"
        indicator_template = torch.ones_like(x[:, :, :1, :, :])
        obs_indicator = indicator_template * obs_mask
        x = torch.cat(
            [x * (1 - obs_mask) + x0 * obs_mask, obs_indicator],
            dim=2,
        )
        x = x.reshape(B * T, self.in_channels, H, W)
        timesteps = timesteps.reshape(B * T)
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = x.type(self.inner_dtype)
        for _, module in enumerate(self.input_blocks):
            h = module(
                h,
                emb,
                attn_mask,
                T=T,
                frame_indices=frame_indices,
            )
            hs.append(h)
        h = self.middle_block(h, emb, attn_mask, T=T, frame_indices=frame_indices)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(
                cat_in,
                emb,
                attn_mask,
                T=T,
                frame_indices=frame_indices,
            )
        h = h.type(x.dtype)
        out = self.out(h)
        h = out.view(B, T, self.out_channels, H, W)
        h = h.permute(0, 2, 1, 3, 4)

        if self.is_learned_sigma:
            return torch.split(h, self._output_channels // 2, dim=1)
        return h
