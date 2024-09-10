"""Defines a 3D Unet for spatio-temporal noise prediction.

This package implements the 3D UNet from "Video Diffusion Models"
(https://arxiv.org/abs/2204.03458).
"""

import torch
from typing import Any, Dict, List, Union

from xdiffusion.layers.embedding import ContextEmbedSequential
from xdiffusion.layers.resnet import (
    Downsample,
    Upsample,
)
from xdiffusion.layers.resnet_3d import (
    ResnetBlockDDPM3D,
    ResnetBlockBigGAN3D,
)
from xdiffusion.layers.utils import EinopsToAndFrom
from xdiffusion.utils import (
    DotConfig,
    instantiate_from_config,
    instantiate_partial_from_config,
)


class Unet(torch.nn.Module):
    """A 3D time-dependent score-based model built upon U-Net architecture."""

    def __init__(
        self,
        config: DotConfig,
    ):
        """Initializes a new instance of Unet.

        Args:
            config: Model configuration parameters.
        """
        super().__init__()

        input_channels = config.input_channels
        self._output_channels = config.output_channels
        num_features = config.num_features
        channel_multipliers = config.channel_multipliers
        is_learned_sigma = config.is_learned_sigma
        dropout = config.dropout
        self._config = config

        self._is_class_conditional = config.is_class_conditional
        self._num_classes = config.num_classes
        self._is_learned_sigma = is_learned_sigma

        # Original DDPM paper had channel multipliers of [1,2,2,2] and input_channels = 128
        channels = list(map(lambda x: num_features * x, channel_multipliers))

        # Double the number of output channels if we are learning the variance.
        if is_learned_sigma:
            self._output_channels = input_channels * 2

        # The time embedding dimension was 4*num_features
        time_emb_dim = num_features * 4

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

        # Original paper implementation had kernel size = 3, stride = 1.
        # Updated to a 3D convolution, of shape (B, C, F, H, W)
        self._initial_convolution = torch.nn.Conv3d(
            in_channels=input_channels,
            out_channels=channels[0],
            kernel_size=(1, 3, 3),
            stride=(1, 1, 1),
            padding=(0, 1, 1),
            bias=False,
        )

        ResnetBlock = (
            ResnetBlockBigGAN3D
            if config.resnet_block_type == "biggan"
            else ResnetBlockDDPM3D
        )

        attention_ds = []
        for res in config.attention_resolutions:
            attention_ds.append(config.input_spatial_size // int(res))

        # The number of resnet blocks in each layer.
        num_resnet_blocks = config.num_resnet_blocks
        if not isinstance(num_resnet_blocks, list):
            num_resnet_blocks = [num_resnet_blocks] * len(channel_multipliers)

        # Setup the downsampling, middle, and upsampling pyramids
        # according to the configuration parameters.
        input_block_chans = [num_features]
        ch = num_features
        ds = 1
        self.downs = torch.nn.ModuleList([])
        for level, mult in enumerate(channel_multipliers):
            for _ in range(num_resnet_blocks[level]):
                layers: List[Any] = [
                    ResnetBlock(
                        dim_in=ch,
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                        dim_out=mult * num_features,
                        use_scale_shift_norm=config.use_scale_shift_norm,
                        use_conv=config.resamp_with_conv,
                        mlp_layers=config.mlp_layers,
                    )
                ]
                ch = mult * num_features
                if ds in attention_ds:
                    # The spatial attention layer needs to attend over the
                    # spatial dimensions, and expects an input of shape (B, L, D),
                    # where D will be the input channels 'ch'
                    layers.append(
                        EinopsToAndFrom(
                            from_einops="b c f h w",
                            to_einops="(b f) c h w",
                            fn=instantiate_partial_from_config(
                                config.conditioning.spatial_context_transformer_layer.to_dict()
                            )(
                                in_channels=ch,
                                context_projection_output_dim=(
                                    config.input_spatial_size // ds
                                )
                                ** 2,
                            ),
                        )
                    )
                    # The temporal attention needs to attend to the frames
                    # only.
                    layers.append(
                        EinopsToAndFrom(
                            from_einops="b c f h w",
                            to_einops="(b h w) c f",
                            fn=instantiate_partial_from_config(
                                config.conditioning.temporal_context_transformer_layer.to_dict()
                            )(in_channels=ch),
                        )
                    )
                self.downs.append(ContextEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_multipliers) - 1:
                self.downs.append(
                    ContextEmbedSequential(
                        ResnetBlock(
                            dim_in=ch,
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                            dim_out=ch,
                            use_scale_shift_norm=config.use_scale_shift_norm,
                            use_conv=config.resamp_with_conv,
                            down=True,
                            mlp_layers=config.mlp_layers,
                        )
                        if config.resblock_updown
                        else Downsample(
                            ch,
                            config.resamp_with_conv,
                            dims=3,
                        )
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        # Middle layers
        self.middle = ContextEmbedSequential(
            ResnetBlock(
                dim_in=ch,
                dim_out=ch,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                use_scale_shift_norm=config.use_scale_shift_norm,
                use_conv=config.resamp_with_conv,
                mlp_layers=config.mlp_layers,
            ),
            EinopsToAndFrom(
                from_einops="b c f h w",
                to_einops="(b f) c h w ",
                fn=instantiate_partial_from_config(
                    config.conditioning.spatial_context_transformer_layer.to_dict()
                )(
                    in_channels=ch,
                    context_projection_output_dim=(config.input_spatial_size // ds)
                    ** 2,
                ),
            ),
            EinopsToAndFrom(
                from_einops="b c f h w",
                to_einops="(b h w) c f",
                fn=instantiate_partial_from_config(
                    config.conditioning.temporal_context_transformer_layer.to_dict()
                )(in_channels=ch),
            ),
            ResnetBlock(
                dim_in=ch,
                dim_out=ch,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                use_scale_shift_norm=config.use_scale_shift_norm,
                use_conv=config.resamp_with_conv,
                mlp_layers=config.mlp_layers,
            ),
        )

        self.ups = torch.nn.ModuleList([])
        for level, mult in list(enumerate(channel_multipliers))[::-1]:
            for i in range(num_resnet_blocks[level] + 1):
                layers = [
                    ResnetBlock(
                        dim_in=ch + input_block_chans.pop(),
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                        dim_out=num_features * mult,
                        use_scale_shift_norm=config.use_scale_shift_norm,
                        use_conv=config.resamp_with_conv,
                        mlp_layers=config.mlp_layers,
                    )
                ]
                ch = num_features * mult
                if ds in attention_ds:
                    # The spatial attention layer needs to attend over the
                    # spatial dimensions, and expects an input of shape (B, L, D),
                    # where D will be the input channels 'ch'
                    layers.append(
                        EinopsToAndFrom(
                            from_einops="b c f h w",
                            to_einops="(b f) c h w",
                            fn=instantiate_partial_from_config(
                                config.conditioning.spatial_context_transformer_layer.to_dict()
                            )(
                                in_channels=ch,
                                context_projection_output_dim=(
                                    config.input_spatial_size // ds
                                )
                                ** 2,
                            ),
                        )
                    )
                    # The temporal attention needs to attend to the frames
                    # only.
                    layers.append(
                        EinopsToAndFrom(
                            from_einops="b c f h w",
                            to_einops="(b h w) c f",
                            fn=instantiate_partial_from_config(
                                config.conditioning.temporal_context_transformer_layer.to_dict()
                            )(in_channels=ch),
                        )
                    )
                if level and i == num_resnet_blocks[level]:
                    layers.append(
                        ResnetBlock(
                            dim_in=ch,
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                            dim_out=ch,
                            use_scale_shift_norm=config.use_scale_shift_norm,
                            use_conv=config.resamp_with_conv,
                            up=True,
                            mlp_layers=config.mlp_layers,
                        )
                        if config.resblock_updown
                        else Upsample(ch, config.resamp_with_conv, dims=3)
                    )
                    ds //= 2
                self.ups.append(ContextEmbedSequential(*layers))

        # Final projection
        self.final_projection = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=num_features),
            torch.nn.SiLU(),
            torch.nn.Conv3d(
                in_channels=num_features,
                out_channels=self._output_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 1, 1),
                padding=(0, 1, 1),
                bias=False,
            ),
        )

        # Run any special initializers
        def _custom_init(module):
            if hasattr(module, "custom_initializer"):
                module.custom_initializer()

        self.apply(_custom_init)

    def forward(
        self,
        x,
        context: Dict,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate noise parameter.

        Args:
            x: Tensor batch of noisy input data, shape (B, C, F, H, W)
            context: Dictionary of context tensors
        """
        # Don't change the original context
        context = context.copy()

        # Add the input batch to the context
        context["x"] = x

        # Transform the context at the top if we have it.
        for context_transformer in self._context_transformers:
            context = context_transformer(context=context, device=x.device)

        # Initial convolution
        h = self._initial_convolution(x)

        hs = [h]
        for module in self.downs:
            h = module(h, context=context)
            hs.append(h)
        h = self.middle(h, context=context)
        for module in self.ups:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, context=context)

        h = self.final_projection(h)

        if self._is_learned_sigma:
            return torch.split(h, self._output_channels // 2, dim=1)
        return h
