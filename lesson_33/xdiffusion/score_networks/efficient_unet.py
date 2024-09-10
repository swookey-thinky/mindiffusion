"""Defines the noise prediction U-Net.

U-Net espilon prediction network from the paper "Denoising Diffusion Probabilistic Models"
(https://arxiv.org/abs/2006.11239).

This package has the following improvements over the original implementation:

This package adds the score network improvements from GLIDE. Namely, the model is trained
with classifier free guidance, and it uses a text conditioning scheme very similar to
Latent Diffusion. The difference is that Latent Diffusion uses a transformer+cross attention
projection at each UNet layer, while GLIDE uses a single transformer block, and only cross attention
at each layer.

This package augments the GLIDE text conditioning with the text and image conditioning
from DaLL*E 2.
"""

import torch
from typing import Dict, List, Union

from xdiffusion.layers.resnet import (
    ResnetBlockEfficient,
    UBlock,
    DBlock,
)
from xdiffusion.utils import (
    DotConfig,
    instantiate_from_config,
    instantiate_partial_from_config,
    type_from_config,
    kwargs_from_config,
)


class Unet(torch.nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

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

        self._label_projection = torch.nn.Identity()
        if self._is_class_conditional:
            # We add 1 to the number of classes so that we can embed
            # a NULL token.
            self._label_projection = torch.nn.Embedding(
                self._num_classes + 1, time_emb_dim
            )

        # Original paper implementation had kernel size = 3, stride = 1
        self._initial_convolution = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        ResnetBlock = ResnetBlockEfficient

        attention_ds = []
        for res in config.attention.attention_resolutions:
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
            downsample = True
            if ds in attention_ds:
                attention_type = type_from_config(
                    config.conditioning.context_transformer_layer.to_dict()
                )
                attention_kwargs = kwargs_from_config(
                    config.conditioning.context_transformer_layer.to_dict()
                )
            else:
                attention_type = None
                attention_kwargs = {}

            self.downs.append(
                DBlock(
                    dim_in=ch,
                    dim_out=mult * num_features,
                    dropout=dropout,
                    num_resnet_blocks=num_resnet_blocks[level],
                    time_embedding_dim=time_emb_dim,
                    downsample=downsample,
                    attention_type=attention_type,
                    attention_kwargs=attention_kwargs,
                )
            )
            ch = mult * num_features
            ds *= 2 if level != (len(channel_multipliers) - 1) else 1
            input_block_chans.append(ch)

        # Skip the last block input channels because the middle layers are not connected
        input_block_chans.pop()
        self.ups = torch.nn.ModuleList([])
        for level, mult in list(enumerate(channel_multipliers))[::-1]:
            upsample = True
            if ds in attention_ds:
                attention_type = type_from_config(
                    config.conditioning.context_transformer_layer.to_dict()
                )
                attention_kwargs = kwargs_from_config(
                    config.conditioning.context_transformer_layer.to_dict()
                )
            else:
                attention_type = None
                attention_kwargs = {}

            self.ups.append(
                UBlock(
                    dim_in=ch
                    + (
                        input_block_chans.pop()
                        if level != (len(channel_multipliers) - 1)
                        else 0
                    ),
                    dim_out=mult * num_features,
                    dropout=dropout,
                    num_resnet_blocks=num_resnet_blocks[level] + 1,
                    time_embedding_dim=time_emb_dim,
                    upsample=upsample,
                    attention_type=attention_type,
                    attention_kwargs=attention_kwargs,
                )
            )
            ch = num_features * mult
            ds //= 2 if upsample else 1

        # Final projection
        self.final_projection = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=num_features),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                in_channels=num_features,
                out_channels=self._output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
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
            x: Tensor batch of noisy input data.
            t: Tensor batch of timestep indices.
            y: (Optional) Tensor batch of integer class labels.
        """
        # Transform the context at the top if we have it. This will generate
        # an embedding to combine with the timestep projection, and the embedded
        # context.
        for context_transformer in self._context_transformers:
            context = context_transformer(context, device=x.device)

        # Initial convolution
        h = self._initial_convolution(x)

        hs = []
        for module in self.downs:
            h = module(h, context=context)
            hs.append(h)

        # Pop the last output because we don't use it.
        hs.pop()

        for idx, module in enumerate(self.ups):
            cat_in = h if idx == 0 else torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, context=context)

        h = self.final_projection(h)

        if self._is_learned_sigma:
            return torch.split(h, self._output_channels // 2, dim=1)
        return h
