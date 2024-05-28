"""Defines the epsilon prediction U-Net.

U-Net espilon prediction network from the paper "Denoising Diffusion Probabilistic Models"
(https://arxiv.org/abs/2006.11239), with Dropout and class conditioning added.
"""

import torch
from typing import Any, List, Optional, Union

from attention import MultiHeadSelfAttention
from layers import (
    Downsample,
    SinusoidalPositionEmbedding,
    ResnetBlockDDPM,
    ResnetBlockBigGAN,
    TimestepEmbedSequential,
    Upsample,
)
from utils import DotConfig


class MNistUnet(torch.nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.

    When built as an epsilon-param model, the model only outputs epsilon, the
    re-parameterized estimate of the mean reverse process. When built as a v-param
    network, the model outputs both epsilon and v, a re-parameterized estimate of the
    variance of the reverse process model.
    """

    def __init__(
        self,
        config: DotConfig,
    ):
        """Initializes a new instance of MNistUnet.

        Args:
            config: Model configuration parameters.
        """
        super().__init__()

        input_channels = config.model.input_channels
        self._output_channels = config.model.output_channels
        num_features = config.model.num_features
        channel_multipliers = config.model.channel_multipliers
        is_v_param = config.model.is_v_param
        dropout = config.model.dropout

        self._is_class_conditional = config.model.is_class_conditional
        self._num_classes = config.data.num_classes
        self._is_v_param = is_v_param

        # Original paper had channel multipliers of [1,2,2,2] and input_channels = 128
        channels = list(map(lambda x: num_features * x, channel_multipliers))
        if is_v_param:
            self._output_channels = input_channels * 2

        # The time embedding dimension was 4*input_channels
        time_emb_dim = num_features * 4

        # Timestep embedding projection
        self.time_proj = torch.nn.Sequential(
            SinusoidalPositionEmbedding(num_features),
            torch.nn.Linear(num_features, time_emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_emb_dim, time_emb_dim),
        )

        if self._is_class_conditional:
            # We add 1 to the number of classes so that we can embed
            # a NULL token.
            self.label_proj = torch.nn.Embedding(self._num_classes + 1, time_emb_dim)

        # Original paper implementation had kernel size = 3, stride = 1
        self.initial_convolution = torch.nn.Conv2d(
            in_channels=input_channels,
            out_channels=channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        ResnetBlock = (
            ResnetBlockBigGAN
            if config.model.resnet_block_type == "biggan"
            else ResnetBlockDDPM
        )

        attention_ds = []
        for res in config.model.attention_resolutions:
            attention_ds.append(config.model.input_spatial_size // int(res))

        # Setup the downsampling, middle, and upsampling pyramids
        # according to the configuration parameters.
        input_block_chans = [num_features]
        ch = num_features
        ds = 1
        self.downs = torch.nn.ModuleList([])
        for level, mult in enumerate(channel_multipliers):
            for _ in range(config.model.num_resnet_blocks):
                layers: List[Any] = [
                    ResnetBlock(
                        dim_in=ch,
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                        dim_out=mult * num_features,
                        use_scale_shift_norm=config.model.use_scale_shift_norm,
                        use_conv=config.model.resamp_with_conv,
                    )
                ]
                ch = mult * num_features
                if ds in attention_ds:
                    layers.append(
                        MultiHeadSelfAttention(
                            ch, num_heads=config.model.num_attention_heads_downsample
                        )
                    )
                self.downs.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_multipliers) - 1:
                self.downs.append(
                    TimestepEmbedSequential(
                        ResnetBlock(
                            dim_in=ch,
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                            dim_out=ch,
                            use_scale_shift_norm=config.model.use_scale_shift_norm,
                            use_conv=config.model.resamp_with_conv,
                            down=True,
                        )
                        if config.model.resblock_updown
                        else Downsample(
                            ch,
                            config.model.resamp_with_conv,
                            dims=2,
                        )
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        # Middle layers
        self.middle = TimestepEmbedSequential(
            ResnetBlock(
                dim_in=ch,
                dim_out=ch,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                use_scale_shift_norm=config.model.use_scale_shift_norm,
                use_conv=config.model.resamp_with_conv,
            ),
            MultiHeadSelfAttention(
                ch, num_heads=config.model.num_attention_heads_downsample
            ),
            ResnetBlock(
                dim_in=ch,
                dim_out=ch,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                use_scale_shift_norm=config.model.use_scale_shift_norm,
                use_conv=config.model.resamp_with_conv,
            ),
        )

        self.ups = torch.nn.ModuleList([])
        for level, mult in list(enumerate(channel_multipliers))[::-1]:
            for i in range(config.model.num_resnet_blocks + 1):
                layers = [
                    ResnetBlock(
                        dim_in=ch + input_block_chans.pop(),
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                        dim_out=num_features * mult,
                        use_scale_shift_norm=config.model.use_scale_shift_norm,
                        use_conv=config.model.resamp_with_conv,
                    )
                ]
                ch = num_features * mult
                if ds in attention_ds:
                    layers.append(
                        MultiHeadSelfAttention(
                            ch, num_heads=config.model.num_attention_heads_downsample
                        )
                    )
                if level and i == config.model.num_resnet_blocks:
                    layers.append(
                        ResnetBlock(
                            dim_in=ch,
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                            dim_out=ch,
                            use_scale_shift_norm=config.model.use_scale_shift_norm,
                            use_conv=config.model.resamp_with_conv,
                            up=True,
                        )
                        if config.model.resblock_updown
                        else Upsample(ch, config.model.resamp_with_conv, dims=2)
                    )
                    ds //= 2
                self.ups.append(TimestepEmbedSequential(*layers))

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

    def forward(
        self,
        x,
        t,
        y: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        # Convert the timestep t to an embedding
        timestep_embedding = self.time_proj(t)

        if self._is_class_conditional:
            assert y is not None and y.shape == (x.shape[0],)
            timestep_embedding = timestep_embedding + self.label_proj(y)

        # Initial convolution
        h = self.initial_convolution(x)

        hs = [h]
        for module in self.downs:
            h = module(h, time_emb=timestep_embedding)
            hs.append(h)
        h = self.middle(h, time_emb=timestep_embedding)
        for module in self.ups:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, time_emb=timestep_embedding)

        h = self.final_projection(h)

        if self._is_v_param:
            return torch.split(h, self._output_channels, dim=1)
        return h
