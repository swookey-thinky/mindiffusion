"""Defines the epsilon prediction U-Net.

U-Net espilon prediction network from the paper "Denoising Diffusion Probabilistic Models"
(https://arxiv.org/abs/2006.11239), with Dropout and class conditioning added.

This network uses the conditioning projection architecture introduced in
"High-Resolution Image Synthesis with Latent Diffusion Models"
(https://arxiv.org/abs/2112.10752), which uses transformer blocks.

This is the same network as in Lesson 5d. However, we have added some
additional initialization configuration so that we can run it at the lower
latent dimensionality
"""

import torch

from layers import Identity, SinusoidalPositionEmbedding, ResnetBlock
from transformer import SpatialTransformer
from typing import Sequence


class ConditionalMNistUNet(torch.nn.Module):
    """A time-dependent score-based model built upon U-Net architecture.

    This model adds a transformer block, with cross attention, to project the
    context embeddings into the network. This is the same conditioning architecture
    as used in High-Resolution Image Synthesis with Latent Diffusion Models
    (https://arxiv.org/abs/2112.10752).
    """

    def __init__(
        self,
        context_dimension: int = 64,
        dropout: float = 0.0,
        model_initial_channels: int = 128,
        model_channel_multipliers: Sequence[int] = [1, 2, 2, 2],
        spatial_width: int = 32,
        attention_resolutions: Sequence[int] = [16],
    ):
        """Initialize a new instance of ConditionalMNistUNetWith.

        Original paper had channel multipliers of [1,2,2,2] and model_initial_channels = 128

        Args:
            content_dimension: The number of dimensions for the context embedding.
            dropout: Dropout rate, from [0,1].
            model_initial_channels:
        """
        super().__init__()

        input_channels = model_initial_channels
        channel_mults = model_channel_multipliers
        channels = list(map(lambda x: input_channels * x, channel_mults))

        # The time embedding dimension was 4*input_channels
        time_emb_dim = input_channels * 4

        # Timestep embedding projection
        self.time_proj = torch.nn.Sequential(
            SinusoidalPositionEmbedding(input_channels),
            torch.nn.Linear(input_channels, time_emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Original paper implementation had kernel size = 3, stride = 1
        self.initial_convolution = torch.nn.Conv2d(
            in_channels=1,
            out_channels=input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # The down/up sampling layers have 4 feature map resolutions for 32x32 input.
        # Note that we are padding the MNist dataset to 32x32 from 28x28 to make the math
        # works out easier. All resolution levels have two convolutional residual blocks,
        # and self-attention layers at the 16 level between the convolutions.
        number_of_layers = len(channels)

        layer_resolution = spatial_width
        down_layers = []
        for layer_idx in range(number_of_layers):

            layer_input_channels = (
                channels[layer_idx] if layer_idx != 0 else input_channels
            )
            layer_output_channels = channels[layer_idx]

            attention_block1 = (
                Identity()
                if layer_idx not in attention_resolutions
                else SpatialTransformer(
                    in_channels=layer_output_channels,
                    n_heads=1,
                    d_head=64,
                    context_dim=context_dimension,
                    dropout=dropout,
                )
            )

            attention_block2 = (
                Identity()
                if layer_idx not in attention_resolutions
                else SpatialTransformer(
                    in_channels=layer_output_channels,
                    n_heads=1,
                    d_head=64,
                    context_dim=context_dimension,
                    dropout=dropout,
                )
            )

            final_conv = (
                torch.nn.Conv2d(
                    layer_output_channels,
                    layer_output_channels,
                    3,
                    padding=1,
                    stride=2,
                )
                if layer_idx < (number_of_layers - 1)
                else Identity()
            )

            layer = torch.nn.ModuleList(
                [
                    ResnetBlock(
                        dim=layer_input_channels,
                        dim_out=layer_output_channels,
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                    ),
                    attention_block1,
                    ResnetBlock(
                        dim=layer_output_channels,
                        dim_out=layer_output_channels,
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                    ),
                    attention_block2,
                    final_conv,
                ]
            )
            down_layers.append(layer)
            layer_resolution = layer_resolution // 2
        self.downs = torch.nn.ModuleList(down_layers)

        # Middle layers
        self.middle = torch.nn.ModuleList(
            [
                # Input (B, 256, 4, 4), Output (B, 256, 4, 4)
                ResnetBlock(
                    dim=channels[-1],
                    dim_out=channels[-1],
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                ),
                SpatialTransformer(
                    in_channels=channels[-1],
                    n_heads=1,
                    d_head=64,
                    context_dim=context_dimension,
                    dropout=dropout,
                ),
                # Input (B, 256, 4, 4), Output (B, 256, 4, 4)
                ResnetBlock(
                    dim=channels[-1],
                    dim_out=channels[-1],
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                ),
            ]
        )

        # The upsampling layers reverse the process from the downsampling
        # layers.
        up_layers = []
        for layer_idx in reversed(range(number_of_layers)):

            layer_input_channels = channels[layer_idx]
            layer_output_channels = (
                channels[layer_idx - 1] if layer_idx != 0 else input_channels
            )

            attention_block1 = (
                Identity()
                if layer_idx not in attention_resolutions
                else SpatialTransformer(
                    in_channels=layer_input_channels,
                    n_heads=1,
                    d_head=64,
                    context_dim=context_dimension,
                    dropout=dropout,
                )
            )

            attention_block2 = (
                Identity()
                if layer_idx not in attention_resolutions
                else SpatialTransformer(
                    in_channels=layer_input_channels,
                    n_heads=1,
                    d_head=64,
                    context_dim=context_dimension,
                    dropout=dropout,
                )
            )

            attention_block3 = (
                Identity()
                if layer_idx not in attention_resolutions
                else SpatialTransformer(
                    in_channels=layer_output_channels,
                    n_heads=1,
                    d_head=64,
                    context_dim=context_dimension,
                    dropout=dropout,
                )
            )

            final_conv = (
                torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode="nearest"),
                    torch.nn.Conv2d(
                        channels[layer_output_channels],
                        channels[layer_output_channels],
                        3,
                        padding=1,
                    ),
                )
                if layer_idx > 0
                else Identity()
            )

            layer = torch.nn.ModuleList(
                [
                    ResnetBlock(
                        dim=channels[layer_input_channels]
                        + channels[layer_input_channels],
                        dim_out=channels[layer_input_channels],
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                    ),
                    attention_block1,
                    ResnetBlock(
                        dim=channels[layer_input_channels]
                        + channels[layer_input_channels],
                        dim_out=channels[layer_input_channels],
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                    ),
                    attention_block2,
                    ResnetBlock(
                        dim=channels[layer_input_channels]
                        + channels[layer_output_channels],
                        dim_out=channels[layer_output_channels],
                        time_emb_dim=time_emb_dim,
                        dropout=dropout,
                    ),
                    attention_block3,
                    final_conv,
                ]
            )
            up_layers.append(layer)
            layer_resolution = layer_resolution * 2
        self.ups = torch.nn.ModuleList(up_layers)

        # Final projection
        self.final_projection = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=input_channels),
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                in_channels=input_channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

    def forward(self, x, t, y=None):
        # Convert the timestep t to an embedding
        timestep_embedding = self.time_proj(t)

        # Initial convolution
        h = self.initial_convolution(x)  # B,C=1,H,W -> B,C=32,H,W

        # Downsampling blocks
        skips = [h]
        for i, layer in enumerate(self.downs):
            block1, attn1, block2, attn2, downsample = layer
            h = block1(h, time_emb=timestep_embedding)
            h = attn1(h, y)
            skips.append(h)
            h = block2(h, time_emb=timestep_embedding)
            h = attn2(h, y)
            skips.append(h)
            h = downsample(h)

            if i != len(self.downs) - 1:
                skips.append(h)

        # Middle layers
        middle_block1, middle_attn, middle_block2 = self.middle
        h = middle_block1(h, time_emb=timestep_embedding)
        h = middle_attn(h, y)
        h = middle_block2(h, time_emb=timestep_embedding)

        for i, layer in enumerate(self.ups):
            block1, attn1, block2, attn2, block3, attn3, upsample = layer

            h = block1(torch.cat([h, skips.pop()], dim=1), time_emb=timestep_embedding)
            h = attn1(h, y)
            h = block2(torch.cat([h, skips.pop()], dim=1), time_emb=timestep_embedding)
            h = attn2(h, y)
            h = block3(torch.cat([h, skips.pop()], dim=1), time_emb=timestep_embedding)
            h = attn3(h, y)
            h = upsample(h)

        h = self.final_projection(h)
        return h
