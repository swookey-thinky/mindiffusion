"""Defines the epsilon prediction U-Net.

U-Net espilon prediction network from the paper "Denoising Diffusion Probabilistic Models"
(https://arxiv.org/abs/2006.11239), with Dropout and class conditioning added.

This network differs from Lesson 5c in the context embedding projection. That lesson
used a naive, straight forward cross attention projection. This network uses
the projection architecture introduced in High-Resolution Image Synthesis with
Latent Diffusion Models (https://arxiv.org/abs/2112.10752), which uses transformer
blocks.
"""

import torch

from layers import Identity, SinusoidalPositionEmbedding, ResnetBlock
from transformer import SpatialTransformer


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
    ):
        """Initialized a new instance of ConditionalMNistUNetWith.

        Args:
            content_dimension: The number of dimensions for the context embedding.
            dropout: Dropout rate, from [0,1].
        """
        super().__init__()

        # Original paper had channel multipliers of [1,2,2,2] and input_channels = 128
        input_channels = 128
        channel_mults = [1, 2, 2, 2]
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

        # For the conditioning, we are replacing the Self Attention blocks
        # with a combination of Self Attention and Cross Attention (on the conditioning
        # context).
        self.downs = torch.nn.ModuleList(
            [
                # Input (B, 128, 32, 32) Output (B, 128, 16, 16)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=input_channels,
                            dim_out=channels[0],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        ResnetBlock(
                            dim=channels[0],
                            dim_out=channels[0],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        torch.nn.Conv2d(
                            channels[0], channels[0], 3, padding=1, stride=2
                        ),
                    ]
                ),
                # Input (B, 128, 16 , 16) Output (B, 256, 8, 8)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[0],
                            dim_out=channels[1],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        SpatialTransformer(
                            in_channels=channels[1],
                            n_heads=1,
                            d_head=64,
                            context_dim=context_dimension,
                            dropout=dropout,
                        ),
                        ResnetBlock(
                            dim=channels[1],
                            dim_out=channels[1],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        SpatialTransformer(
                            in_channels=channels[1],
                            n_heads=1,
                            d_head=64,
                            context_dim=context_dimension,
                            dropout=dropout,
                        ),
                        torch.nn.Conv2d(
                            channels[1], channels[1], 3, padding=1, stride=2
                        ),
                    ]
                ),
                # Input (B, 256, 8, 8), Output (B, 256, 4, 4)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[1],
                            dim_out=channels[2],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        ResnetBlock(
                            dim=channels[2],
                            dim_out=channels[2],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        torch.nn.Conv2d(
                            channels[2], channels[2], 3, padding=1, stride=2
                        ),
                    ]
                ),
                # Input (B, 256, 4, 4), Output (B, 256, 4, 4)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[2],
                            dim_out=channels[3],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        ResnetBlock(
                            dim=channels[3],
                            dim_out=channels[3],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        Identity(),
                    ]
                ),
            ]
        )

        # Middle layers
        self.middle = torch.nn.ModuleList(
            [
                # Input (B, 256, 4, 4), Output (B, 256, 4, 4)
                ResnetBlock(
                    dim=channels[3],
                    dim_out=channels[3],
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                ),
                SpatialTransformer(
                    in_channels=channels[3],
                    n_heads=1,
                    d_head=64,
                    context_dim=context_dimension,
                    dropout=dropout,
                ),
                # Input (B, 256, 4, 4), Output (B, 256, 4, 4)
                ResnetBlock(
                    dim=channels[3],
                    dim_out=channels[3],
                    time_emb_dim=time_emb_dim,
                    dropout=dropout,
                ),
            ]
        )

        # Upsampling layers
        self.ups = torch.nn.ModuleList(
            [
                # Input (B, 256, 4, 4), Output (B, 256, 8, 8)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[3] + channels[3],
                            dim_out=channels[3],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        ResnetBlock(
                            dim=channels[3] + channels[3],
                            dim_out=channels[3],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        ResnetBlock(
                            dim=channels[3] + channels[2],
                            dim_out=channels[2],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=2, mode="nearest"),
                            torch.nn.Conv2d(channels[2], channels[2], 3, padding=1),
                        ),
                    ]
                ),
                # Input (B, 256, 8, 8), Output (B, 256, 16, 16)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[2] + channels[2],
                            dim_out=channels[2],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        ResnetBlock(
                            dim=channels[2] + channels[2],
                            dim_out=channels[2],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        ResnetBlock(
                            dim=channels[2] + channels[1],
                            dim_out=channels[1],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=2, mode="nearest"),
                            torch.nn.Conv2d(channels[1], channels[1], 3, padding=1),
                        ),
                    ]
                ),
                # Input (B, 256, 16, 16), Output (B, 256, 32, 32)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[1] + channels[1],
                            dim_out=channels[1],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        SpatialTransformer(
                            in_channels=channels[1],
                            n_heads=1,
                            d_head=64,
                            context_dim=context_dimension,
                            dropout=dropout,
                        ),
                        ResnetBlock(
                            dim=channels[1] + channels[1],
                            dim_out=channels[1],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        SpatialTransformer(
                            in_channels=channels[1],
                            n_heads=1,
                            d_head=64,
                            context_dim=context_dimension,
                            dropout=dropout,
                        ),
                        ResnetBlock(
                            dim=channels[1] + channels[0],
                            dim_out=channels[0],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        SpatialTransformer(
                            in_channels=channels[0],
                            n_heads=1,
                            d_head=64,
                            context_dim=context_dimension,
                            dropout=dropout,
                        ),
                        torch.nn.Sequential(
                            torch.nn.Upsample(scale_factor=2, mode="nearest"),
                            torch.nn.Conv2d(channels[0], channels[0], 3, padding=1),
                        ),
                    ]
                ),
                # Input (B, 128, 32, 32), Output (B, 128, 32, 32)
                torch.nn.ModuleList(
                    [
                        ResnetBlock(
                            dim=channels[0] + channels[0],
                            dim_out=channels[0],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        ResnetBlock(
                            dim=channels[0] + channels[0],
                            dim_out=channels[0],
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        ResnetBlock(
                            dim=channels[0] + input_channels,
                            dim_out=input_channels,
                            time_emb_dim=time_emb_dim,
                            dropout=dropout,
                        ),
                        Identity(),
                        Identity(),
                    ]
                ),
            ]
        )

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
