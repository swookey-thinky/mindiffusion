"""Utility layers used in defining a DDPM with Dropout."""

import torch

class Block(torch.nn.Module):
    """
    A convolutional block which makes up the two convolutional
    layers in the ResnetBlock.
    """

    def __init__(
        self,
        dim,
        dim_out,
        dropout=0.0,
    ):
        super().__init__()
        self.proj = torch.nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=dim)
        self.act = torch.nn.SiLU()
        self.dropout = (
            torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity()
        )

    def forward(self, x):
        # The original paper implementation uses norm->swish->projection
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x


class ResnetBlock(torch.nn.Module):
    """ResNet block based on WideResNet architecture.

    From DDPM, uses GroupNorm instead of weight normalization and Swish activation.
    """

    def __init__(
        self,
        dim,
        dim_out,
        time_emb_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.timestep_proj = (
            torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out)

        # In the DDPM implementation, dropout was added to the second
        # resnet layer in the block, in front of the final convolution.
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.residual_proj = torch.nn.Linear(dim, dim_out)
        self.dim_out = dim_out

    def forward(self, x, time_emb=None):
        B, C, H, W = x.shape

        h = self.block1(x)

        # Add in the timstep embedding between blocks 1 and 2
        if time_emb is not None and self.timestep_proj is not None:
            h += self.timestep_proj(time_emb)[:, :, None, None]

        h = self.block2(h)

        # Project the residual channel to the output dimensions
        if C != self.dim_out:
            x = self.residual_proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return h + x
