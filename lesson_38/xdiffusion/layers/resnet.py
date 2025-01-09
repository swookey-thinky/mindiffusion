"""Utility layers used in defining a DDPM Unets with Dropout."""

import torch
from typing import Dict, Optional, Type

from xdiffusion.layers.utils import conv_nd, avg_pool_nd, zero_module, ContextBlock


class ResnetBlockDDPM(ContextBlock):
    """ResNet block based on WideResNet architecture.

    From DDPM, uses GroupNorm instead of weight normalization and Swish activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        time_emb_dim=None,
        use_scale_shift_norm=False,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self._use_scale_shift_norm = use_scale_shift_norm
        self.timestep_proj = (
            torch.nn.Sequential(
                torch.nn.SiLU(),
                torch.nn.Linear(
                    time_emb_dim, 2 * dim_out if use_scale_shift_norm else dim_out
                ),
            )
            if time_emb_dim is not None
            else None
        )

        self.block1 = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=dim_in),
            torch.nn.SiLU(),
            torch.nn.Conv2d(dim_in, dim_out, 3, padding=1),
        )

        # In the DDPM implementation, dropout was added to the second
        # resnet layer in the block, in front of the final convolution.
        self.block2 = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=dim_out),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity(),
            torch.nn.Conv2d(dim_out, dim_out, 3, padding=1),
        )

        self.residual_proj = torch.nn.Linear(dim_in, dim_out)
        self.dim_out = dim_out

    def forward(self, x, context: Dict):
        B, C, H, W = x.shape
        h = self.block1(x)

        # Add in the timstep embedding between blocks 1 and 2
        if "timestep_embedding" in context and self.timestep_proj is not None:
            time_emb = context["timestep_embedding"]
            emb_out = self.timestep_proj(time_emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]

            # Scale/Shift of the norm here is from the IDDPM paper,
            # as one of their improvements.
            if self._use_scale_shift_norm:
                out_norm, out_rest = self.block2[0], self.block2[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h += emb_out
                h = self.block2(h)

        # Project the residual channel to the output dimensions
        if C != self.dim_out:
            x = self.residual_proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return h + x


class ResnetBlockBigGAN(ContextBlock):
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
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        dim_in,
        time_emb_dim,
        dropout,
        dim_out=None,
        use_conv=False,
        use_scale_shift_norm=False,
        up=False,
        down=False,
        **kwargs,
    ):
        super().__init__()
        self.input_channels = dim_in
        self.emb_channels = time_emb_dim
        self.dropout = dropout
        self.out_channels = dim_out or dim_in
        self.use_conv = use_conv

        # 2D convolutions
        convolution_dims = 2

        # Scale/shift norm is called Adaptive Group Normalization
        # in the paper.
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=dim_in),
            torch.nn.SiLU(),
            conv_nd(convolution_dims, dim_in, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(dim_in, False, convolution_dims)
            self.x_upd = Upsample(dim_in, False, convolution_dims)
        elif down:
            self.h_upd = Downsample(dim_in, False, convolution_dims)
            self.x_upd = Downsample(dim_in, False, convolution_dims)
        else:
            self.h_upd = self.x_upd = torch.nn.Identity()

        self.emb_layers = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(
                time_emb_dim,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=self.out_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    convolution_dims, self.out_channels, self.out_channels, 3, padding=1
                )
            ),
        )

        if self.out_channels == dim_in:
            self.skip_connection = torch.nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                convolution_dims, dim_in, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(
                convolution_dims, dim_in, self.out_channels, 1
            )

    def forward(self, x, context: Dict):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb = context["timestep_embedding"]
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


class ResnetBlockEfficient(ContextBlock):
    """A residual block used in Efficient UNets.

    Notably, this block does NOT combine the time embeddings in the residual
    block, nor is this an upsampling or downsampling block. Figure A.27
    in the Imagen paper.
    """

    def __init__(
        self,
        dim_in,
        dropout=0.0,
        dim_out=None,
        scale_skip_connection: bool = True,
        **kwargs,
    ):
        super().__init__()

        self._input_channels = dim_in
        self._output_channels = dim_out or dim_in

        self._resnet_path = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=self._input_channels),
            torch.nn.SiLU(),
            conv_nd(2, self._input_channels, self._output_channels, 3, padding=1),
            torch.nn.GroupNorm(num_groups=32, num_channels=self._output_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout),
            zero_module(
                conv_nd(2, self._output_channels, self._output_channels, 3, padding=1)
            ),
        )

        self._scale_skip_connection = scale_skip_connection
        self._skip_connection = conv_nd(
            2, self._input_channels, self._output_channels, 1
        )

    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self._resnet_path(x)
        residual = self._skip_connection(x) + h

        if self._scale_skip_connection:
            # 1 / sqrt(2), https://arxiv.org/pdf/2104.07636.pdf
            residual *= 0.7071
        return residual


class DBlock(ContextBlock):
    """DBlock from Imagen.

    Figure A.28.
    """

    def __init__(
        self,
        dim_in,
        num_resnet_blocks,
        time_embedding_dim,
        downsample: bool,
        attention_type: Optional[Type],
        attention_kwargs: Dict = {},
        dropout=0.0,
        dim_out=None,
        **kwargs,
    ):
        super().__init__()

        self._input_channels = dim_in
        self._output_channels = dim_out or dim_in

        # Downsampling convolution, if we have one.
        if downsample:
            self._downsampling_convolution = conv_nd(
                2,
                self._input_channels,
                self._input_channels,
                3,
                padding=1,
                stride=2,
            )
        else:
            self._downsampling_convolution = torch.nn.Identity()

        # Combines the timestep embeddings
        self._embedding_layers = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(time_embedding_dim, self._input_channels),
        )

        # Resnet blocks
        self._resnet_blocks = torch.nn.Sequential(
            *[
                ResnetBlockEfficient(
                    dim_in=self._input_channels if idx == 0 else self._output_channels,
                    dim_out=self._output_channels,
                    dropout=dropout,
                )
                for idx in range(num_resnet_blocks)
            ]
        )

        # Attention (self or cross) layer, if it exists.
        if attention_type is not None:
            self._attention = attention_type(
                in_channels=self._output_channels, **attention_kwargs
            )
        else:
            self._attention = torch.nn.Identity()

    def forward(self, x, context: Dict):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self._downsampling_convolution(x)

        # Combine the embeddings
        timestep_embedding = context["timestep_embedding"]
        timestep_embedding_out = self._embedding_layers(timestep_embedding).type(
            h.dtype
        )

        while len(timestep_embedding_out.shape) < len(h.shape):
            timestep_embedding_out = timestep_embedding_out[..., None]
        h = h + timestep_embedding_out

        # Now the resnet blocks
        h = self._resnet_blocks(h)

        # Finally attention
        if isinstance(self._attention, ContextBlock):
            h = self._attention(h, context=context)
        else:
            h = self._attention(h)
        return h


class UBlock(ContextBlock):
    """DBlock from Imagen.

    Figure A.29.
    """

    def __init__(
        self,
        dim_in,
        num_resnet_blocks,
        time_embedding_dim,
        upsample: bool,
        attention_type: Optional[Type],
        attention_kwargs: Dict = {},
        dropout=0.0,
        dim_out=None,
        **kwargs,
    ):
        super().__init__()
        self._input_channels = dim_in
        self._output_channels = dim_out or dim_in

        # Downsampling convolution, if we have one. Otherwise this
        # is the initial convolution.
        if upsample:
            self._upsample = Upsample(channels=self._output_channels, use_conv=True)
        else:
            self._upsample = torch.nn.Identity()

        # Combines the timestep embeddings
        self._embedding_layers = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.nn.Linear(time_embedding_dim, self._input_channels),
        )

        # Resnet blocks
        self._resnet_blocks = torch.nn.Sequential(
            *[
                ResnetBlockEfficient(
                    dim_in=self._input_channels if idx == 0 else self._output_channels,
                    dim_out=self._output_channels,
                    dropout=dropout,
                )
                for idx in range(num_resnet_blocks)
            ]
        )

        # Attention (self or cross) layer, if it exists.
        if attention_type is not None:
            self._attention = attention_type(
                in_channels=self._output_channels, **attention_kwargs
            )
        else:
            self._attention = torch.nn.Identity()

    def forward(self, x, context: Dict):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # Combine the embeddings
        h = x
        timestep_embedding = context["timestep_embedding"]
        timestep_embedding_out = self._embedding_layers(timestep_embedding).type(
            h.dtype
        )

        while len(timestep_embedding_out.shape) < len(h.shape):
            timestep_embedding_out = timestep_embedding_out[..., None]
        h = h + timestep_embedding_out

        # Now the resnet blocks
        h = self._resnet_blocks(h)

        # Finally attention
        if isinstance(self._attention, ContextBlock):
            h = self._attention(h, context=context)
        else:
            h = self._attention(h)

        # Upsample
        h = self._upsample(h)
        return h


class Downsample(torch.nn.Module):
    """A downsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
            downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        kernel_size = 3 if dims != 3 else (1, 3, 3)
        padding = 1 if dims != 3 else (0, 1, 1)
        if use_conv:
            self.op = conv_nd(
                dims, channels, channels, kernel_size, stride=stride, padding=padding
            )
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(torch.nn.Module):
    """An upsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
            upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims

        kernel_size = 3 if dims != 3 else (1, 3, 3)
        padding = 1 if dims != 3 else (0, 1, 1)

        if use_conv:
            self.conv = conv_nd(dims, channels, channels, kernel_size, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = torch.nn.functional.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
