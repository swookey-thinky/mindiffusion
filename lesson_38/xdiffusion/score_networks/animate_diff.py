"""Score network for Animate-Diff.

Wraps an existing spatial score network with temporal blocks, and freezes
the existing spatial layers. Uses a temporal transformer for the temporal blocks,
without cross attention.
"""

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from typing import Any, Dict, List, Union

from xdiffusion.layers.embedding import ContextEmbedSequential
from xdiffusion.layers.transformer import FeedForward
from xdiffusion.layers.utils import ContextBlock, ContextIdentity, zero_module
from xdiffusion.score_networks.unet import Unet
from xdiffusion.utils import (
    DotConfig,
)


class PositionalEncoding(torch.nn.Module):
    """Basic sinusoidal position encoding for relative positions."""

    def __init__(self, dim, max_pos=512):
        super().__init__()

        pos = torch.arange(max_pos)

        freq = torch.arange(dim // 2) / dim
        freq = (freq * torch.tensor(10000).log()).exp()

        x = rearrange(pos, "L -> L 1") / freq
        x = rearrange(x, "L d -> L d 1")

        pe = torch.cat((x.sin(), x.cos()), dim=-1)
        self.pe = rearrange(pe, "L d sc -> L (d sc)")
        self.dummy = torch.nn.Parameter(torch.rand(1))

    def forward(self, length):
        enc = self.pe[:length]
        enc = enc.to(self.dummy.device)
        return enc


class TemporalSelfAttention(torch.nn.Module):
    def __init__(self, dim, num_frames, num_heads=8):
        super().__init__()
        self.num_frames = num_frames
        self.num_heads = num_heads

        self.pos_enc = PositionalEncoding(dim)

        head_dim = dim // num_heads
        proj_dim = head_dim * num_heads
        self.q_proj = torch.nn.Linear(dim, proj_dim, bias=False)

        self.k_proj = torch.nn.Linear(dim, proj_dim, bias=False)
        self.v_proj = torch.nn.Linear(dim, proj_dim, bias=False)
        self.o_proj = torch.nn.Linear(proj_dim, dim, bias=False)

        self.alpha = torch.nn.Parameter(torch.ones(1))

    def forward(self, q, context: Dict):
        skip = q

        kv = None
        mask = None

        if "video_mask" in context:
            mask = context["video_mask"]
        q = q + self.pos_enc(self.num_frames)
        kv = q

        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q = rearrange(q, "b hw t (heads d) -> b hw heads t d", heads=self.num_heads)
        k = rearrange(k, "b hw t (heads d) -> b hw heads t d", heads=self.num_heads)
        v = rearrange(v, "b hw t (heads d) -> b hw heads t d", heads=self.num_heads)

        # TODO: Add back in the video mask
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
        out = rearrange(out, "b hw heads t d -> b hw t (heads d)")
        out = self.o_proj(out)

        with torch.no_grad():
            self.alpha.clamp_(0, 1)

        out = self.alpha * skip + (1 - self.alpha) * out
        return out


class TemporalTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_frames: int,
        num_attention_heads: int,
        num_attention_blocks: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        attention_blocks = []
        norms = []

        for _ in range(num_attention_blocks):
            attention_blocks.append(
                TemporalSelfAttention(
                    dim=dim, num_frames=num_frames, num_heads=num_attention_heads
                )
            )
            norms.append(torch.nn.LayerNorm(dim))

        self.attention_blocks = torch.nn.ModuleList(attention_blocks)
        self.norms = torch.nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, glu=True)
        self.ff_norm = torch.nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Dict,
    ):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_x = norm(x)
            x = (
                attention_block(
                    norm_x,
                    context,
                )
                + x
            )

        x = self.ff(self.ff_norm(x)) + x
        return x


class TemporalTransformer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_frames: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_attention_blocks_per_layer: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self._num_frames = num_frames
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        self.proj_in = torch.nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_frames=num_frames,
                    num_attention_heads=num_attention_heads,
                    num_attention_blocks=num_attention_blocks_per_layer,
                    dropout=dropout,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = torch.nn.Linear(inner_dim, in_channels)
        self.proj_out = zero_module(self.proj_out)

    def forward(self, x: torch.Tensor, context: Dict):
        BT, C, H, W = x.shape
        residual = x

        x = rearrange(x, "(b t) c h w -> b (h w) t c", t=self._num_frames)

        x = self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.proj_in(x)

        # Transformer Blocks
        for block in self.transformer_blocks:
            x = block(
                x,
                context=context,
            )

        # output
        x = self.proj_out(x)
        x = rearrange(x, "b (h w) t c -> (b t) c h w", t=self._num_frames, h=H, w=W)

        return x + residual


class AnimateDiffUnet(Unet):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(
        self,
        config: DotConfig,
    ):
        """Initializes a new instance of Unet.

        Args:
            config: Model configuration parameters.
        """
        super().__init__(config.spatial_score_network)

        # Match the number of blocks so that we can insert all of the right
        # temporal layers.
        num_features = config.spatial_score_network.num_features
        num_frames = config.input_number_of_frames
        channel_multipliers = config.spatial_score_network.channel_multipliers
        channels = list(map(lambda x: num_features * x, channel_multipliers))

        attention_ds = []
        for res in config.spatial_score_network.attention.attention_resolutions:
            attention_ds.append(
                config.spatial_score_network.input_spatial_size // int(res)
            )

        # The number of resnet blocks in each layer.
        num_resnet_blocks = config.spatial_score_network.num_resnet_blocks
        if not isinstance(num_resnet_blocks, list):
            num_resnet_blocks = [num_resnet_blocks] * len(channel_multipliers)

        # Setup the downsampling, middle, and upsampling. This is just the temporal
        # layers, since the spatial layers are already accounted for. Insert
        # identity mappings for the attention if its not used.
        input_block_chans = [num_features]
        ch = num_features
        ds = 1
        self.motion_modules_down = torch.nn.ModuleList([])
        for level, mult in enumerate(channel_multipliers):
            for _ in range(num_resnet_blocks[level]):
                ch = mult * num_features

                # Add the motion module for this level. This is a temporal
                # transformer block, with self attention only (no cross attention
                # with the text embeddings).
                temporal_layers = []
                if ds in attention_ds:
                    # Identity for the resnet block in this layer
                    temporal_layers.append(ContextIdentity())

                    # Motion module after the attention block
                    temporal_layers.append(
                        TemporalTransformer(
                            in_channels=ch,
                            num_frames=num_frames,
                            num_attention_heads=config.motion_module.num_attention_heads,
                            num_attention_blocks_per_layer=config.motion_module.num_attention_blocks_per_layer,
                            attention_head_dim=config.motion_module.attention_head_dims,
                            num_layers=config.motion_module.num_layers,
                        )
                    )
                else:
                    # It's just the resnet block
                    temporal_layers.append(
                        TemporalTransformer(
                            in_channels=ch,
                            num_frames=num_frames,
                            num_attention_heads=config.motion_module.num_attention_heads,
                            num_attention_blocks_per_layer=config.motion_module.num_attention_blocks_per_layer,
                            attention_head_dim=config.motion_module.attention_head_dims,
                            num_layers=config.motion_module.num_layers,
                        )
                    )
                self.motion_modules_down.append(
                    ContextEmbedSequential(*temporal_layers)
                )
                input_block_chans.append(ch)
            if level != len(channel_multipliers) - 1:
                # There is no temporal downsampling, so just add an
                # identity here.
                self.motion_modules_down.append(
                    ContextEmbedSequential(ContextIdentity())
                )
                input_block_chans.append(ch)
                ds *= 2

        # Middle layers
        self.motion_modules_middle = ContextEmbedSequential(
            ContextIdentity(),
            TemporalTransformer(
                in_channels=ch,
                num_frames=num_frames,
                num_attention_heads=config.motion_module.num_attention_heads,
                num_attention_blocks_per_layer=config.motion_module.num_attention_blocks_per_layer,
                attention_head_dim=config.motion_module.attention_head_dims,
                num_layers=config.motion_module.num_layers,
            ),
            ContextIdentity(),
        )

        self.motion_modules_up = torch.nn.ModuleList([])
        for level, mult in list(enumerate(channel_multipliers))[::-1]:
            for i in range(num_resnet_blocks[level] + 1):
                ch = num_features * mult
                temporal_layers = []
                if ds in attention_ds:
                    # Identity for the resnet
                    temporal_layers.append(ContextIdentity())

                    # The temporal transformer after the attention
                    temporal_layers.append(
                        TemporalTransformer(
                            in_channels=ch,
                            num_frames=num_frames,
                            num_attention_heads=config.motion_module.num_attention_heads,
                            num_attention_blocks_per_layer=config.motion_module.num_attention_blocks_per_layer,
                            attention_head_dim=config.motion_module.attention_head_dims,
                            num_layers=config.motion_module.num_layers,
                        )
                    )
                else:
                    temporal_layers.append(
                        TemporalTransformer(
                            in_channels=ch,
                            num_frames=num_frames,
                            num_attention_heads=config.motion_module.num_attention_heads,
                            num_attention_blocks_per_layer=config.motion_module.num_attention_blocks_per_layer,
                            attention_head_dim=config.motion_module.attention_head_dims,
                            num_layers=config.motion_module.num_layers,
                        )
                    )

                if level and i == num_resnet_blocks[level]:
                    # There is no temporal upsampling, so just add an
                    # identity here.
                    temporal_layers.append(ContextIdentity())
                    ds //= 2
                self.motion_modules_up.append(ContextEmbedSequential(*temporal_layers))

        # Run any special initializers
        def _custom_init(module):
            if hasattr(module, "custom_initializer"):
                module.custom_initializer()

        self.apply(_custom_init)

        self.requires_grad_(False)
        for t in self.motion_modules_up:
            t.requires_grad_(True)
        self.motion_modules_middle.requires_grad_(True)
        for t in self.motion_modules_down:
            t.requires_grad_(True)

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
        assert len(self.ups) == len(self.motion_modules_up)
        assert len(self.downs) == len(self.motion_modules_down)

        # Transform the context at the top if we have it. This will generate
        # an embedding to combine with the timestep projection, and the embedded
        # context.
        for context_transformer in self._context_transformers:
            context = context_transformer(context, device=x.device)

        # Reshape the video data to spatial
        B, C, F, H, W = x.shape
        x = rearrange(x, "b c f h w -> (b f) c h w")

        # Initial convolution
        h = self._initial_convolution(x)

        hs = [h]

        # Interleave down blocks
        for module, temporal_module in zip(self.downs, self.motion_modules_down):
            # Each module is a sequential embedding, so need to interleave each block inside
            if isinstance(module, torch.nn.Sequential):
                assert isinstance(temporal_module, torch.nn.Sequential)
                for spatial, temporal in zip(module, temporal_module):
                    if isinstance(spatial, ContextBlock):
                        h = spatial(h, context=context)
                    else:
                        h = spatial(h)
                    h = temporal(h, context=context)
            else:
                h = module(h, context=context)
                h = temporal_module(h, context=context)
            hs.append(h)

        # Interleave middle blocks
        for spatial, temporal in zip(self.middle, self.motion_modules_middle):
            h = spatial(h, context=context)
            h = temporal(h, context=context)

        # Interleave up blocks
        for module, temporal_module in zip(self.ups, self.motion_modules_up):
            h = torch.cat([h, hs.pop()], dim=1)

            # Each module is a sequential embedding, so need to interleave each block inside
            if isinstance(module, torch.nn.Sequential):
                assert isinstance(temporal_module, torch.nn.Sequential)
                for spatial, temporal in zip(module, temporal_module):
                    if isinstance(spatial, ContextBlock):
                        h = spatial(h, context=context)
                    else:
                        h = spatial(h)

                    h = temporal(h, context=context)
            else:
                h = module(h, context=context)
                h = temporal_module(h, context=context)

        h = self.final_projection(h)

        # Reshape the video data to spatial
        h = rearrange(h, "(b f) c h w -> b c f h w", f=F)

        if self._is_learned_sigma:
            return torch.split(h, self._output_channels // 2, dim=1)
        return h
