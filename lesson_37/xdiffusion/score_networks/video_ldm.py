"""Score network for Video-LDM.

Wraps an existing spatial score network with temporal blocks, and freezes
the existing spatial layers.
"""

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from typing import Any, Dict, List, Union

from xdiffusion.layers.embedding import ContextEmbedSequential
from xdiffusion.layers.utils import ContextBlock, ContextIdentity
from xdiffusion.score_networks.unet import Unet
from xdiffusion.utils import (
    DotConfig,
)


class Conv3DLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_frames):
        super().__init__()

        self.to_3d = Rearrange("(b t) c h w -> b c t h w", t=num_frames)
        self.to_2d = Rearrange("b c t h w -> (b t) c h w")

        k, p = (3, 1, 1), (1, 0, 0)
        self.block1 = torch.nn.Sequential(
            torch.nn.GroupNorm(32, in_dim),
            torch.nn.SiLU(),
            torch.nn.Conv3d(in_dim, out_dim, kernel_size=k, stride=1, padding=p),
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.GroupNorm(32, out_dim),
            torch.nn.SiLU(),
            torch.nn.Conv3d(out_dim, out_dim, kernel_size=k, stride=1, padding=p),
        )

        self.alpha = torch.nn.Parameter(torch.ones(1))

    def forward(self, x, context: Dict):
        h = self.to_3d(x)

        h = self.block1(h)
        h = self.block2(h)

        h = self.to_2d(h)

        with torch.no_grad():
            self.alpha.clamp_(0, 1)

        out = self.alpha * x + (1 - self.alpha) * h
        return out


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


class TemporalAttentionLayer(torch.nn.Module):
    def __init__(self, dim, num_frames, num_heads=8, kv_dim=None):
        super().__init__()
        self.num_frames = num_frames
        self.num_heads = num_heads

        self.pos_enc = PositionalEncoding(dim)

        head_dim = dim // num_heads
        proj_dim = head_dim * num_heads
        self.q_proj = torch.nn.Linear(dim, proj_dim, bias=False)

        kv_dim = kv_dim or dim
        self.k_proj = torch.nn.Linear(kv_dim, proj_dim, bias=False)
        self.v_proj = torch.nn.Linear(kv_dim, proj_dim, bias=False)
        self.o_proj = torch.nn.Linear(proj_dim, dim, bias=False)

        self.alpha = torch.nn.Parameter(torch.ones(1))

    def forward(self, q, context: Dict):
        skip = q

        kv = None
        mask = None

        if "text_embeddings" in context:
            kv = context["text_embeddings"]
        if "video_mask" in context:
            mask = context["video_mask"]

        bt, c, h, w = q.shape
        q = rearrange(q, "(b t) c h w -> b (h w) t c", t=self.num_frames)

        q = q + self.pos_enc(self.num_frames)

        kv = kv[:: self.num_frames] if kv is not None else q
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q = rearrange(q, "b hw t (heads d) -> b hw heads t d", heads=self.num_heads)
        k = rearrange(k, "b s (heads d) -> b 1 heads s d", heads=self.num_heads)
        v = rearrange(v, "b s (heads d) -> b 1 heads s d", heads=self.num_heads)

        # TODO: Add back in the video mask
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None)
        out = rearrange(out, "b hw heads t d -> b hw t (heads d)")
        out = self.o_proj(out)

        out = rearrange(out, "b (h w) t c -> (b t) c h w", h=h, w=w)

        with torch.no_grad():
            self.alpha.clamp_(0, 1)

        out = self.alpha * skip + (1 - self.alpha) * out
        return out


class VideoLDMUnet(Unet):
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
        self.temporal_downs = torch.nn.ModuleList([])
        for level, mult in enumerate(channel_multipliers):
            for _ in range(num_resnet_blocks[level]):
                ch = mult * num_features
                temporal_layers: List[Any] = [
                    Conv3DLayer(
                        in_dim=ch,
                        out_dim=ch,
                        num_frames=num_frames,
                    )
                ]
                if ds in attention_ds:
                    temporal_layers.append(
                        TemporalAttentionLayer(
                            dim=ch,
                            num_frames=num_frames,
                            num_heads=config.spatial_score_network.conditioning.context_transformer_layer.params.heads,
                            kv_dim=config.spatial_score_network.conditioning.context_transformer_layer.params.context_dim,
                        )
                    )
                self.temporal_downs.append(ContextEmbedSequential(*temporal_layers))
                input_block_chans.append(ch)
            if level != len(channel_multipliers) - 1:
                # There is no temporal downsampling, so just add an
                # identity here.
                self.temporal_downs.append(ContextEmbedSequential(ContextIdentity()))
                input_block_chans.append(ch)
                ds *= 2

        # Middle layers
        self.temporal_middle = ContextEmbedSequential(
            Conv3DLayer(
                in_dim=ch,
                out_dim=ch,
                num_frames=num_frames,
            ),
            TemporalAttentionLayer(
                dim=ch,
                num_frames=num_frames,
                num_heads=config.spatial_score_network.conditioning.context_transformer_layer.params.heads,
                kv_dim=config.spatial_score_network.conditioning.context_transformer_layer.params.context_dim,
            ),
            Conv3DLayer(
                in_dim=ch,
                out_dim=ch,
                num_frames=num_frames,
            ),
        )

        self.temporal_ups = torch.nn.ModuleList([])
        for level, mult in list(enumerate(channel_multipliers))[::-1]:
            for i in range(num_resnet_blocks[level] + 1):
                ch = num_features * mult
                temporal_layers = [
                    Conv3DLayer(
                        in_dim=ch,
                        out_dim=ch,
                        num_frames=num_frames,
                    )
                ]
                if ds in attention_ds:
                    temporal_layers.append(
                        TemporalAttentionLayer(
                            dim=ch,
                            num_frames=num_frames,
                            num_heads=config.spatial_score_network.conditioning.context_transformer_layer.params.heads,
                            kv_dim=config.spatial_score_network.conditioning.context_transformer_layer.params.context_dim,
                        )
                    )
                if level and i == num_resnet_blocks[level]:
                    # There is no temporal upsampling, so just add an
                    # identity here.
                    temporal_layers.append(ContextIdentity())
                    ds //= 2
                self.temporal_ups.append(ContextEmbedSequential(*temporal_layers))

        # Run any special initializers
        def _custom_init(module):
            if hasattr(module, "custom_initializer"):
                module.custom_initializer()

        self.apply(_custom_init)

        self.requires_grad_(False)
        for t in self.temporal_ups:
            t.requires_grad_(True)
        self.temporal_middle.requires_grad_(True)
        for t in self.temporal_downs:
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
        assert len(self.ups) == len(self.temporal_ups)
        assert len(self.downs) == len(self.temporal_downs)

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
        for module, temporal_module in zip(self.downs, self.temporal_downs):
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
        for spatial, temporal in zip(self.middle, self.temporal_middle):
            h = spatial(h, context=context)
            h = temporal(h, context=context)

        # Interleave up blocks
        for module, temporal_module in zip(self.ups, self.temporal_ups):
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
