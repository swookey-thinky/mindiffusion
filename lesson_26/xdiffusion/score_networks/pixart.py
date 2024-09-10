"""PixArt blocks and models, from https://pixart-alpha.github.io/"""

import torch
from typing import Dict

from xdiffusion.layers.attention import (
    MultiHeadSelfAttention,
    LastChannelCrossAttention,
)
from xdiffusion.layers.drop import DropPath
from xdiffusion.layers.embedding import PatchEmbed
from xdiffusion.layers.mlp import Mlp
from xdiffusion.layers.utils import get_2d_sincos_pos_embed
from xdiffusion.utils import (
    instantiate_from_config,
    instantiate_partial_from_config,
)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PixArtAlphaBlock(torch.nn.Module):
    """A PixArt block with adaptive layer norm (adaLN-single) conditioning."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        window_size=0,
        input_size=None,
        use_rel_pos=False,
        **block_kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            **block_kwargs,
        )
        self.cross_attn = LastChannelCrossAttention(
            query_dim=hidden_size,
            context_dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
        )
        self.norm2 = torch.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: torch.nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=0,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        self.window_size = window_size

        # The per-layer embedding (instead of the per-layer MLP used in DiT).
        # From the PixArt paper: "To utilize the aforementioned pretrained weights,
        # all E(i)’s are initialized to values that yield the same S(i)
        # as the DiT without c for a selected t (empirically, we use t = 500).
        # This design effectively replaces the layer-specific MLPs with a global MLP
        # and layer-specific trainable embeddings while preserving compatibility with
        # the pretrained weights from DiT."
        self.scale_shift_table = torch.nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )

    def forward(self, x, y, t, mask=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x = x + self.drop_path(
            gate_msa
            * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(
                B, N, C
            )
        )
        if y is not None:
            x = x + self.cross_attn(x, y)
        x = x + self.drop_path(
            gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        )

        return x


class PixArtAlphaFinalLayer(torch.nn.Module):
    """The final layer of PixArt.

    Incorporates the adaLN-Single.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = torch.nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = torch.nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True
        )
        self.scale_shift_table = torch.nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PixArtAlpha(torch.nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        config: Dict,
        **kwargs,
    ):
        super().__init__()

        self._config = config
        input_size = config.input_spatial_size
        patch_size = config.patch_size
        in_channels = config.input_channels
        hidden_size = config.hidden_size
        depth = config.depth
        num_heads = config.num_heads
        mlp_ratio = config.mlp_ratio
        pred_sigma = config.is_learned_sigma
        use_rel_pos = config.use_rel_pos
        lewei_scale = config.lewei_scale

        if "window_size" not in config:
            window_size = 0
        else:
            window_size = config.window_size

        if "window_block_indexes" not in config:
            window_block_indexes = []
        else:
            window_block_indexes = config.window_block_indexes

        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.lewei_scale = (lewei_scale,)

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )

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

        num_patches = self.x_embedder.num_patches
        self.base_size = input_size // self.patch_size
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, num_patches, hidden_size))

        # The adaLN-single block, applied once and shared across layers.
        self.t_block = torch.nn.Sequential(
            torch.nn.SiLU(), torch.nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Stochastic depth decay rule
        drop_path = [x.item() for x in torch.linspace(0, config.drop_path, depth)]

        self.blocks = torch.nn.ModuleList(
            [
                PixArtAlphaBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    input_size=(input_size // patch_size, input_size // patch_size),
                    window_size=window_size if i in window_block_indexes else 0,
                    use_rel_pos=use_rel_pos if i in window_block_indexes else False,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = PixArtAlphaFinalLayer(
            hidden_size, patch_size, self.out_channels
        )

        self.initialize_weights()

    def forward(self, x, context: Dict, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        # Don't change the original context
        context = context.copy()

        # Transform the context at the top if we have it.
        for context_transformer in self._context_transformers:
            context = context_transformer(context=context, device=x.device)

        pos_embed = self.pos_embed
        self.h, self.w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        x = (
            self.x_embedder(x) + pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        t = context["timestep_embedding"]  # (N, D)
        t0 = self.t_block(t)

        # If this is a class conditional model, then use the classes
        # as the context. Otherwise, use the text embeddings.
        y = None
        if "context_key" in self._config:
            y = context[self._config.context_key]

        # Create the mask for the cross attention bias
        y_lens = None
        # if y is not None and len(y.shape) > 2:
        #     y_lens = [y.shape[1]] * y.shape[0]
        #     # Squeeze y down into embedded shape of x, for the cross attention
        #     # block. x (patch embedded) comes in as (B, num_patches, hidden_size),
        #     # y comes in a (B, text_seq_len, text_context_dimension=hidden_size).
        #     # We squeeze y here into (1, text_seq_len*batch_size, hidden_size)
        #     y = y.view(1, -1, x.shape[-1])
        for block in self.blocks:
            x = block(x, y, t0, y_lens)  # (N, T, D)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def load_model_weights(self, state_dict: Dict):
        # Igore the position embeddings in the saved file.
        state_dict_keys = ["pos_embed", "base_model.pos_embed", "model.pos_embed"]
        for key in state_dict_keys:
            if key in state_dict:
                del state_dict[key]
                break
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print("Loaded model state dictionary.")
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def freeze_for_alignment(self):
        """Freezes all layers except the cross attention layers in the transformer blocks."""
        for param in self.parameters():
            param.requires_grad = False

        num_grads = 0
        for param in self.parameters():
            if param.requires_grad:
                num_grads += 1
        print(f"{num_grads} exists after freezing.")
        for block in self.blocks:
            for param in block.cross_attn.parameters():
                param.requires_grad = True
        num_grads = 0
        for param in self.parameters():
            if param.requires_grad:
                num_grads += 1
        print(f"{num_grads} exists after unfreezing cross attention.")

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches**0.5),
            lewei_scale=self.lewei_scale,
            base_size=self.base_size,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        torch.nn.init.normal_(self._projections["timestep"].mlp[0].weight, std=0.02)
        torch.nn.init.normal_(self._projections["timestep"].mlp[2].weight, std=0.02)
        torch.nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            # torch.nn.init.constant_(block.cross_attn.proj.weight, 0)
            # torch.nn.init.constant_(block.cross_attn.proj.bias, 0)
            torch.nn.init.constant_(block.cross_attn.to_out.weight, 0)
            torch.nn.init.constant_(block.cross_attn.to_out.bias, 0)

        # Zero-out output layers:
        torch.nn.init.constant_(self.final_layer.linear.weight, 0)
        torch.nn.init.constant_(self.final_layer.linear.bias, 0)

        # Run any special initializers
        def _custom_init(module):
            if hasattr(module, "custom_initializer"):
                module.custom_initializer()

        self.apply(_custom_init)
