"""Defines the epsilon prediction U-Net.

U-Net espilon prediction network from the paper "Denoising Diffusion Probabilistic Models"
(https://arxiv.org/abs/2006.11239), with Dropout and class conditioning added.

This package adds the score network improvements from GLIDE. Namely, the model is trained
with classifier free guidance, and it uses a text conditioning scheme very similar to
Latent Diffusion. The difference is that Latent Diffusion uses a transformer+cross attention
projection at each UNet layer, while GLIDE uses a single transformer block, and only cross attention
at each layer.

This package augments the GLIDE text conditioning with the text and image conditioning
from DaLL*E 2.
"""

from einops.layers.torch import Rearrange
import torch
from typing import Any, List, Optional, Union

from attention import MultiHeadCrossAttention
from layers import (
    Downsample,
    SinusoidalPositionEmbedding,
    ResnetBlockDDPM,
    ResnetBlockBigGAN,
    TimestepAndConditioningEmbedSequential,
    Upsample,
)
from transformer import LayerNorm, Transformer
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
        self._config = config

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

        # Projects the image embeddings into 4 additional tokens
        # of context.
        self._project_image_embeddings = torch.nn.Sequential(
            torch.nn.Linear(
                config.model.context_size,
                config.model.context_size * config.model.num_image_tokens,
            ),
            Rearrange("b (n d) -> b n d", n=config.model.num_image_tokens),
        )

        # If we have a context transformer, let's create it here. This is
        # a GLIDE style transformer for embedding the text context
        # into both the timestep embedding and the attention layers.
        transformer_width = None
        if "context_transformer" in config.model:
            transformer_width = (
                config.model.context_transformer.attention_channels
                * config.model.context_transformer.attention_heads
            )
            self._context_transformer = Transformer(
                context_size=config.model.context_size,
                layers=config.model.context_transformer.num_layers,
                attention_channels=config.model.context_transformer.attention_channels,
                attention_heads=config.model.context_transformer.attention_heads,
            )

            if config.model.context_transformer.final_layer_norm:
                self._final_layer_norm = LayerNorm(normalized_shape=transformer_width)
            else:
                self._final_layer_norm = None

            if config.model.context_transformer.padding:
                self._padding_embedding = torch.nn.Parameter(
                    torch.empty(
                        config.model.context_size,
                        transformer_width,
                        dtype=torch.float32,
                    )
                )
            else:
                self._padding_embedding = None

            self._positional_embedding = torch.nn.Parameter(
                torch.empty(
                    1,
                    config.model.context_size,
                    dtype=torch.float32,
                )
            )

            # Projects the context into the same dimensions as the timestep embedding.
            self._transformer_proj = torch.nn.Linear(transformer_width, time_emb_dim)

        else:
            self._context_transformer = None
            self._final_layer_norm = None
            self._padding_embedding = None

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
                        MultiHeadCrossAttention(
                            channels=ch,
                            encoder_channels=transformer_width,
                            num_heads=config.model.attention_heads,
                            num_head_channels=config.model.attention_channels,
                        )
                    )
                self.downs.append(TimestepAndConditioningEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_multipliers) - 1:
                self.downs.append(
                    TimestepAndConditioningEmbedSequential(
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
        self.middle = TimestepAndConditioningEmbedSequential(
            ResnetBlock(
                dim_in=ch,
                dim_out=ch,
                time_emb_dim=time_emb_dim,
                dropout=dropout,
                use_scale_shift_norm=config.model.use_scale_shift_norm,
                use_conv=config.model.resamp_with_conv,
            ),
            MultiHeadCrossAttention(
                channels=ch,
                encoder_channels=transformer_width,
                num_heads=config.model.attention_heads,
                num_head_channels=config.model.attention_channels,
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
                        MultiHeadCrossAttention(
                            channels=ch,
                            encoder_channels=transformer_width,
                            num_heads=config.model.attention_heads,
                            num_head_channels=config.model.attention_channels,
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
                self.ups.append(TimestepAndConditioningEmbedSequential(*layers))

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
        text_embeddings: Optional[torch.Tensor] = None,
        image_embeddings: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Calculate noise parameter.

        Args:
            x: Tensor batch of noisy input data.
            t: Tensor batch of timestep indices.
            y: (Optional) Tensor batch of integer class labels.
        """
        # Convert the timestep t to an embedding
        timestep_embedding = self.time_proj(t)
        assert text_embeddings is not None and image_embeddings is not None

        # Transform the context if we have it. The context is a combination
        # of the text embeddings, the positional text embeddings, and the
        # image embeddings.
        context = None
        if self._context_transformer is not None:
            context_outputs = self._get_context_embedding(
                image_embeddings,
                text_embeddings,
                dtype=timestep_embedding.dtype,
                mask=None,
            )
            xf_proj, context = context_outputs["xf_proj"], context_outputs["xf_out"]
            timestep_embedding = timestep_embedding + xf_proj.to(timestep_embedding)

        if self._is_class_conditional:
            assert y is not None and y.shape == (x.shape[0],)
            timestep_embedding = timestep_embedding + self.label_proj(y)

        # Initial convolution
        h = self.initial_convolution(x)

        hs = [h]
        for module in self.downs:
            h = module(h, time_emb=timestep_embedding, context=context)
            hs.append(h)
        h = self.middle(h, time_emb=timestep_embedding, context=context)
        for module in self.ups:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, time_emb=timestep_embedding, context=context)

        h = self.final_projection(h)

        if self._is_v_param:
            return torch.split(h, self._output_channels // 2, dim=1)
        return h

    def _get_context_embedding(self, image_embeddings, text_embeddings, mask, dtype):
        # context = text_embeddings

        # Project the image embeddings into the extra tokens. Both image and text
        # token arrays are shape [B, C, context_size].
        image_tokens = self._project_image_embeddings(image_embeddings)
        text_tokens = text_embeddings[:, None, :]

        # context = torch.cat([text_embeddings[:, None, :], image_tokens], dim=1)

        # First add the text tokens and their positional embedding
        xf_in = text_tokens
        xf_in = xf_in + self._positional_embedding[None]

        # Now add in the image tokens
        xf_in = torch.cat([xf_in, image_tokens], dim=-2)
        if self._config.model.context_transformer.padding:
            assert mask is not None
            xf_in = torch.where(mask[..., None], xf_in, self._padding_embedding[None])

        # Transform the context
        xf_out = self._context_transformer(xf_in.to(dtype))

        if self._final_layer_norm is not None:
            xf_out = self._final_layer_norm(xf_out)

        # Project the output
        xf_proj = self._transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)
        return outputs
