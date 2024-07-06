from abc import abstractmethod
from einops.layers.torch import Rearrange
import math
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer
from typing import Callable, Dict, List, Optional, Tuple, Union

from image_diffusion.utils import freeze, prob_mask_like
from image_diffusion.layers.attention import AttentionPooling
from image_diffusion.layers.clip import FrozenCLIPTextEmbedder
from image_diffusion.layers.mlp import Mlp
from image_diffusion.layers.utils import ContextBlock, Format, nchw_to, to_2tuple

try:
    from torch import _assert
except ImportError:

    def _assert(condition: bool, message: str):
        assert condition, message


class ContextEmbedSequential(torch.nn.Sequential):
    """Sequential module for timestep and conditional embeddings.

    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, context):
        for layer in self:
            if isinstance(layer, ContextBlock):
                x = layer(x, context=context)
            else:
                x = layer(x)
        return x


class SinusoidalPositionEmbedding(torch.nn.Module):
    """Implementation of Sinusoidal Position Embedding.

    Originally introduced in the paper "Attention Is All You Need",
    the original tensorflow implementation is here:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py#L408
    """

    def __init__(self, embedding_dim, max_time: float, theta=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta
        self.max_time = max_time

    def forward(self, t, **kwargs):
        device = t.device

        x = t * 1000.0 / self.max_time

        half_dim = self.embedding_dim // 2
        embedding = math.log(self.theta) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=device) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding


class TimestepEmbeddingProjection(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        time_embedding_mult: int,
        max_time: float = 1000.0,
        **kwargs,
    ):
        super().__init__()
        time_embedding_dimension = num_features * time_embedding_mult
        self._projection = torch.nn.Sequential(
            SinusoidalPositionEmbedding(num_features, max_time=max_time),
            torch.nn.Linear(num_features, time_embedding_dimension),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embedding_dimension, time_embedding_dimension),
        )

    def forward(self, timestep: torch.Tensor, **kwargs):
        # Make sure there are no NaNs in the timestep embedding.
        # This is a debugging step because on my local 3090
        # this seems to happen sometimes, not sure why.
        projection = self._projection(timestep)
        if torch.isnan(projection).any():
            print(timestep)
            print(projection)
            assert False
        return projection


class InvCosTimestepEmbeddingProjection(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        time_embedding_mult: int,
        max_time: float = 1000.0,
        clip_min: int = -20,
        clip_max: int = 20,
    ):
        super().__init__()
        self._clip_min = clip_min
        self._clip_max = clip_max

        time_embedding_dimension = num_features * time_embedding_mult
        self._projection = torch.nn.Sequential(
            SinusoidalPositionEmbedding(num_features, max_time=max_time),
            torch.nn.Linear(num_features, time_embedding_dimension),
            torch.nn.SiLU(),
            torch.nn.Linear(time_embedding_dimension, time_embedding_dimension),
        )

    def forward(self, timestep: torch.Tensor, **kwargs):
        timestep_input = torch.arctan(
            torch.exp(-0.5 * torch.clip(timestep, self._clip_min, self._clip_max))
        ) / (0.5 * np.pi)

        # Make sure there are no NaNs in the timestep embedding.
        # This is a debugging step because on my local 3090
        # this seems to happen sometimes, not sure why.
        projection = self._projection(timestep_input)
        if torch.isnan(projection).any():
            print(timestep)
            print(projection)
            print(timestep_input)
            assert False
        return projection


class PooledTextEmbeddingsToTimestep(torch.nn.Module):
    def __init__(
        self,
        text_embedding_dim: int,
        time_embedding_dim: int,
        attention_pooling_heads: int,
        **kwargs,
    ):
        super().__init__()
        self._encoder_pooling = torch.nn.Sequential(
            torch.nn.LayerNorm(text_embedding_dim),
            AttentionPooling(attention_pooling_heads, text_embedding_dim),
            torch.nn.Linear(text_embedding_dim, time_embedding_dim),
            torch.nn.LayerNorm(time_embedding_dim),
        )

    def forward(self, context: Dict, **kwargs):
        assert "text_embeddings" in context
        assert "timestep_embedding" in context
        pooling_out = self._encoder_pooling(context["text_embeddings"])
        timestep_embedding = context["timestep_embedding"]
        timestep_embedding = timestep_embedding + pooling_out.to(timestep_embedding)
        context["timestep_embedding"] = timestep_embedding
        return context


class ImageEmbeddingProjection(torch.nn.Module):
    def __init__(self, context_size: int, num_image_tokens: int):
        super().__init__()
        self._project_image_embeddings = torch.nn.Sequential(
            torch.nn.Linear(
                context_size,
                context_size * num_image_tokens,
            ),
            Rearrange(
                "b (n d) -> b n d",
                n=num_image_tokens,
            ),
        )

    def forward(self, image_embedding: torch.Tensor, **kwargs):
        return self._project_image_embeddings(image_embedding)


class TextTokenProjection(torch.nn.Module):
    def __init__(self, token_vocabulary_size: int, width: int):
        super().__init__()
        self._projection = torch.nn.Embedding(
            token_vocabulary_size,
            width,
        )

    def forward(self, tokens, **kwargs):
        return self._projection(tokens)


class ContextProjection(torch.nn.Module):
    """Projects an entry in the context into a different dimensionality."""

    def __init__(
        self,
        input_context_key: str,
        output_context_key: str,
        in_features: int,
        hidden_features: int,
        out_features: int,
        custom_initialization: bool = False,
        **kwargs,
    ):
        super().__init__()
        self._input_context_key = input_context_key
        self._output_context_key = output_context_key
        self._custom_initialization = custom_initialization
        self.y_proj = Mlp(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            act_layer=lambda: torch.nn.GELU(approximate="tanh"),
            drop=0,
        )

    def forward(self, context: Dict, **kwargs):
        assert self._input_context_key in context
        context[self._output_context_key] = self.y_proj(
            context[self._input_context_key]
        )
        return context

    def custom_initializer(self):
        if self._custom_initialization:
            torch.nn.init.normal_(self.y_proj.fc1.weight, std=0.02)
            torch.nn.init.normal_(self.y_proj.fc2.weight, std=0.02)


class RunProjection(torch.nn.Module):
    """Runs a defined projection."""

    def __init__(
        self,
        input_context_key: str,
        output_context_key: str,
        projection_key: str,
        projections: torch.nn.ModuleDict,
        **kwargs,
    ):
        super().__init__()
        self._input_context_key = input_context_key
        self._output_context_key = output_context_key
        self._projection_key = projection_key
        self._projections = projections

    def forward(self, context: Dict, device, **kwargs):
        assert (
            self._input_context_key in context
        ), f"{self._input_context_key} not found for projection {self._projection_key}."
        assert self._projection_key in self._projections

        context[self._output_context_key] = self._projections[self._projection_key](
            context[self._input_context_key], context=context, device=device
        )
        return context


class CLIPTextTokenProjection(torch.nn.Module):
    def __init__(self, text_sequence_length: int):
        super().__init__()
        self._embedder = FrozenCLIPTextEmbedder(max_length=text_sequence_length)

    def forward(self, tokens, **kwargs):
        # Tokens come in as a dictionary from the CLIP text encoder
        assert "input_ids" in tokens and "attention_mask" in tokens
        with torch.no_grad():
            text_embeddings, last_hidden_state = self._embedder.embed(tokens=tokens)
        return last_hidden_state.detach()


class T5TextPromptsToTokens(torch.nn.Module):
    def __init__(self, max_length: int, model_name: str, **kwargs):
        super().__init__()
        self._max_length = max_length

        self._tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

    def forward(self, prompts, context: Dict, device, **kwargs):
        with torch.no_grad():
            tokens_dict = self._tokenizer(
                prompts,
                max_length=self._max_length,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )

        # Add the text tokens to the context
        text_inputs_on_device = {}
        for k, v in tokens_dict.items():
            text_inputs_on_device[k] = v.detach().to(device)
        return text_inputs_on_device


class T5TextTokensToEmbedding(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        self._text_encoder = freeze(T5EncoderModel.from_pretrained(model_name))

    def forward(self, tokens, context: Dict, **kwargs):
        # Tokens come in as a dictionary from the CLIP text encoder
        assert "input_ids" in tokens and "attention_mask" in tokens, f"{tokens}"
        with torch.no_grad():
            embedding_dict = self._text_encoder(**tokens)
        return embedding_dict["last_hidden_state"].detach()


class DiTTimestepEmbedding(torch.nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, timestep: torch.Tensor, **kwargs):
        t_freq = self.timestep_embedding(timestep, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

    def custom_initializer(self):
        # Initialize timestep embedding MLP:
        torch.nn.init.normal_(self.mlp[0].weight, std=0.02)
        torch.nn.init.normal_(self.mlp[2].weight, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding


class DiTLabelEmbedding(torch.nn.Module):
    """Class label embeddings for DiT.

    Embeds class labels into vector representations. Also handles label dropout
    for classifier-free guidance.
    """

    def __init__(
        self,
        num_classes,
        hidden_size,
        drop_prob: float = 0.0,
        unconditional_override: bool = False,
    ):
        super().__init__()

        # Add one for classifier free guidance, if we have it.
        self.embedding_table = torch.nn.Embedding(num_classes + 1, hidden_size)
        self.num_classes = num_classes
        self._unconditional_override = unconditional_override
        self._drop_prob = drop_prob

        # Initialize label embedding table:
        torch.nn.init.normal_(self.embedding_table.weight, std=0.02)

    def forward(self, labels, **kwargs):
        if self._unconditional_override:
            labels = torch.zeros_like(labels) + self.num_classes
        embeddings = self.embedding_table(labels)

        if self._drop_prob > 0.0:
            drop_mask = prob_mask_like(
                (embeddings.shape[0],), self._drop_prob, device=embeddings.device
            )
            null_classes_emb = torch.zeros_like(embeddings)
            embeddings = torch.where(drop_mask[:, None], null_classes_emb, embeddings)
        return embeddings


class DiTCombineEmbeddngs(torch.nn.Module):
    """Combines the timestep and labels into a single embedding."""

    def __init__(
        self,
        output_context_key: str,
        source_context_keys: List[str],
        projections: torch.nn.ModuleDict,
        **kwargs,
    ):
        super().__init__()
        self._output_context_key = output_context_key
        self._source_context_keys = source_context_keys
        self._projections = projections

    def forward(self, context: Dict, **kwargs):
        x = context[self._source_context_keys[0]]

        for key in self._source_context_keys[1:]:
            x += context[key]
        context[self._output_context_key] = x
        return context


class PatchEmbed(torch.nn.Module):
    """2D Image to Patch Embedding.

    Based on the implementation in timm from:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py
    """

    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
        self,
        img_size: Optional[int] = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = True,
        output_fmt: Optional[str] = None,
        bias: bool = True,
        strict_img_size: bool = True,
        dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple(
                [s // p for s, p in zip(self.img_size, self.patch_size)]
            )
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = torch.nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        self.norm = norm_layer(embed_dim) if norm_layer else torch.nn.Identity()

    def feat_ratio(self, as_scalar=True) -> Union[Tuple[int, int], int]:
        if as_scalar:
            return max(self.patch_size)
        else:
            return self.patch_size

    def dynamic_feat_size(self, img_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get grid (feature) size for given image size taking account of dynamic padding.
        NOTE: must be torchscript compatible so using fixed tuple indexing
        """
        if self.dynamic_img_pad:
            return math.ceil(img_size[0] / self.patch_size[0]), math.ceil(
                img_size[1] / self.patch_size[1]
            )
        else:
            return img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(
                    H == self.img_size[0],
                    f"Input height ({H}) doesn't match model ({self.img_size[0]}).",
                )
                _assert(
                    W == self.img_size[1],
                    f"Input width ({W}) doesn't match model ({self.img_size[1]}).",
                )
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).",
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).",
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x
