from abc import abstractmethod
from einops.layers.torch import Rearrange
import math
import torch
from typing import Dict, List

from image_diffusion.utils import freeze
from image_diffusion.layers.attention import AttentionPooling
from image_diffusion.layers.utils import ContextBlock


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

    def __init__(self, embedding_dim, theta=10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.theta = theta

    def forward(self, x, **kwargs):
        device = x.device
        half_dim = self.embedding_dim // 2
        embedding = math.log(self.theta) / (half_dim - 1)
        embedding = torch.exp(torch.arange(half_dim, device=device) * -embedding)
        embedding = x[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding


class TimestepEmbeddingProjection(torch.nn.Module):
    def __init__(self, num_features: int, time_embedding_mult: int):
        super().__init__()
        time_embedding_dimension = num_features * time_embedding_mult
        self._projection = torch.nn.Sequential(
            SinusoidalPositionEmbedding(num_features),
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

    def forward(self, context: Dict):
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


class T5TextTokensToEmbedding(torch.nn.Module):
    def __init__(self, model_name: str):
        super().__init__()

        from transformers import T5EncoderModel

        self._text_encoder = freeze(T5EncoderModel.from_pretrained(model_name))

    def forward(self, tokens, context: Dict, **kwargs):
        with torch.no_grad():
            text_encoder_embeddings = self._text_encoder(
                input_ids=tokens,
                attention_mask=context["text_tokens_attention_mask"],
            )["last_hidden_state"].detach()
            return text_encoder_embeddings
