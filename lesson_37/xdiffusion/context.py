"""Class for holding model conditioning.

Incorporates different conditioning available to the model, such
as timesteps, class labels, image embeddings, text embeddings, etc.

Possible conditioning signals include:

classes
timestep
timestep_embedding
text_tokens
text_embeddings
image_imbeddings
"""

from abc import abstractmethod
from einops import rearrange
import time
import torch
from transformers import (
    AutoTokenizer,
    T5Tokenizer,
    T5EncoderModel,
    CLIPTextModelWithProjection,
)
from typing import Dict, List, Optional, Union

from xdiffusion.layers.clip import FrozenCLIPTextTokenizer
from xdiffusion.tokenizer.bpe import get_encoder


class ContextAdapter(torch.nn.Module):
    """Basic block which accepts a context conditioning."""

    @abstractmethod
    def forward(self, context: Dict):
        """Apply the module to `x` given `context` conditioning."""


class NullContextAdapter(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, context: Dict):
        return None


class IgnoreContextAdapter(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, context: Dict, *args, **kwargs):
        return context


class IgnoreInputPreprocessor(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x


class UnconditionalTextPromptsAdapter(torch.nn.Module):
    def forward(self, context: Dict):
        new_context = context.copy()
        text_prompts = context["text_prompts"]
        new_context["text_prompts"] = [""] * len(text_prompts)
        return new_context


class UnconditionalEmbeddingAdapter(torch.nn.Module):
    def __init__(self, embedding_shape: List[int]):
        super().__init__()
        self.embedding_shape = embedding_shape
        assert len(embedding_shape) == 2
        num_tokens = embedding_shape[0]
        in_channels = embedding_shape[1]

        self.register_buffer(
            "y_embedding",
            torch.nn.Parameter(torch.randn(num_tokens, in_channels) / in_channels**0.5),
        )

    def forward(self, context: Dict):
        new_context = context.copy()
        embeddings = context["text_embeddings"]

        # Match the shape of the input embeddings
        unconditional_text_embeddings = self.y_embedding

        while len(unconditional_text_embeddings.shape) != len(embeddings.shape):
            unconditional_text_embeddings = unconditional_text_embeddings.unsqueeze(0)

        tile_shape = [1] * len(unconditional_text_embeddings.shape)
        tile_shape[0] = embeddings.shape[0]
        unconditional_text_embeddings = torch.tile(
            unconditional_text_embeddings, dims=tile_shape
        )
        assert (
            unconditional_text_embeddings.shape == embeddings.shape
        ), f"{unconditional_text_embeddings.shape} {embeddings.shape}"
        new_context["text_embeddings"] = unconditional_text_embeddings
        return new_context


class TextTokenAdapter(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, context: Dict):
        return context["text_tokens"]


class TextEmbeddingsAdapter(torch.nn.Module):
    def __init__(
        self,
        swap_context_channels: bool = False,
        input_projection_dim: int = -1,
        output_projection_dim: int = -1,
        **kwargs,
    ):
        super().__init__()
        self._swap_context_channels = swap_context_channels

        if output_projection_dim > 0 and input_projection_dim > 0:
            self._projection = torch.nn.Linear(
                input_projection_dim, output_projection_dim
            )
        else:
            self._projection = torch.nn.Identity()

    def forward(self, context: Dict):
        x = (
            context["text_embeddings"].permute(0, 2, 1)
            if self._swap_context_channels
            else context["text_embeddings"]
        )
        x = self._projection(x)
        return x


class TextTokenProjectionAdapter(torch.nn.Module):
    def __init__(self, projections: torch.nn.ModuleDict, **kwargs):
        super().__init__()
        self._projections = projections

    def forward(self, context: Dict, **kwargs):
        context["text_embeddings"] = self._projections["text_tokens"](
            context["text_tokens"], context=context
        )
        return context


class ContextEmbeddingAdapter(torch.nn.Module):
    def forward(self, context: Dict):
        return context["context_embedding"]


class UnconditionalTextPromptsAdapter(torch.nn.Module):
    def forward(self, context: Dict):
        new_context = context.copy()
        text_prompts = context["text_prompts"]
        new_context["text_prompts"] = [""] * len(text_prompts)
        return new_context


class UnconditionalClassesAdapter(torch.nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self._num_classes = num_classes

    def forward(self, context: Dict, **kwargs):
        new_context = context.copy()
        classes = context["classes"]
        new_context["classes"] = torch.zeros_like(classes) + self._num_classes
        return new_context


class TextPromptsPreprocessor(torch.nn.Module):
    def __init__(self, text_context_size: int, **kwargs):
        super().__init__()
        self._text_context_size = text_context_size
        self._text_encoder = get_encoder()

    def forward(self, context: Dict, device, **kwargs):
        if "text_prompts" in context:
            prompts = context["text_prompts"]

            with torch.no_grad():
                tokens = (
                    self._text_encoder.tokenize(
                        texts=prompts,
                        context_length=self._text_context_size,
                        truncate_text=True,
                    )
                    .detach()
                    .to(device)
                )

            # Add the text tokens to the context
            assert "text_tokens" not in context
            context["text_tokens"] = tokens
        return context


class CLIPTextPromptsPreprocessor(torch.nn.Module):
    def __init__(self, text_sequence_length: int, **kwargs):
        super().__init__()
        self._text_sequence_length = text_sequence_length
        self._text_tokenizer = FrozenCLIPTextTokenizer(max_length=text_sequence_length)

    def forward(self, context: Dict, device, **kwargs):
        start_time = time.perf_counter()
        if "text_prompts" in context:
            prompts = context["text_prompts"]

            with torch.no_grad():
                tokens_dict = self._text_tokenizer(prompts=prompts, device=device)

            # Add the text tokens to the context
            assert "text_tokens" not in context
            context["text_tokens"] = tokens_dict
        end_time = time.perf_counter()
        latency = end_time - start_time
        return context


class T5TextPromptsPreprocessor(torch.nn.Module):
    def __init__(self, max_length: int, model_name: str, **kwargs):
        super().__init__()
        self._max_length = max_length

        self._tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)

    def forward(self, context: Dict, device, **kwargs):
        if "text_prompts" in context:
            prompts = context["text_prompts"]
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
            assert "text_tokens" not in context
            text_inputs_on_device = {}
            for k, v in tokens_dict.items():
                text_inputs_on_device[k] = v.detach().to(device)
            context["text_tokens"] = text_inputs_on_device
        return context


class SD3TextPromptsPreprocessor(torch.nn.Module):
    def __init__(
        self,
        first_clip_model_name: str,
        first_clip_max_length: int,
        second_clip_model_name: str,
        second_clip_max_length: int,
        t5_model_name: str,
        t5_max_length: int,
    ):
        super().__init__()

        self._clip_tokenizer_1 = AutoTokenizer.from_pretrained(first_clip_model_name)
        self._clip_tokenizer_2 = AutoTokenizer.from_pretrained(second_clip_model_name)
        self._t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        self._clip_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
            first_clip_model_name
        )
        self._clip_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            second_clip_model_name
        )
        self._t5_encoder = T5EncoderModel.from_pretrained(t5_model_name)
        self._clip_tokenizer_1_max_length = first_clip_max_length
        self._clip_tokenizer_2_max_length = second_clip_max_length
        self._t5_max_length = t5_max_length

    def forward(self, context: Dict, device, **kwargs):
        if "text_prompts" in context:
            prompts = context["text_prompts"]
            with torch.no_grad():
                prompt_embeds_1, pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                    prompt=prompts,
                    clip_encoder=self._clip_encoder_1,
                    clip_tokenizer=self._clip_tokenizer_1,
                    max_length=self._clip_tokenizer_1_max_length,
                    device=device,
                )
                prompt_embeds_2, pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                    prompt=prompts,
                    clip_encoder=self._clip_encoder_2,
                    clip_tokenizer=self._clip_tokenizer_2,
                    max_length=self._clip_tokenizer_2_max_length,
                    device=device,
                )

                t5_prompt_embed = self._get_t5_prompt_embeds(
                    prompt=prompts,
                    max_sequence_length=self._t5_max_length,
                    device=device,
                    t5_tokenizer=self._t5_tokenizer,
                    t5_encoder=self._t5_encoder,
                )
                # Concatenate the CLIP prompt embeddings
                # "We also concatenate the penultimate hidden representations channel-wise to a CLIP
                # context conditioning c^CLIP_txt ∈ R^77×2048"
                clip_prompt_embeds = torch.cat(
                    [prompt_embeds_1, prompt_embeds_2], dim=-1
                )

                # Pad the CLIP prompt embeddings to the T5 prompt embeddings
                # "Finally, we zero-pad c^CLIP_txt along the channel axis to 4096 dimensions
                # to match the T5 representation"
                if t5_prompt_embed.shape[-1] > clip_prompt_embeds.shape[-1]:
                    clip_prompt_embeds = torch.nn.functional.pad(
                        clip_prompt_embeds,
                        (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
                    )
                elif clip_prompt_embeds.shape[-1] > t5_prompt_embed.shape[-1]:
                    t5_prompt_embed = torch.nn.functional.pad(
                        t5_prompt_embed,
                        (0, clip_prompt_embeds.shape[-1] - t5_prompt_embed.shape[-1]),
                    )

                # Concatentate the CLIP and T5 text embeddings
                # "and concatenate it along the sequence axis with c^T5_txt
                # to obtain the final context representation c_txt ∈ R^154×4096"
                prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

                # Concatenate the Pooled CLIP prompt embeddings
                # "We concatenate the pooled outputs, of sizes 768 and 1280 respectively, to obtain
                # a vector conditioning c_vec ∈ R^2048"
                pooled_prompt_embeds = torch.cat(
                    [pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1
                )
            context["text_embeddings"] = prompt_embeds.detach().to(device)
            context["pooled_text_embeddings"] = pooled_prompt_embeds.detach().to(device)
        return context

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        clip_encoder: CLIPTextModelWithProjection,
        clip_tokenizer: AutoTokenizer,
        max_length: int,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = clip_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = clip_tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )

        clip_encoder_device = clip_encoder.device
        clip_encoder = clip_encoder.to(device)
        prompt_embeds = clip_encoder(
            text_input_ids.to(device), output_hidden_states=True
        )
        clip_encoder = clip_encoder.to(clip_encoder_device)

        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        prompt_embeds = prompt_embeds.to(dtype=clip_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(
            batch_size * num_images_per_prompt, -1
        )

        return prompt_embeds, pooled_prompt_embeds

    def _get_t5_prompt_embeds(
        self,
        t5_tokenizer: AutoTokenizer,
        t5_encoder: T5EncoderModel,
        device: Optional[torch.device],
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 256,
        dtype: Optional[torch.dtype] = None,
    ):
        dtype = dtype or t5_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = t5_tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer_max_length - 1 : -1]
            )
            print(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        t5_device = t5_encoder.device
        t5_encoder = t5_encoder.to(device)
        prompt_embeds = t5_encoder(text_input_ids.to(device))[0]
        t5_encoder = t5_encoder.to(t5_device)

        dtype = t5_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds


class SpatialBatchForVideo(torch.nn.Module):
    """Convert spatial data into batched spatio-temporal data."""

    def __init__(
        self,
        input_context_key: str,
        num_frames: str,
        **kwargs,
    ):
        super().__init__()
        self._input_context_key = input_context_key
        self._num_frames = num_frames

    def forward(self, context: Dict, device, **kwargs):
        assert (
            self._input_context_key in context
        ), f"{self._input_context_key} not found for projection {self._projection_key}."

        # Batch the context item with the number of frames
        x = context[self._input_context_key]
        # Add the frame dimension
        x = x[:, None, ...]
        # Tile the number of frames
        tile_dims = [1] * len(x.shape)
        tile_dims[1] = self._num_frames
        x = torch.tile(x, dims=tile_dims)
        # Re-batch the temporal dimensions
        x = rearrange(x, "b f ... -> (b f) ...")
        context[self._input_context_key] = x
        return context
