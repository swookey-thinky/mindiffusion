"""Implements a frozen CLIP embedder.

This package implements a frozen CLIP embedder, as used in Stable Diffusion, DaLL*E 2,
and others. This implementation is based on the original Stable Diffusion code at
https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/encoders/modules.py#L137
"""

import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    AutoProcessor,
    AutoTokenizer,
)
from typing import Dict, List


class FrozenCLIPEmbedder(torch.nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77
    ):
        super().__init__()
        self._text_model = CLIPTextModelWithProjection.from_pretrained(version)
        self._vision_model = CLIPVisionModelWithProjection.from_pretrained(version)
        self._processor = AutoProcessor.from_pretrained(version)
        self._tokenizer = AutoTokenizer.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self._text_model = self._text_model.eval()
        self._vision_model = self._vision_model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor, prompts: List[str]):
        assert images.shape[0] == len(prompts)

        # Normalize our data samples around the mean and stdev of MNIST, in order
        # make the source distribution have mean 0 and unit variance.
        MNIST_MEAN = 0.1307
        MNIST_STDEV = 0.3081

        # Images are already in [0,1] so do not rescale.
        image_inputs = self._processor(
            images=images,
            do_rescale=False,
            image_mean=MNIST_MEAN,
            image_std=MNIST_STDEV,
            return_tensors="pt",
        )
        text_inputs = self._tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        # Move all of the inputs to the same device as the inputs that were
        # passed in.
        image_inputs_on_device = {}
        for k, v in image_inputs.items():
            if k == "pixel_values":
                # CLIP image embeddings require 3 channel data
                image_inputs_on_device[k] = torch.tile(v.to(self.device), (1, 3, 1, 1))
            else:
                image_inputs_on_device[k] = v.to(self.device)
        text_inputs_on_device = {}
        for k, v in text_inputs.items():
            text_inputs_on_device[k] = v.to(self.device)

        text_outputs = self._text_model(**text_inputs_on_device)
        image_outputs = self._vision_model(**image_inputs_on_device)
        return (
            image_outputs["image_embeds"],
            text_outputs["text_embeds"],
            text_outputs["last_hidden_state"],
        )

    def encode(self, images: torch.Tensor, prompts: List[str]):
        """Gets image and text embeddings using CLIP.

        Args:
            images: Tensor batch of unnormalized image data
            prompts: List of string prompts.

        Returns:
            Tuple of:
                image_embeddings: Tensor batch of image embeddings
                text_embeddings: Tensor batch of prompt embeddings
                text_encodings: Tensor batch of prompt encodings
        """
        return self(images, prompts)

    def encode_text(self, prompts: List[str]):
        """Gets text embeddings using CLIP.

        Args:
            prompts: List of string prompts.

        Returns:
            Tuple of:
                text_embeddings: Tensor batch of prompt embeddings
                text_encodings: Tensor batch of prompt encodings
        """
        text_inputs = self._tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        text_inputs_on_device = {}
        for k, v in text_inputs.items():
            text_inputs_on_device[k] = v.to(self.device)

        text_outputs = self._text_model(**text_inputs_on_device)
        return (
            text_outputs["text_embeds"],
            text_outputs["last_hidden_state"],
        )

    def tokenize(self, **kwargs):
        return self._tokenizer(**kwargs)


class FrozenCLIPTextEmbedder(torch.nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77
    ):
        super().__init__()
        self._text_model = CLIPTextModelWithProjection.from_pretrained(version)
        self._tokenizer = AutoTokenizer.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self._text_model = self._text_model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, prompts: List[str]):
        text_inputs = self._tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        text_inputs_on_device = {}
        for k, v in text_inputs.items():
            text_inputs_on_device[k] = v.detach().to(self.device)

        text_outputs = self._text_model(**text_inputs_on_device)
        return (
            text_outputs["text_embeds"],
            text_outputs["last_hidden_state"],
        )

    def encode(self, prompts: List[str]):
        """Gets image and text embeddings using CLIP.

        Args:
            images: Tensor batch of unnormalized image data
            prompts: List of string prompts.

        Returns:
            Tuple of:
                image_embeddings: Tensor batch of image embeddings
                text_embeddings: Tensor batch of prompt embeddings
                text_encodings: Tensor batch of prompt encodings
        """
        return self(prompts)

    def embed(self, tokens: Dict):
        text_outputs = self._text_model(**tokens)
        return (
            text_outputs["text_embeds"],
            text_outputs["last_hidden_state"],
        )

    def tokenize(self, **kwargs):
        return self._tokenizer(**kwargs)


class FrozenCLIPTextTokenizer(torch.nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77
    ):
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, prompts: List[str], device) -> Dict:
        text_inputs = self._tokenizer(
            prompts,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        text_inputs_on_device = {}
        for k, v in text_inputs.items():
            text_inputs_on_device[k] = v.detach().to(device)
        return text_inputs_on_device

    def tokenize(self, prompts: List[str], device) -> Dict:
        return self(prompts, device)
