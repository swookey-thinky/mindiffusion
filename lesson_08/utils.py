"""Utilities used in the lesson."""

import torch
from torch.utils.data import DataLoader
from typing import Any, TypeVar
import yaml

from tokenizer import SimpleTokenizer

logit_laplace_eps: float = 0.1


class DotConfig:
    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k) -> Any:
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v


def load_yaml(yaml_path: str) -> DotConfig:
    """Loads a YAML configuration file."""
    with open(yaml_path, "r") as fp:
        return DotConfig(yaml.load(fp, yaml.CLoader))


def map_pixels(x: torch.Tensor) -> torch.Tensor:
    """Map pixels from (0,1) to (eps, 1-eps)."""
    return (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps


def unmap_pixels(x: torch.Tensor) -> torch.Tensor:
    """Map pixels back into (0,1).

    Inverse of map_pixels.
    """
    return torch.clamp((x - logit_laplace_eps) / (1 - 2 * logit_laplace_eps), 0, 1)


def exists(val):
    """True if the item is not None."""
    return val is not None


def is_empty(t):
    """True if the item has no elements."""
    return t.nelement() == 0


def cycle(dataloader: DataLoader):
    """Cycles through the dataloader class forever.

    Useful for when you want to cycle through a DataLoader for
    a finite number of timesteps.
    """
    while True:
        for data in dataloader:
            yield data


T = TypeVar("T", bound=torch.nn.Module)


def freeze(model: T) -> T:
    """Freeze the parameters of a model."""
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def split_tensor_along_last_dim(tensor, num_partitions, contiguous_split_chunks=False):
    """
    Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list


def convert_labels_to_tokens(
    labels: torch.Tensor,
    tokenizer: SimpleTokenizer,
    text_token_length: int,
    return_prompts: bool = False,
) -> torch.Tensor:
    """Converts MNIST class labels to embeddings.

    Supports both the strings "0" and "zero" to describe the
    class labels.
    """
    # The conditioning we pass to the model will be a vectorized-form of
    # MNIST classes. Since we have a fixed number of classes, we can create
    # a hard-coded "embedding" of the MNIST class label.
    text_labels = [
        ("zero", "0"),
        ("one", "1"),
        ("two", "2"),
        ("three", "3"),
        ("four", "4"),
        ("five", "5"),
        ("six", "6"),
        ("seven", "7"),
        ("eight", "8"),
        ("nine", "9"),
    ]

    # First convert the labels into a list of string prompts
    prompts = [
        text_labels[labels[i]][torch.randint(0, len(text_labels[labels[i]]), size=())]
        for i in range(labels.shape[0])
    ]

    # Convert the prompts into context embeddings. Use the text encoder
    # we created earlier to convert the text labels in vector tensors.
    text_tokens = tokenizer.tokenize(prompts, context_length=text_token_length)

    if return_prompts:
        return text_tokens, prompts
    else:
        return text_tokens


def top_k(logits, thres=0.5):
    """Keeps the top K logits, pushing the rest to -inf.

    Args:
        logits: Logits to analyze.
        thres: Threshold of logits to keep, from (0,1)
    """
    num_logits = logits.shape[-1]

    # By default, we keep half the logits.
    k = max(int((1 - thres) * num_logits), 1)

    # Grab the top-k
    val, ind = torch.topk(logits, k)

    # Push the non-top logits to -inf
    probs = torch.full_like(logits, float("-inf"))

    # Keep the top-k logits.
    probs.scatter_(1, ind, val)
    return probs
