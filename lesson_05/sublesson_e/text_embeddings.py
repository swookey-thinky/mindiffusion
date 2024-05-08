"""Utility packages for converting MNIST labels to text embeddings."""

import torch


def convert_labels_to_embeddings(
    labels: torch.Tensor, text_encoder: torch.nn.Module, return_prompts: bool = False
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
    text_embeddings = text_encoder.encode(prompts)

    if return_prompts:
        return text_embeddings, prompts
    else:
        return text_embeddings
