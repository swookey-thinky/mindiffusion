from functools import partial
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from typing import Callable, List, Tuple


def load_mnist(
    training_height: int,
    training_width: int,
    invert: bool = False,
    split: str = "train",
) -> Tuple[Dataset, Callable[[torch.Tensor], List[str]]]:
    assert split in ["train", "validation"]

    if invert:
        xforms = [
            # To make the math work out easier, resize the MNIST
            # images from (28,28) to (32, 32).
            transforms.Resize(size=(training_height, training_width)),
            # Conversion to tensor scales the data from (0,255)
            # to (0,1).
            transforms.ToTensor(),
            # Invert the dataset for LoRA training
            transforms.Lambda(_invert),
        ]
    else:
        xforms = [
            # To make the math work out easier, resize the MNIST
            # images from (28,28) to (32, 32).
            transforms.Resize(size=(training_height, training_width)),
            # Conversion to tensor scales the data from (0,255)
            # to (0,1).
            transforms.ToTensor(),
        ]
    if split == "train":
        # Load the MNIST dataset. This is a supervised dataset so
        # it contains both images and class labels. We will ignore the class
        # labels for now.
        return (
            MNIST(
                ".",
                train=True,
                transform=transforms.Compose(xforms),
                download=True,
            ),
            convert_labels_to_prompts,
        )
    else:
        # Load the MNIST dataset. This is a supervised dataset so
        # it contains both images and class labels. We will ignore the class
        # labels for now.
        return (
            MNIST(
                ".",
                train=False,
                transform=transforms.Compose(xforms),
                download=True,
            ),
            convert_labels_to_prompts,
        )


def convert_labels_to_prompts(labels: torch.Tensor) -> List[str]:
    """Converts MNIST class labels to text prompts.

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
    return prompts


def _invert(x: torch.Tensor) -> torch.Tensor:
    return transforms.functional.invert(x)
