import torch
from torch.utils.data import Dataset
from typing import Callable, List, Tuple

from xdiffusion.utils import DotConfig


def load_dataset(
    dataset_name: str,
    config: DotConfig,
    split: str = "train",
) -> Tuple[Dataset, Callable[[torch.Tensor], List[str]]]:
    assert dataset_name in [
        "image/mnist",
        "image/mnist_inverted",
        "image/moving_mnist",
        "image/moving_mnist_inverted",
        "image/cifar10",
        "image/mnist_embedded_gemma_2",
        "video/moving_mnist",
    ]

    if dataset_name == "image/mnist":
        from xdiffusion.datasets.mnist import load_mnist

        return load_mnist(
            training_height=config.image_size,
            training_width=config.image_size,
            split=split,
        )
    elif dataset_name == "image/mnist_inverted":
        from xdiffusion.datasets.mnist import load_mnist

        return load_mnist(
            training_height=config.image_size,
            training_width=config.image_size,
            split=split,
            invert=True,
        )
    elif dataset_name == "image/moving_mnist":
        from xdiffusion.datasets.moving_mnist import load_moving_mnist_image

        return load_moving_mnist_image(
            training_height=config.image_size,
            training_width=config.image_size,
            split=split,
        )
    elif dataset_name == "image/moving_mnist_inverted":
        from xdiffusion.datasets.moving_mnist import load_moving_mnist_image

        return load_moving_mnist_image(
            training_height=config.image_size,
            training_width=config.image_size,
            split=split,
            invert=True,
        )
    elif dataset_name == "image/cifar10":
        from xdiffusion.datasets.cifar10 import load_cifar10

        return load_cifar10(
            training_height=config.image_size,
            training_width=config.image_size,
            split=split,
        )
    elif dataset_name == "image/mnist_embedded_gemma_2":
        from xdiffusion.datasets.mnist_embedded_gemma_2 import load_mnist

        return load_mnist(
            training_height=config.image_size,
            training_width=config.image_size,
            split=split,
        )

    raise NotImplementedError(f"Dataset '{dataset_name}' not implemented yet.")
