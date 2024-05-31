"""Lesson 15 - GLIDE


```
python sample_with_guidance.py --diffusion_checkpoint <path to diffusion checkpoint>
```
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
import torch
from torchinfo import summary
from torchvision import utils
from typing import List

from diffusion_model import GaussianDiffusion_GLIDE
from utils import load_yaml, DotConfig

OUTPUT_NAME = "output/samples_with_guidance"


def run_lesson_15_sample(
    diffusion_checkpoint: str,
    num_samples: int,
    config_path: str,
):
    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    diffusion_model = GaussianDiffusion_GLIDE(config=config)
    checkpoint = torch.load(diffusion_checkpoint)
    diffusion_model.load_state_dict(checkpoint["model_state_dict"])
    summary(
        diffusion_model._score_network,
        [(128, 1, 32, 32), (128,), (128, config.model.text_context_size)],
    )

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Move the model and the classifier to the accelerator as well.
    diffusion_model = accelerator.prepare(diffusion_model)
    sample(
        diffusion_model=diffusion_model,
        config=config,
        num_samples=num_samples,
    )


def sample(
    diffusion_model: GaussianDiffusion_GLIDE,
    config: DotConfig,
    num_samples=64,
):
    device = next(diffusion_model.parameters()).device

    # Sample from the model to check the quality.
    classes = torch.randint(
        0, config.data.num_classes, size=(num_samples,), device=device
    )
    prompts = convert_labels_to_prompts(classes)
    for cfg in [0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 20.0]:
        output = diffusion_model.sample(
            prompts=prompts,
            num_samples=num_samples,
            classifier_free_guidance=cfg,
        )

        if config.model.is_class_conditional:
            samples, labels = output
        else:
            samples = output

        # Save the samples into an image grid
        utils.save_image(
            samples,
            str(f"{OUTPUT_NAME}/sample-cfg-{cfg}.png"),
            nrow=int(math.sqrt(num_samples)),
        )

    # Save the prompts we used.
    with open(f"{OUTPUT_NAME}/sample.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{prompts[i]} ")


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


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_checkpoint", type=str)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--config", type=str, default="configs/mnist_glide.yaml")
    args = parser.parse_args()

    run_lesson_15_sample(
        diffusion_checkpoint=args.diffusion_checkpoint,
        num_samples=args.num_samples,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
