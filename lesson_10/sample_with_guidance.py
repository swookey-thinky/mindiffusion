"""Lesson 10 - Guided Diffusion - Classifier Guided Sampling

Sampling script for sampling from a Diffusion Model from
"Diffusion Models Beat GANs on Image Synthesis" (https://arxiv.org/abs/2105.05233)
using classifier guidance.

To run this script, install all of the necessary requirements
and run:

```
python sample_with_guidance.py --diffusion_checkpoint <path to diffusion checkpoint> --classifier_checkpoint <path to classifier checkpoint>
```
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
import torch
from torchinfo import summary
from torchvision import utils

from classifier import GuidedDiffusionClassifier
from diffusion_model import GaussianDiffusion_GuidedDiffusion
from utils import load_yaml
from score_network import MNistUnet

OUTPUT_NAME = "output/samples_with_guidance"


def run_lesson_10_sample(
    diffusion_checkpoint: str,
    classifier_checkpoint: str,
    num_samples: int,
    classifier_scale: float,
    config_path: str,
):
    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    diffusion_model = GaussianDiffusion_GuidedDiffusion(
        score_network_type=MNistUnet, config=config
    )
    checkpoint = torch.load(diffusion_checkpoint)
    diffusion_model.load_state_dict(checkpoint["model_state_dict"])
    summary(
        diffusion_model._score_network,
        [(128, 1, 32, 32), (128,), (128,)],
        dtypes=[torch.float32, torch.float32, torch.int],
    )

    # Create the classifier we are going to train, with a UNet
    # specifically for the MNIST dataset.
    classifier = GuidedDiffusionClassifier(config=config)
    checkpoint = torch.load(classifier_checkpoint)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    classifier.eval()
    summary(classifier, [(128, 1, 32, 32), (128,)])

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Move the model and the classifier to the accelerator as well.
    diffusion_model, classifier = accelerator.prepare(diffusion_model, classifier)
    sample(
        diffusion_model=diffusion_model,
        classifier=classifier,
        classifier_scale=classifier_scale,
        num_samples=num_samples,
    )


def sample(
    diffusion_model: GaussianDiffusion_GuidedDiffusion,
    classifier: GuidedDiffusionClassifier,
    classifier_scale,
    image_size: int = 32,
    num_channels: int = 1,
    num_samples: int = 64,
):
    # Grab the function we use to calculate the clssifier guidance.
    guidance_fn = diffusion_model.get_classifier_guidance(classifier, classifier_scale)

    # Sample from the model to check the quality
    output = diffusion_model.sample(
        image_size=image_size,
        num_channels=num_channels,
        batch_size=num_samples,
        guidance_fn=guidance_fn,
    )

    if diffusion_model._is_class_conditional:
        guided_samples, labels = output
    else:
        guided_samples = output
        labels = None

    # Also sample with no guidance, using the same labels
    output = diffusion_model.sample(
        image_size=image_size,
        num_channels=num_channels,
        batch_size=num_samples,
        classes=labels,
    )

    if diffusion_model._is_class_conditional:
        unguided_samples, labels = output
    else:
        unguided_samples = output
        labels = None

    # Save the samples into an image grid
    utils.save_image(
        guided_samples,
        str(f"{OUTPUT_NAME}/guided_samples.png"),
        nrow=int(math.sqrt(num_samples)),
    )
    utils.save_image(
        unguided_samples,
        str(f"{OUTPUT_NAME}/unguided_samples.png"),
        nrow=int(math.sqrt(num_samples)),
    )

    # Save the labels if we have them
    if labels is not None:
        with open(f"{OUTPUT_NAME}/labels.txt", "w") as fp:
            for i in range(num_samples):
                if i != 0 and (i % math.sqrt(num_samples)) == 0:
                    fp.write("\n")
                fp.write(f"{labels[i]} ")


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_checkpoint", type=str)
    parser.add_argument("--diffusion_checkpoint", type=str)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--classifier_scale", type=float, default=10.0)
    parser.add_argument("--config", type=str, default="configs/mnist_v_param.yaml")
    args = parser.parse_args()

    run_lesson_10_sample(
        diffusion_checkpoint=args.diffusion_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        num_samples=args.num_samples,
        classifier_scale=args.classifier_scale,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
