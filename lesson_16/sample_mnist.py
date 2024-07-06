"""Lesson 10 - Image Super-Resolution via Iterative Refinement.

Helper script to sample from a pre-trained model. Uses the MNIST
test dataset (which was not seen during training) for conditioning.
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm

from utils import cycle, load_yaml, DotConfig
from diffusion_model import GaussianDiffusion_SR3
from score_network import MNistUnet

OUTPUT_NAME = "output/samples"


def run_lesson_10_sample(num_samples: int, config_path: str, diffusion_checkpoint: str):
    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    validation_dataset = MNIST(
        ".",
        train=False,
        transform=transforms.Compose(
            [
                # To make the math work out easier, resize the MNIST
                # images from (28,28) to (32, 32).
                transforms.Resize(size=(32, 32)),
                # Conversion to tensor scales the data from (0,255)
                # to (0,1).
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset, batch_size=num_samples, shuffle=True, num_workers=4
    )

    # Open the model configuration
    config = load_yaml(config_path)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    diffusion_model = GaussianDiffusion_SR3(score_network_type=MNistUnet, config=config)
    summary(
        diffusion_model._score_network,
        [
            (
                128,
                config.model.input_channels,
                config.model.input_spatial_size,
                config.model.input_spatial_size,
            ),
            (128,),
        ],
    )
    checkpoint = torch.load(diffusion_checkpoint)
    diffusion_model.load_state_dict(checkpoint["model_state_dict"])
    step = checkpoint["step"]

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )
    device = accelerator.device

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    validation_dataloader = accelerator.prepare(validation_dataloader)

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    validation_dataloader = cycle(validation_dataloader)

    # Move the model to the accelerator as well.
    diffusion_model = accelerator.prepare(diffusion_model)

    # Save and sample the final step.
    sample(
        diffusion_model=diffusion_model,
        step=step,
        config=config,
        num_samples=num_samples,
        validation_dataloader=validation_dataloader,
    )


def sample(
    diffusion_model: GaussianDiffusion_SR3,
    step,
    config: DotConfig,
    validation_dataloader,
    num_samples=64,
):
    device = next(diffusion_model.parameters()).device

    # Grab some random samples from the validation dataset to
    # test the super-resolution capabilities.
    low_res_originals = next(validation_dataloader)[0]
    low_res_context = transforms.functional.resize(
        low_res_originals, (8, 8), antialias=True
    )

    # Sample from the model to check the quality
    output = diffusion_model.sample(
        low_res_context=low_res_context,
        num_samples=num_samples,
    )

    if diffusion_model._is_class_conditional:
        samples, labels = output
    else:
        samples = output
        labels = None

    # Save the samples into an image grid
    utils.save_image(
        samples,
        str(f"{OUTPUT_NAME}/sample-{step}.png"),
        nrow=int(math.sqrt(num_samples)),
    )
    utils.save_image(
        low_res_context,
        str(f"{OUTPUT_NAME}/low_res_context-{step}.png"),
        nrow=int(math.sqrt(num_samples)),
    )
    utils.save_image(
        low_res_originals,
        str(f"{OUTPUT_NAME}/low_res_original-{step}.png"),
        nrow=int(math.sqrt(num_samples)),
    )
    utils.save_image(
        transforms.functional.resize(
            low_res_context,
            (config.model.input_spatial_size, config.model.input_spatial_size),
            antialias=True,
        ),
        str(f"{OUTPUT_NAME}/low_res_upsampled-{step}.png"),
        nrow=int(math.sqrt(num_samples)),
    )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--config_path", type=str, default="configs/mnist_sr3.yaml")
    parser.add_argument("--diffusion_checkpoint", type=str, required=True)

    args = parser.parse_args()

    run_lesson_10_sample(
        num_samples=args.num_samples,
        config_path=args.config_path,
        diffusion_checkpoint=args.diffusion_checkpoint,
    )


if __name__ == "__main__":
    main()
