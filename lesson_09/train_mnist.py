"""Lesson 9 - Improved Denoising Diffusion Probabilistic Models

Training script for training a Gaussian Diffusion Model from
"Improved Denoising Diffusion Probabilistic Models" (https://arxiv.org/abs/2102.09672)

To run this script, install all of the necessary requirements
and run:

```
python train_mnist.py
```
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

from diffusion_model import GaussianDiffusion_ImprovedDDPM
from utils import cycle, load_yaml
from score_network import MNistUnet

OUTPUT_NAME = "output"


def run_lesson_09(num_training_steps: int, batch_size: int, config_path: str):
    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Load the MNIST dataset. This is a supervised dataset so
    # it contains both images and class labels. We will ignore the class
    # labels for now.
    dataset = MNIST(
        ".",
        train=True,
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

    # Create the dataloader for the MNIST dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Open the model configuration
    config = load_yaml(config_path)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    diffusion_model = GaussianDiffusion_ImprovedDDPM(
        score_network_type=MNistUnet, config=config
    )
    summary(diffusion_model._score_network, [(128, 1, 32, 32), (128,)])

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )
    device = accelerator.device

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader = accelerator.prepare(dataloader)

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Now create the optimizer. The optimizer choice and parameters come from
    # the DDPM paper:
    # "We tried Adam [31] and RMSProp early on in our experimentation process and chose the
    #  former. We left the hyperparameters to their standard values. We set the learning
    #  rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5
    #  for the 256 × 256 images, which seemed unstable to train with the larger learning rate."
    optimizer = Adam(diffusion_model.parameters(), lr=2e-4, betas=(0.9, 0.99))

    # Move the model and the optimizer to the accelerator as well.
    diffusion_model, optimizer = accelerator.prepare(diffusion_model, optimizer)

    # Step counter to keep track of training
    step = 0
    # We will sample the diffusion model every N steps, to monitor
    # training and see how it improves over time.
    save_and_sample_every_n = 100
    # Not mentioned in the DDPM paper, but the original implementation
    # used gradient clipping during training.
    max_grad_norm = 1.0
    average_loss = 0.0
    average_loss_cumulative = 0.0

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes, but we don't need the class
            data = next(dataloader)[0].to(device)

            # Calculate the loss on the batch of training data.
            loss_dict = diffusion_model.loss_on_batch(data)
            loss = loss_dict["loss"]

            # Calculate the gradients at each step in the network.
            accelerator.backward(loss)
            # On a multi-gpu machine or cluster, wait for all of the workers
            # to finish.
            accelerator.wait_for_everyone()

            # Clip the gradients.
            accelerator.clip_grad_norm_(diffusion_model.parameters(), max_grad_norm)

            # Perform the gradient descent step using the optimizer.
            optimizer.step()

            # Resent the gradients for the next step.
            optimizer.zero_grad()

            # Show the current loss in the progress bar.
            progress_bar.set_description(
                f"loss: {loss.item():.4f} avg_loss: {average_loss:.4f}"
            )
            average_loss_cumulative += loss.item()

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                sample(diffusion_model, step)
                save(diffusion_model, step, loss, optimizer)
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save the final results
    sample(diffusion_model, step)
    save(diffusion_model, step, loss, optimizer)


def sample(diffusion_model, step, image_size: int = 32, num_channels: int = 1):
    num_samples = 64
    with torch.inference_mode():
        # Sample from the model to check the quality
        samples = diffusion_model.sample(
            image_size=image_size,
            num_channels=num_channels,
            batch_size=num_samples,
        )
    # Ssve the samples into an image grid
    utils.save_image(
        samples,
        str(f"{OUTPUT_NAME}/sample-{step}.png"),
        nrow=int(math.sqrt(num_samples)),
    )


def save(diffusion_model, step, loss, optimizer):
    # Save a corresponding model checkpoint.
    torch.save(
        {
            "step": step,
            "model_state_dict": diffusion_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"{OUTPUT_NAME}/gaussian_diffusion_improved_ddpm-{step}.pt",
    )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--config", type=str, default="configs/mnist_v_param.yaml")
    args = parser.parse_args()

    run_lesson_09(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
