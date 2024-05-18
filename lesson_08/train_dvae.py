"""Train a variational autoencoder for MNIST.

This is a simple VAE to help reduce the dimensionality of the MNIST
dataset from 1x32x32 to 1x8x8.
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm

from utils import cycle, load_yaml, map_pixels
from dvae import DiscreteVAE

OUTPUT_NAME = "output/dvae"


def run_lesson_08_dvae(num_training_steps: int, batch_size: int, config_path: str):
    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

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
                transforms.Resize(
                    size=(config.data.image_size, config.data.image_size)
                ),
                # Conversion to tensor scales the data from (0,255)
                # to (0,1).
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )

    # Create the dataloader for the MNIST dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load the dVAE from the checkpoint
    vae = DiscreteVAE(
        num_groups=config.model.vae.num_groups,
        input_channels=config.model.vae.input_channels,
        vocab_size=config.model.vae.vocab_size,
        hidden_size=config.model.vae.hidden_size,
        num_blocks_per_group=config.model.vae.num_blocks_per_group,
    )

    summary(
        vae,
        [
            (
                128,
                config.model.vae.input_channels,
                config.data.image_size,
                config.data.image_size,
            )
        ],
    )

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

    # Basic optimizer for training.
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)

    # Move the model and the optimizer to the accelerator as well.
    vae, optimizer, scheduler = accelerator.prepare(vae, optimizer, scheduler)

    # Step counter to keep track of training
    step = 0

    # We will sample the autoencoder every N steps, to monitor
    # training and see how it improves over time.
    save_and_sample_every_n = 100

    # Clip the graidents for smoother training.
    max_grad_norm = 1.0
    temp = 1.0

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, _ = next(dataloader)

            # Calculate the loss on the batch of training data.
            loss, rx = vae(map_pixels(images), temperature=0.0625, kl_weight=0.0)

            # Calculate the gradients at each step in the network.
            accelerator.backward(loss)

            # On a multi-gpu machine or cluster, wait for all of the workers
            # to finish.
            accelerator.wait_for_everyone()

            # Clip the gradients.
            accelerator.clip_grad_norm_(vae.parameters(), max_grad_norm)

            # Perform the gradient descent step using the optimizer.
            optimizer.step()

            # Resent the gradients for the next step.
            optimizer.zero_grad()

            # Show the current loss in the progress bar.
            progress_bar.set_description(f"loss: {loss.item():.4f}")

            # To help visualize training, periodically sample from the
            # autoencoder to see how well its doing.
            if step % save_and_sample_every_n == 0:
                # Save the reconstructed samples into an image grid
                utils.save_image(
                    rx,
                    str(f"{OUTPUT_NAME}/sample-{step}.png"),
                    nrow=int(math.sqrt(batch_size)),
                )
                utils.save_image(
                    images,
                    str(f"{OUTPUT_NAME}/original-{step}.png"),
                    nrow=int(math.sqrt(batch_size)),
                )

                # Save a corresponding model checkpoint.
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": vae.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    f"{OUTPUT_NAME}/vae-{step}.pt",
                )

                temp = max(temp * math.exp(-1e-6 * step), 0.5)
                scheduler.step()

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save the reconstructed samples into an image grid
    utils.save_image(
        rx,
        str(f"{OUTPUT_NAME}/sample-{step}.png"),
        nrow=int(math.sqrt(batch_size)),
    )
    utils.save_image(
        images,
        str(f"{OUTPUT_NAME}/original-{step}.png"),
        nrow=int(math.sqrt(batch_size)),
    )

    # Save a corresponding model checkpoint.
    torch.save(
        {
            "step": step,
            "model_state_dict": vae.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"{OUTPUT_NAME}/dvae-{step}.pt",
    )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--config_path", type=str, default="configs/dall_e_mnist.yaml")
    args = parser.parse_args()

    run_lesson_08_dvae(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
