"""Train a variational autoencoder for MNIST.

This is a simple VAE to help reduce the dimensionality of the MNIST
dataset from 1x32x32 to 1x8x8.
"""

from accelerate import Accelerator
import argparse
from einops import rearrange
import math
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm

from utils import (
    cycle,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)
from autoencoder import MNISTAutoEncoderKL

OUTPUT_NAME = "output/autoencoder"


def run_lesson_11_autoencoder(num_training_steps: int, batch_size: int):
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

    # Create the autoencoder we will train.
    vae = MNISTAutoEncoderKL()
    summary(
        vae,
        [(128, 1, 32, 32)],
    )

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(split_batches=False, mixed_precision="no")
    device = accelerator.device

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader = accelerator.prepare(dataloader)

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Basic optimizer for training.
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0001)

    # Define a basic loss function for autoencoder training. We combine
    # a reconstruction loss with a KL divergence regularization term.
    def loss_function(y, x, mu, std):
        # The reproduction loss is just the difference between
        # the original data and the autoencoder reconstructed data.
        reproduction_loss = torch.nn.functional.mse_loss(
            y.view(-1, 1024), x.view(-1, 1024), reduction="sum"
        )

        # Imposes a slight KL-penalty towards a standard normal on the learned latent.
        kld = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2)
        return reproduction_loss + kld, -reproduction_loss, -kld

    # Move the model and the optimizer to the accelerator as well.
    vae, optimizer = accelerator.prepare(vae, optimizer)

    # Step counter to keep track of training
    step = 0

    # We will sample the autoencoder every N steps, to monitor
    # training and see how it improves over time.
    save_and_sample_every_n = 1000

    # Clip the graidents for smoother training.
    max_grad_norm = 1.0

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, _ = next(dataloader)

            # Calculate the loss on the batch of training data.
            x, mu, std = vae(normalize_to_neg_one_to_one(images))
            x = unnormalize_to_zero_to_one(x)
            loss, _, _ = loss_function(images, x, mu, std)

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

            # Update the current step.
            step += 1

            # To help visualize training, periodically sample from the
            # autoencoder to see how well its doing.
            if step != 0 and step % save_and_sample_every_n == 0:
                # Save the reconstructed samples into an image grid
                utils.save_image(
                    x,
                    str(f"{OUTPUT_NAME}/sample-{step}.png"),
                    nrow=int(math.sqrt(batch_size)),
                )
                utils.save_image(
                    images,
                    str(f"{OUTPUT_NAME}/original-{step}.png"),
                    nrow=int(math.sqrt(batch_size)),
                )

                # Generate an unconditional sample as well from
                # a random latent.
                with torch.no_grad():
                    noise = torch.randn(batch_size, 1, 64).to(device)
                    x_hat = vae.decoder(noise)
                    x_hat = rearrange(x_hat, "b c (h w) -> b c h w", h=32, w=32)
                    x_hat = unnormalize_to_zero_to_one(x_hat)

                    utils.save_image(
                        x_hat,
                        str(f"{OUTPUT_NAME}/unconditional-{step}.png"),
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

            # Update the training progress bar in the console.
            progress_bar.update(1)


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    run_lesson_11_autoencoder(
        num_training_steps=args.num_training_steps, batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
