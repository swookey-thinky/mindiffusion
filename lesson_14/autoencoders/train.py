"""Train a variational autoencoder for MNIST.

This is a simple VAE to help reduce the dimensionality of the MNIST
dataset from 1x32x32 to 1x8x8.
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
from einops import rearrange
import math
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision import utils as torch_utils
from torchvision.datasets import MNIST
from tqdm import tqdm

from utils import (
    cycle,
    load_yaml,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)
from autoencoders.kl import AutoencoderKL

OUTPUT_NAME = "output/autoencoder_kl"


def train_autoencoder(num_training_steps: int, batch_size: int, config_path: str):
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

    # Create the autoencoder we will train.
    vae = AutoencoderKL(config)
    summary(
        vae,
        [(128, 1, 32, 32)],
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

    # Configure the optimizers for training.
    opts, _ = vae.configure_optimizers(learning_rate=4.5e-6)

    # Move the model and the optimizer to the accelerator as well.
    vae = accelerator.prepare(vae)
    optimizers = []
    for opt in opts:
        optimizers.append(accelerator.prepare(opt))

    # Step counter to keep track of training
    step = 0

    # We will sample the autoencoder every N steps, to monitor
    # training and see how it improves over time.
    save_and_sample_every_n = 100

    # Clip the graidents for smoother training.
    max_grad_norm = 1.0
    average_losses = [0.0 for _ in optimizers]
    average_losses_cumulative = [0.0 for _ in optimizers]
    average_posterior_mean = 0.0
    average_posterior_mean_cumulative = 0.0
    average_posterior_std = 0.0
    average_posterior_std_cumulative = 0.0

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, labels = next(dataloader)

            batch = {"image": normalize_to_neg_one_to_one(images), "label": labels}

            # Calculate the loss on the batch of training data.
            current_loss = []
            for optimizer_idx, optimizer in enumerate(optimizers):
                loss, reconstructions, posterior = vae.training_step(
                    batch, -1, optimizer_idx, step
                )

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

                average_losses_cumulative[optimizer_idx] += loss.item()
                current_loss.append(loss.item())
            average_posterior_mean_cumulative += posterior.mean.detach().mean()
            average_posterior_std_cumulative += posterior.std.detach().mean()

            x = unnormalize_to_zero_to_one(reconstructions)

            # Show the current loss in the progress bar.
            progress_bar.set_description(
                f"loss: {[f'{val:.4f}' for idx, val in enumerate(current_loss)]} avg_loss: {[f'{val:.4f}' for idx, val in enumerate(average_losses)]} KL: {posterior.kl().detach().mean():.4f} posterior_mean: {average_posterior_mean:.4f} posterior_std: {average_posterior_std:.4f}"
            )

            # To help visualize training, periodically sample from the
            # autoencoder to see how well its doing.
            if step % save_and_sample_every_n == 0:
                average_losses = [
                    loss / save_and_sample_every_n for loss in average_losses_cumulative
                ]
                average_losses_cumulative = [0.0 for _ in optimizers]
                average_posterior_mean = (
                    average_posterior_mean_cumulative / save_and_sample_every_n
                )
                average_posterior_std = (
                    average_posterior_std_cumulative / save_and_sample_every_n
                )
                average_posterior_mean_cumulative = 0.0
                average_posterior_std_cumulative = 0.0

                # Save the reconstructed samples into an image grid
                torch_utils.save_image(
                    x,
                    str(f"{OUTPUT_NAME}/reconstructed-{step}.png"),
                    nrow=int(math.sqrt(batch_size)),
                )
                torch_utils.save_image(
                    images,
                    str(f"{OUTPUT_NAME}/original-{step}.png"),
                    nrow=int(math.sqrt(batch_size)),
                )

                # Generate an unconditional sample as well from
                # a random latent.
                latent_resolution = config.encoder_decoder_config.resolution // 2 ** (
                    len(config.encoder_decoder_config.ch_mult) - 1
                )
                with torch.inference_mode():
                    noise = torch.randn(
                        batch_size,
                        config.embed_dim,
                        latent_resolution,
                        latent_resolution,
                    ).to(device)
                    x_hat = vae.decode(noise)
                    x_hat = unnormalize_to_zero_to_one(x_hat)

                    torch_utils.save_image(
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

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save the last samples
    torch_utils.save_image(
        x,
        str(f"{OUTPUT_NAME}/reconstructed-{step}.png"),
        nrow=int(math.sqrt(batch_size)),
    )
    torch_utils.save_image(
        images,
        str(f"{OUTPUT_NAME}/original-{step}.png"),
        nrow=int(math.sqrt(batch_size)),
    )

    # Generate an unconditional sample as well from
    # a random latent.
    with torch.inference_mode():
        noise = torch.randn(batch_size, config.embed_dim, 8, 8).to(device)
        x_hat = vae.decode(noise)
        x_hat = unnormalize_to_zero_to_one(x_hat)

        torch_utils.save_image(
            x_hat,
            str(f"{OUTPUT_NAME}/unconditional-{step}.png"),
            nrow=int(math.sqrt(batch_size)),
        )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--config_path", type=str, default="autoencoders/configs/mnist_4x8x8.yaml"
    )
    args = parser.parse_args()

    train_autoencoder(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
