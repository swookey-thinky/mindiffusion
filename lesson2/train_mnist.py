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

from diffusion_model import GaussianDiffusion_DPM
from utils import (
    cycle,
    mnist_unnormalize,
)

OUTPUT_NAME = "output"


def run_lesson_2(num_training_steps: int, batch_size: int):
    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    dataset = MNIST(
        ".",
        train=True,
        transform=transforms.Compose(
            [
                # Conversion to tensor scales the data from (0,255)
                # to (0,1).
                transforms.ToTensor(),
            ]
        ),
        download=True,
    )

    # Create the dataloader for the MNIST dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Create the diffusion model to train
    num_timesteps = 1000
    diffusion_model = GaussianDiffusion_DPM(num_timesteps=num_timesteps)

    # Summarize the model to get information about the number of parameters
    # and layers inside.
    summary(
        diffusion_model._mu_sigma_predictor,
        [(128, 1, 28, 28), (128,)],
        dtypes=[torch.float32, torch.int64],
    )

    accelerator = Accelerator(
        DataLoaderConfiguration(split_batches=False), mixed_precision="no"
    )
    device = accelerator.device

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader = accelerator.prepare(dataloader)

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Now create the optimizer.
    optimizer = Adam(diffusion_model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, verbose=True, min_lr=1e-5
    )

    # Move the model and the optimizer to the accelerator as well.
    diffusion_model, optimizer, scheduler = accelerator.prepare(
        diffusion_model, optimizer, scheduler
    )

    # We will sample the diffusion model every N steps, to monitor
    # training and see how it improves over time.
    save_and_sample_every_n = 100

    step = 0
    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        average_loss = 0.0
        average_loss_snapshot = 0.0
        while step < num_training_steps:
            # The dataset has images and classes, but we don't need the class
            data = next(dataloader)[0].to(device)

            # Calculate the loss on the batch
            loss = diffusion_model.calculate_loss_on_batch(data)

            # Calculate the gradients at each step in the network.
            accelerator.backward(loss)
            # On a multi-gpu machine or cluster, wait for all of the workers
            # to finish.
            accelerator.wait_for_everyone()

            # Perform the gradient descent step using the optimizer.
            optimizer.step()

            # Resent the gradients for the next step.
            optimizer.zero_grad()

            # Show the current loss in the progress bar.
            progress_bar.set_description(
                f"loss: {loss.item():.4f} average_loss: {average_loss_snapshot: .4f}"
            )
            average_loss += loss.item()

            if step != 0 and step % save_and_sample_every_n == 0:
                average_loss = average_loss / float(save_and_sample_every_n)
                average_loss_snapshot = average_loss
                scheduler.step(average_loss)

                num_samples = 64
                x_t = diffusion_model.reverse_diffusion_full_trajectory(
                    num_timesteps, num_samples=num_samples, device=device
                )

                utils.save_image(
                    mnist_unnormalize(x_t),
                    str(f"{OUTPUT_NAME}/sample_{step}_loss_{average_loss:.4f}.png"),
                    nrow=int(math.sqrt(num_samples)),
                )

                # Save a corresponding model checkpoint.
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": diffusion_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    f"{OUTPUT_NAME}/gaussian_diffusion_dpm-{step}.pt",
                )

                average_loss = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    run_lesson_2(num_training_steps=args.num_training_steps, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
