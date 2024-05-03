"""Lesson 4 - Noise Conditioned Score Networks (v2)

Training script for training a Gaussian Diffusion Model from
"Improved Techniques for Training Score-Based Generative Models"
(https://arxiv.org/abs/2006.09011).

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

from diffusion_model import GaussianDiffusion_NCSNv2
from score_network import NCSNv2
from utils import cycle, EMAHelper


OUTPUT_NAME = "output"


def run_lesson_4(num_training_steps: int, batch_size: int, resume_ckpt: str):
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
    diffusion_model = GaussianDiffusion_NCSNv2()

    loaded_checkpoint = None
    if resume_ckpt:
        loaded_checkpoint = torch.load(resume_ckpt)

    # Load the model weights if they exist
    if loaded_checkpoint:
        diffusion_model.load_state_dict(loaded_checkpoint["model_state_dict"])

    # Summarize the model to get information about the number of parameters
    # and layers inside.
    summary(
        diffusion_model._scorenet,
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
    optimizer = Adam(diffusion_model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    if loaded_checkpoint:
        optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])

    # Move the model and the optimizer to the accelerator as well.
    diffusion_model, optimizer = accelerator.prepare(diffusion_model, optimizer)

    # Create the EMA helper for sampling
    ema_helper = EMAHelper()
    ema_helper.register(diffusion_model._scorenet)

    # We will sample the diffusion model every N steps, to monitor
    # training and see how it improves over time.
    save_and_sample_every_n = 100

    step = 0
    if loaded_checkpoint:
        step = loaded_checkpoint["step"]

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

            # update the averages
            ema_helper.update(diffusion_model._scorenet)

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

                # Sample from the model to show progress. Use the EMA model
                # for sampling.
                ema_model = NCSNv2(diffusion_model.sigmas).to(device)
                ema_model.load_state_dict(diffusion_model._scorenet.state_dict())
                ema_helper.copy_params(ema_model)
                ema_model.requires_grad_(False)
                ema_model.eval()

                num_samples = 64
                x_t = diffusion_model.reverse_diffusion_full_trajectory(
                    num_samples=num_samples, device=device, score_model=ema_model
                )

                utils.save_image(
                    x_t,
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
                    f"{OUTPUT_NAME}/gaussian_diffusion_ncsn_v2-{step}.pt",
                )

                del ema_model
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
    parser.add_argument("--num_training_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--resume_ckpt", type=str, default="")
    args = parser.parse_args()

    run_lesson_4(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        resume_ckpt=args.resume_ckpt,
    )


if __name__ == "__main__":
    main()
