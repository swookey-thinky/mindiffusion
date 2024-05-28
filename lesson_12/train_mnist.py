"""Lesson 12 - Cascaded Diffusion Model."""

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
from diffusion_model import GaussianDiffusion_CDM

OUTPUT_NAME = "output"


def run_lesson_12(
    num_training_steps: int,
    batch_size: int,
    config_path: str,
):
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
    diffusion_model = GaussianDiffusion_CDM(config=config)
    summary(
        diffusion_model._base_stage._score_network,
        [
            (
                128,
                config.base_stage.model.input_channels,
                config.base_stage.model.input_spatial_size,
                config.base_stage.model.input_spatial_size,
            ),
            (128,),
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

    # Now create the optimizer. The optimizer choice and parameters come from
    # the paper:
    # "We tried Adam [31] and RMSProp early on in our experimentation process and chose the
    #  former. We left the hyperparameters to their standard values. We set the learning
    #  rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5
    #  for the 256 × 256 images, which seemed unstable to train with the larger learning rate."
    opts = diffusion_model.configure_optimizers(learning_rate=2e-4)

    # Move the model and the optimizer to the accelerator as well.
    diffusion_model = accelerator.prepare(diffusion_model)
    optimizers = []
    for opt in opts:
        optimizers.append(accelerator.prepare(opt))

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
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, _ = next(dataloader)

            stage_loss = 0
            for stage_idx in range(config.num_cascades + 1):
                # Calculate the loss on the batch of training data.
                loss_dict = diffusion_model.loss_on_batch(images, stage_idx=stage_idx)
                loss = loss_dict["loss"]

                # Calculate the gradients at each step in the network.
                accelerator.backward(loss)

                # On a multi-gpu machine or cluster, wait for all of the workers
                # to finish.
                accelerator.wait_for_everyone()

                # Clip the gradients.
                accelerator.clip_grad_norm_(
                    (
                        diffusion_model._base_stage.parameters()
                        if stage_idx == 0
                        else diffusion_model._super_resolution_stages[
                            stage_idx - 1
                        ].parameters()
                    ),
                    max_grad_norm,
                )

                # Perform the gradient descent step using the optimizer.
                optimizers[stage_idx].step()

                # Resent the gradients for the next step.
                optimizers[stage_idx].zero_grad()
                stage_loss += loss.item()

            # Show the current loss in the progress bar.
            stage_loss = stage_loss / (config.num_cascades + 1)
            progress_bar.set_description(
                f"loss: {stage_loss:.4f} avg_loss: {average_loss:.4f}"
            )
            average_loss_cumulative += stage_loss

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                sample(
                    diffusion_model=diffusion_model,
                    step=step,
                    config=config,
                )
                save(diffusion_model, step, loss)
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save and sample the final step.
    sample(
        diffusion_model=diffusion_model,
        step=step,
        config=config,
    )
    save(diffusion_model, step, loss)


def sample(
    diffusion_model: GaussianDiffusion_CDM,
    step,
    config: DotConfig,
    num_samples=64,
):
    device = next(diffusion_model.parameters()).device

    # Sample from the model to check the quality
    output, all_stage_output = diffusion_model.sample(
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

    # Save all of the stages output
    for stage_idx in range(len(all_stage_output)):
        utils.save_image(
            all_stage_output[stage_idx],
            str(f"{OUTPUT_NAME}/sample-{step}-stage-{stage_idx}.png"),
            nrow=int(math.sqrt(num_samples)),
        )


def save(diffusion_model, step, loss):
    # Save a corresponding model checkpoint.
    torch.save(
        {
            "step": step,
            "model_state_dict": diffusion_model.state_dict(),
            "loss": loss,
        },
        f"{OUTPUT_NAME}/cdm-{step}.pt",
    )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--config_path", type=str, default="configs/mnist_cdm.yaml")

    args = parser.parse_args()

    run_lesson_12(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
