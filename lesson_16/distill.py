"""Distills a trained model

From the paper "Progressive Distillation for Fast Sampling of Diffusion Models"
(https://arxiv.org/abs/2202.00512)
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm
from typing import List

from image_diffusion.utils import (
    cycle,
    freeze,
    load_yaml,
    unfreeze,
    DotConfig,
)
from image_diffusion.ddpm import GaussianDiffusion_DDPM
from image_diffusion.diffusion import DiffusionModel, PredictionType
from image_diffusion.cascade import GaussianDiffusionCascade
from image_diffusion.samplers.ddim import DDIMSampler

OUTPUT_NAME = "output/mnist/distilled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def distill(
    batch_size: int,
    teacher_model_checkpoint: str,
    config_path: str,
    distillation_iterations: int,
    num_training_steps_per_iteration: int,
    save_and_sample_every_n: int,
    initial_sampling_steps: int,
):
    # Open the model configuration
    config = load_yaml(config_path)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    if "diffusion_cascade" in config:
        teacher_diffusion_model = GaussianDiffusionCascade(config)
        student_diffusion_model = GaussianDiffusionCascade(config)
    else:
        teacher_diffusion_model = GaussianDiffusion_DDPM(config=config)
        student_diffusion_model = GaussianDiffusion_DDPM(config=config)

    # Load the model weights if we have them
    if teacher_model_checkpoint:
        teacher_diffusion_model.load_checkpoint(teacher_model_checkpoint)

    # Build context to display the model summary.
    student_diffusion_model.print_model_summary()

    # Distillation only works for V prediction continuous time models.
    if (
        not student_diffusion_model._noise_scheduler.continuous()
        or student_diffusion_model._prediction_type != PredictionType.V
    ):
        raise NotImplemented(
            "Distillation only supports v-prediction models with continuous time formulation."
        )

    N = initial_sampling_steps
    for iteration_idx in range(distillation_iterations):
        # Initialize the student from the teacher
        student_diffusion_model.load_state_dict(teacher_diffusion_model.state_dict())

        print(f"Distilling model into {N} sampling steps...")
        single_distillation_iteration(
            batch_size=batch_size,
            teacher_diffusion_model=teacher_diffusion_model,
            student_diffusion_model=student_diffusion_model,
            config=config,
            num_training_steps_per_iteration=num_training_steps_per_iteration,
            save_and_sample_every_n=save_and_sample_every_n,
            N=N,
            iteration_idx=iteration_idx,
            config_path=config_path,
        )

        # Student becomes the teacher
        teacher_diffusion_model.load_state_dict(student_diffusion_model.state_dict())

        # Halve the number of sampling steps
        N = N // 2


def single_distillation_iteration(
    batch_size: int,
    teacher_diffusion_model: DiffusionModel,
    student_diffusion_model: DiffusionModel,
    num_training_steps_per_iteration: int,
    save_and_sample_every_n: int,
    N: int,
    config: DotConfig,
    config_path: str,
    iteration_idx: int,
):
    global OUTPUT_NAME
    LOCAL_OUTPUT_NAME = (
        f"{OUTPUT_NAME}/{str(Path(config_path).stem)}/{iteration_idx}_{N}"
    )

    # Ensure the output directories exist
    os.makedirs(LOCAL_OUTPUT_NAME, exist_ok=True)

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

    validation_dataset = MNIST(
        ".",
        train=False,
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

    num_samples = 64
    validation_dataloader = DataLoader(
        dataset, batch_size=num_samples, shuffle=True, num_workers=4
    )

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader, validation_dataloader = accelerator.prepare(
        dataloader, validation_dataloader
    )

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Now create the optimizer. The optimizer choice and parameters come from
    # the paper:
    # "We tried Adam [31] and RMSProp early on in our experimentation process and chose the
    #  former. We left the hyperparameters to their standard values. We set the learning
    #  rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5
    #  for the 256 × 256 images, which seemed unstable to train with the larger learning rate."
    optimizers = student_diffusion_model.configure_optimizers(learning_rate=2e-4)

    # Move the model and the optimizer to the accelerator as well.
    student_diffusion_model, teacher_diffusion_model = accelerator.prepare(
        student_diffusion_model, teacher_diffusion_model
    )
    for optimizer_idx in range(len(optimizers)):
        optimizers[optimizer_idx] = accelerator.prepare(optimizers[optimizer_idx])

    # Configure the learning rate schedule
    learning_rate_schedules = student_diffusion_model.configure_learning_rate_schedule(
        optimizers
    )
    for schedule_idx in range(len(learning_rate_schedules)):
        learning_rate_schedules[schedule_idx] = accelerator.prepare(
            learning_rate_schedules[schedule_idx]
        )

    # Step counter to keep track of training
    step = 0

    # Not mentioned in the DDPM paper, but the original implementation
    # used gradient clipping during training.
    max_grad_norm = 1.0
    average_loss = 0.0
    average_loss_cumulative = 0.0

    # Freeze the teacher model
    teacher_diffusion_model = freeze(teacher_diffusion_model)
    student_diffusion_model = unfreeze(student_diffusion_model)
    teacher_diffusion_model.eval()
    student_diffusion_model.train()

    with tqdm(initial=step, total=num_training_steps_per_iteration) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps_per_iteration:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, classes = next(dataloader)
            context = {"classes": classes}

            # Convert the labels to prompts
            prompts = convert_labels_to_prompts(classes)
            context["text_prompts"] = prompts

            # Train each cascade in the model using the given data.
            stage_loss = 0
            for (
                student_model_for_layer,
                teacher_model_for_layer,
                optimizer_for_layer,
                schedule_for_layer,
            ) in zip(
                student_diffusion_model.models(),
                teacher_diffusion_model.models(),
                optimizers,
                learning_rate_schedules,
            ):
                # Is this a super resolution model? If it is, then generate
                # the low resolution imagery as conditioning.
                config_for_layer = student_model_for_layer.config()
                context_for_layer = context.copy()
                images_for_layer = images

                if "super_resolution" in config_for_layer:
                    # First create the low resolution context.
                    low_resolution_spatial_size = (
                        config_for_layer.super_resolution.low_resolution_spatial_size
                    )
                    low_resolution_images = transforms.functional.resize(
                        images,
                        size=(
                            low_resolution_spatial_size,
                            low_resolution_spatial_size,
                        ),
                        antialias=True,
                    )
                    context_for_layer[
                        config_for_layer.super_resolution.conditioning_key
                    ] = low_resolution_images

                # If the images are not the right shape for the model input, then
                # we need to resize them too. This could happen if we are the intermediate
                # super resolution layers of a multi-layer cascade.
                model_input_spatial_size = config_for_layer.data.image_size

                B, C, H, W = images.shape
                if H != model_input_spatial_size or W != model_input_spatial_size:
                    images_for_layer = transforms.functional.resize(
                        images,
                        size=(
                            model_input_spatial_size,
                            model_input_spatial_size,
                        ),
                        antialias=True,
                    )

                # Calculate the loss on the batch of training data.
                loss_dict = student_model_for_layer.distillation_loss_on_batch(
                    images=images_for_layer,
                    context=context_for_layer,
                    teacher_diffusion_model=teacher_model_for_layer,
                    N=N,
                )
                loss = loss_dict["loss"]
                if torch.isnan(loss).any():
                    assert False

                # Calculate the gradients at each step in the network.
                accelerator.backward(loss)

                # On a multi-gpu machine or cluster, wait for all of the workers
                # to finish.
                accelerator.wait_for_everyone()

                # Clip the gradients.
                accelerator.clip_grad_norm_(
                    student_model_for_layer.parameters(),
                    max_grad_norm,
                )

                # Perform the gradient descent step using the optimizer.
                optimizer_for_layer.step()
                schedule_for_layer.step()

                # Resent the gradients for the next step.
                optimizer_for_layer.zero_grad()
                stage_loss += loss.item()
            # Show the current loss in the progress bar.
            stage_loss = stage_loss / len(optimizers)
            progress_bar.set_description(
                f"loss: {stage_loss:.4f} avg_loss: {average_loss:.4f}"
            )
            average_loss_cumulative += stage_loss

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                sample(
                    diffusion_model=student_diffusion_model,
                    step=step,
                    config=config,
                    num_samples=num_samples,
                    num_sampling_steps=N,
                    validation_dataloader=validation_dataloader,
                    output_path=LOCAL_OUTPUT_NAME,
                )
                save(student_diffusion_model, step, loss, optimizers, LOCAL_OUTPUT_NAME)
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save and sample the final step.
    sample(
        diffusion_model=student_diffusion_model,
        step=step,
        config=config,
        num_samples=num_samples,
        num_sampling_steps=N,
        validation_dataloader=validation_dataloader,
        output_path=LOCAL_OUTPUT_NAME,
    )
    save(student_diffusion_model, step, loss, optimizers, LOCAL_OUTPUT_NAME)


def sample(
    diffusion_model: DiffusionModel,
    step,
    config: DotConfig,
    validation_dataloader: DataLoader,
    num_sampling_steps: int,
    output_path: str,
    num_samples=64,
    sample_with_guidance: bool = False,
):
    device = next(diffusion_model.parameters()).device

    context = {}
    if "super_resolution" in config:
        images, classes = next(iter(validation_dataloader))
        prompts = convert_labels_to_prompts(classes)
        context["text_prompts"] = prompts
        context["classes"] = classes

        # Downsample to create the low resolution context
        low_resolution_spatial_size = (
            config.super_resolution.low_resolution_spatial_size
        )
        low_resolution_images = transforms.functional.resize(
            images,
            size=(
                low_resolution_spatial_size,
                low_resolution_spatial_size,
            ),
            antialias=True,
        )
        context[config.super_resolution.conditioning_key] = low_resolution_images
    else:
        # Sample from the model to check the quality.
        classes = torch.randint(
            0, config.data.num_classes, size=(num_samples,), device=device
        )
        prompts = convert_labels_to_prompts(classes)
        context["text_prompts"] = prompts
        context["classes"] = classes

    assert not sample_with_guidance, "Guidance sampling not implemented yet."
    samples, intermediate_stage_output = diffusion_model.sample(
        num_samples=num_samples,
        context=context,
        num_sampling_steps=num_sampling_steps,
        sampler=DDIMSampler(),
    )

    # Save the samples into an image grid
    utils.save_image(
        samples,
        str(f"{output_path}/sample-{step}.png"),
        nrow=int(math.sqrt(num_samples)),
    )

    # Save the intermedidate stages if they exist
    if intermediate_stage_output is not None:
        for layer_idx, intermediate_output in enumerate(intermediate_stage_output):
            utils.save_image(
                intermediate_output,
                str(f"{output_path}/sample-{step}-stage-{layer_idx}.png"),
                nrow=int(math.sqrt(num_samples)),
            )

    # Save the prompts that were used
    with open(f"{output_path}/sample-{step}.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{context['text_prompts'][i]} ")

    # Save the low-resolution imagery if it was used.
    if "super_resolution" in config:
        utils.save_image(
            context[config.super_resolution.conditioning_key],
            str(f"{output_path}/low_resolution_context-{step}.png"),
            nrow=int(math.sqrt(num_samples)),
        )


def save(
    diffusion_model,
    step,
    loss,
    optimizers: List[torch.optim.Optimizer],
    output_path: str,
):
    # Save a corresponding model checkpoint.
    torch.save(
        {
            "step": step,
            "model_state_dict": diffusion_model.state_dict(),
            "num_optimizers": len(optimizers),
            "optimizer_state_dicts": [
                optimizer.state_dict() for optimizer in optimizers
            ],
            "loss": loss,
        },
        f"{output_path}/diffusion-{step}.pt",
    )


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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--teacher_model_checkpoint", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--distillation_iterations", type=int, default=2)
    parser.add_argument("--num_training_steps_per_iteration", type=int, default=10000)
    parser.add_argument("--save_and_sample_every_n", type=int, default=100)
    parser.add_argument("--initial_sampling_steps", type=int, default=500)
    args = parser.parse_args()

    distill(
        batch_size=args.batch_size,
        teacher_model_checkpoint=args.teacher_model_checkpoint,
        config_path=args.config_path,
        distillation_iterations=args.distillation_iterations,
        num_training_steps_per_iteration=args.num_training_steps_per_iteration,
        save_and_sample_every_n=args.save_and_sample_every_n,
        initial_sampling_steps=args.initial_sampling_steps,
    )


if __name__ == "__main__":
    main()
