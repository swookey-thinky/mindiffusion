"""Distillation using a consistency model.

From "Consistency Models" [https://arxiv.org/abs/2303.01469].
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
from pathlib import Path
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm
from typing import List

from xdiffusion.diffusion.ddpm import GaussianDiffusion_DDPM
from xdiffusion.diffusion import DiffusionModel
from xdiffusion.diffusion.cascade import GaussianDiffusionCascade
from xdiffusion.layers.ema import create_ema_and_scales_fn
from xdiffusion.utils import cycle, get_obj_from_str, load_yaml, DotConfig

OUTPUT_NAME = "output/image/mnist/distilled"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def distill(
    num_training_steps: int,
    batch_size: int,
    teacher_config_path: str,
    student_config_path: str,
    teacher_checkpoint: str,
    save_and_sample_every_n: int,
    resume_from: str = None,
):
    global OUTPUT_NAME
    OUTPUT_NAME = f"{OUTPUT_NAME}/{str(Path(student_config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    teacher_config = load_yaml(teacher_config_path)
    student_config = load_yaml(student_config_path)

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
                    size=(
                        student_config.data.image_size,
                        student_config.data.image_size,
                    )
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

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    if "diffusion_cascade" in teacher_config:
        teacher_diffusion_model = GaussianDiffusionCascade(teacher_config)
    elif "target" in teacher_config:
        teacher_diffusion_model = get_obj_from_str(teacher_config["target"])(
            teacher_config
        )
    else:
        teacher_diffusion_model = GaussianDiffusion_DDPM(config=teacher_config)

    if "diffusion_cascade" in student_config:
        raise NotImplementedError("Cascades not support in consistency distillation")
    elif "target" in student_config:
        student_diffusion_model = get_obj_from_str(student_config["target"])(
            student_config
        )
    else:
        student_diffusion_model = GaussianDiffusion_DDPM(config=student_config)

    # Load the teacher model weights
    teacher_diffusion_model.load_checkpoint(teacher_checkpoint)
    teacher_diffusion_model.eval()

    # Initialize the student model with the teacher model weights
    for dst, src in zip(
        student_diffusion_model.parameters(), teacher_diffusion_model.parameters()
    ):
        dst.data.copy_(src.data)
    teacher_diffusion_model.requires_grad_(False)

    # Build context to display the model summary.
    student_diffusion_model.print_model_summary()

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

    # Load the optimizers if we have them from the checkpoint
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu")
        num_optimizers = checkpoint["num_optimizers"]
        for i in range(num_optimizers):
            optimizers[i].load_state_dict(checkpoint["optimizer_state_dicts"][i])

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

    # If there is an EMA configuration section, create it here
    if "exponential_moving_average" in student_config.diffusion:
        ema_scale_fn = create_ema_and_scales_fn(
            total_steps=num_training_steps,
            distill_steps_per_iter=(
                student_config.diffusion.exponential_moving_average.distill_steps_per_iter
                if "distill_steps_per_iter"
                in student_config.diffusion.exponential_moving_average
                else 0
            ),
            **student_config.diffusion.exponential_moving_average.to_dict(),
        )
    else:
        # Default scale function returns the target ema rate
        ema_scale_fn = lambda step: 0.9999, 0

    # Step counter to keep track of training
    step = 0

    # Not mentioned in the DDPM paper, but the original implementation
    # used gradient clipping during training.
    max_grad_norm = 1.0
    average_loss = 0.0
    average_loss_cumulative = 0.0

    # Make sure the teacher model can generate good images
    sample(
        diffusion_model=teacher_diffusion_model,
        step="teacher_initial",
        config=teacher_config,
    )

    # Make sure the student diffusion model can sample images using the
    # the same sampler as the teacher
    sample(
        diffusion_model=student_diffusion_model,
        step="student_initial",
        config=student_config,
        sampler=teacher_diffusion_model._sampler,
    )

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, classes = next(dataloader)
            context = {"classes": classes}

            # Convert the labels to prompts
            prompts = convert_labels_to_prompts(classes)
            context["text_prompts"] = prompts

            # Train each cascade in the model using the given data.
            stage_loss = 0
            for model_for_layer, optimizer_for_layer, schedule_for_layer in zip(
                student_diffusion_model.models(), optimizers, learning_rate_schedules
            ):
                # Is this a super resolution model? If it is, then generate
                # the low resolution imagery as conditioning.
                config_for_layer = model_for_layer.config()
                context_for_layer = context.copy()
                images_for_layer = images

                context_for_layer["step"] = step
                context_for_layer["total_steps"] = num_training_steps

                if "super_resolution" in config_for_layer:
                    # First create the low resolution context.
                    low_resolution_spatial_size = (
                        config_for_layer.super_resolution.low_resolution_size
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
                loss_dict = model_for_layer.distillation_loss_on_batch(
                    images=images_for_layer,
                    context=context_for_layer,
                    N=1,
                    teacher_diffusion_model=teacher_diffusion_model,
                )
                loss = loss_dict["loss"]

                # Calculate the gradients at each step in the network.
                accelerator.backward(loss)

                # On a multi-gpu machine or cluster, wait for all of the workers
                # to finish.
                accelerator.wait_for_everyone()

                # Clip the gradients.
                accelerator.clip_grad_norm_(
                    model_for_layer.parameters(),
                    max_grad_norm,
                )

                # Perform the gradient descent step using the optimizer.
                optimizer_for_layer.step()
                schedule_for_layer.step()

                # Update the ema parameters for the model if they are supported
                model_for_layer.update_ema(step, num_training_steps, ema_scale_fn)

                # Reset the gradients for the next step.
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
                    config=student_config,
                    num_samples=num_samples,
                    sample_with_guidance=False,
                    validation_dataloader=validation_dataloader,
                )
                save(student_diffusion_model, step, loss, optimizers, student_config)
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
        config=student_config,
        num_samples=num_samples,
        sample_with_guidance=False,
        validation_dataloader=validation_dataloader,
    )
    save(student_diffusion_model, step, loss, optimizers, student_config)


def sample(
    diffusion_model: DiffusionModel,
    step,
    config: DotConfig,
    validation_dataloader: DataLoader = None,
    num_samples=64,
    sample_with_guidance: bool = False,
    sampler=None,
):
    device = next(diffusion_model.parameters()).device

    context = {}
    if "super_resolution" in config:
        images, classes = next(iter(validation_dataloader))
        prompts = convert_labels_to_prompts(classes)
        context["text_prompts"] = prompts
        context["classes"] = classes

        # Downsample to create the low resolution context
        low_resolution_spatial_size = config.super_resolution.low_resolution_size
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

    if sample_with_guidance:
        for guidance in [0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 20.0]:
            samples, intermediate_stage_output = diffusion_model.sample(
                num_samples=num_samples,
                context=context,
                classifier_free_guidance=guidance,
            )

            # Save the samples into an image grid
            utils.save_image(
                samples,
                str(f"{OUTPUT_NAME}/sample-{step}-cfg-{guidance}.png"),
                nrow=int(math.sqrt(num_samples)),
            )

            # Save the intermedidate stages if they exist
            if intermediate_stage_output is not None:
                for layer_idx, intermediate_output in enumerate(
                    intermediate_stage_output
                ):
                    utils.save_image(
                        intermediate_output,
                        str(
                            f"{OUTPUT_NAME}/sample-{step}-cfg-{guidance}-stage-{layer_idx}.png"
                        ),
                        nrow=int(math.sqrt(num_samples)),
                    )

    else:
        samples, intermediate_stage_output = diffusion_model.sample(
            num_samples=num_samples, context=context, sampler=sampler
        )

        # Save the samples into an image grid
        utils.save_image(
            samples,
            str(f"{OUTPUT_NAME}/sample-{step}.png"),
            nrow=int(math.sqrt(num_samples)),
        )

        # Save the intermedidate stages if they exist
        if intermediate_stage_output is not None:
            for layer_idx, intermediate_output in enumerate(intermediate_stage_output):
                utils.save_image(
                    intermediate_output,
                    str(f"{OUTPUT_NAME}/sample-{step}-stage-{layer_idx}.png"),
                    nrow=int(math.sqrt(num_samples)),
                )

    # Save the prompts that were used
    with open(f"{OUTPUT_NAME}/sample-{step}.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{context['text_prompts'][i]} ")

    # Save the low-resolution imagery if it was used.
    if "super_resolution" in config:
        utils.save_image(
            context[config.super_resolution.conditioning_key],
            str(f"{OUTPUT_NAME}/low_resolution_context-{step}.png"),
            nrow=int(math.sqrt(num_samples)),
        )


def save(
    diffusion_model,
    step,
    loss,
    optimizers: List[torch.optim.Optimizer],
    config: DotConfig,
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
            "config": config.to_dict(),
        },
        f"{OUTPUT_NAME}/diffusion-{step}.pt",
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
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--teacher_config_path", type=str, required=True)
    parser.add_argument("--student_config_path", type=str, required=True)
    parser.add_argument("--teacher_checkpoint", type=str, required=True)
    parser.add_argument("--save_and_sample_every_n", type=int, default=1000)
    args = parser.parse_args()

    distill(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        teacher_config_path=args.teacher_config_path,
        student_config_path=args.student_config_path,
        teacher_checkpoint=args.teacher_checkpoint,
        save_and_sample_every_n=args.save_and_sample_every_n,
    )


if __name__ == "__main__":
    main()
