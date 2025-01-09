from accelerate import cpu_offload, Accelerator, DataLoaderConfiguration
from accelerate.utils import GradientAccumulationPlugin
from accelerate import DistributedDataParallelKwargs
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
from typing import Callable, List, Optional

from xdiffusion.datasets.utils import load_dataset
from xdiffusion.diffusion.ddpm import GaussianDiffusion_DDPM
from xdiffusion.diffusion import DiffusionModel
from xdiffusion.diffusion.cascade import GaussianDiffusionCascade
from xdiffusion.layers.ema import create_ema_and_scales_fn
from xdiffusion.lora import inject_trainable_lora, save_lora_weights
from xdiffusion.utils import (
    cycle,
    freeze,
    get_obj_from_str,
    instantiate_from_config,
    load_yaml,
    DotConfig,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(
    num_training_steps: int,
    batch_size: int,
    config_path: str,
    dataset_name: str,
    output_path: str,
    sample_with_guidance: bool = False,
    save_and_sample_every_n: int = 1000,
    load_model_weights_from_checkpoint: str = "",
    resume_from: str = "",
    mixed_precision: str = "",
    use_lora_training: bool = False,
    compile: Optional[bool] = None,
):
    """Trains an image diffusion model from a config file.

    Args:
        num_training_steps: The number of training steps to train for.
        batch_size: The batch size to use in training.
        config_path: Full path to the configuration file for training.
        dataset_name: The name of the dataset to use in training.
        output_path: Full path to store output artifacts.
    """
    # Open the model configuration
    config = load_yaml(config_path)

    if "training" in config and "dataset" in config.training:
        dataset_name = config.training.dataset
    else:
        if not dataset_name:
            raise ValueError(
                "--dataset_name must be passed if there is no dataset specified in the config file."
            )

    # Use the batch size from the training path, unless overridden by the command line
    if "training" in config and "batch_size" in config.training:
        if batch_size <= 0:
            # Only override if its not on the command line
            batch_size = config.training.batch_size
    else:
        # It's not in the config file, make sure its specified
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be specified in the configuration file or on the command line with --batch_size"
            )

    if use_lora_training:
        OUTPUT_NAME = f"{output_path}/{dataset_name}/{str(Path(config_path).stem)}/lora"
    else:
        OUTPUT_NAME = f"{output_path}/{dataset_name}/{str(Path(config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Check to see if we are using gradient accumulation
    gradient_accumulation_steps = 1
    if "training" in config and "gradient_accumulation_steps" in config.training:
        gradient_accumulation_steps = config.training.gradient_accumulation_steps

    if "training" in config and not mixed_precision:
        mixed_precision = config.training.mixed_precision
    else:
        if not mixed_precision:
            mixed_precision = "no"

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # The accelerate library will handle of the GPU device management for us.
    # Make sure to create it early so that we can gate some data loading on it.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision=mixed_precision,
        gradient_accumulation_plugin=(
            GradientAccumulationPlugin(
                num_steps=gradient_accumulation_steps,
                adjust_scheduler=True,
                sync_with_dataloader=False,
            )
            if gradient_accumulation_steps > 1
            else None
        ),
        kwargs_handlers=[ddp_kwargs],
        step_scheduler_with_optimizer=False,
    )
    accelerator.print(
        f"Training with {gradient_accumulation_steps} gradient accumulation steps."
    )
    accelerator.print(f"Training with mixed precision: {mixed_precision}.")

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    if "diffusion_cascade" in config:
        diffusion_model = GaussianDiffusionCascade(config)
    elif "target" in config:
        diffusion_model = get_obj_from_str(config["target"])(config)
    else:
        diffusion_model = GaussianDiffusion_DDPM(config=config)

    # Load the model weights if we have them
    if load_model_weights_from_checkpoint:
        diffusion_model.load_checkpoint(load_model_weights_from_checkpoint)

    if resume_from:
        diffusion_model.load_checkpoint(resume_from)

    if use_lora_training:
        # Inject the trainable LoRA parameters
        print("Injecting and training LoRA weights only.")
        diffusion_model = freeze(diffusion_model)
        _, _ = inject_trainable_lora(diffusion_model, verbose=True)

    # Build context to display the model summary.
    diffusion_model.print_model_summary()

    # Now load the dataset. Do it on the main process first in case we have to download
    # it.
    with accelerator.main_process_first():
        if "training" in config and "dataset" in config.training:
            dataset, convert_labels_to_prompts = load_dataset(
                dataset_name=config.training.dataset, config=config.data, split="train"
            )
            validation_dataset, _ = load_dataset(
                dataset_name=config.training.dataset,
                config=config.data,
                split="validation",
            )

        else:
            dataset, convert_labels_to_prompts = load_dataset(
                dataset_name=dataset_name, config=config.data, split="train"
            )
            validation_dataset, _ = load_dataset(
                dataset_name=dataset_name, config=config.data, split="validation"
            )

    # Create the dataloader for the MNIST dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    num_samples = 64
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=num_samples, shuffle=True, num_workers=1
    )

    # Now create the optimizer. The optimizer choice and parameters come from
    # the paper:
    # "We tried Adam [31] and RMSProp early on in our experimentation process and chose the
    #  former. We left the hyperparameters to their standard values. We set the learning
    #  rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5
    #  for the 256 × 256 images, which seemed unstable to train with the larger learning rate."
    optimizers = diffusion_model.configure_optimizers(learning_rate=2e-4)

    # Load the optimizers if we have them from the checkpoint
    if resume_from:
        checkpoint = torch.load(resume_from, map_location="cpu")
        num_optimizers = checkpoint["num_optimizers"]
        for i in range(num_optimizers):
            optimizers[i].load_state_dict(checkpoint["optimizer_state_dicts"][i])

    # Configure the learning rate schedule
    learning_rate_schedules = diffusion_model.configure_learning_rate_schedule(
        optimizers
    )

    # Move everything to the accelerator together
    all_device_objects = accelerator.prepare(
        diffusion_model,
        dataloader,
        validation_dataloader,
        *optimizers,
        *learning_rate_schedules,
    )
    diffusion_model = all_device_objects[0]
    dataloader = all_device_objects[1]
    validation_dataloader = all_device_objects[2]
    optimizers = all_device_objects[3 : 3 + len(optimizers)]
    learning_rate_schedules = all_device_objects[3 + len(optimizers) :]
    assert len(optimizers) == len(
        learning_rate_schedules
    ), "Optimizers and learning rate schedules are not the same length!"

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # If there is a separate sampling section, create the prompt encoder
    # here, and put it on the CPU.
    if "sampling" in config:
        prompt_encoder = instantiate_from_config(
            config.sampling.prompt_encoder.to_dict()
        )

        if getattr(prompt_encoder, "enable_cpu_offload", False):
            prompt_encoder = cpu_offload(
                prompt_encoder,
                execution_device=next(iter(diffusion_model.parameters())).device,
            )
    else:
        prompt_encoder = None

    # If there is an EMA configuration section, create it here
    if "diffusion" in config and "exponential_moving_average" in config.diffusion:
        ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=config.diffusion.exponential_moving_average.target_ema_mode,
            start_ema=config.diffusion.exponential_moving_average.start_ema,
            scale_mode=config.diffusion.exponential_moving_average.scale_mode,
            start_scales=config.diffusion.exponential_moving_average.start_scales,
            end_scales=config.diffusion.exponential_moving_average.end_scales,
            total_steps=num_training_steps,
            distill_steps_per_iter=(
                config.diffusion.exponential_moving_average.distill_steps_per_iter
                if "distill_steps_per_iter"
                in config.diffusion.exponential_moving_average
                else 0
            ),
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

    do_compile = False
    if compile is not None:
        do_compile = compile
    else:
        if "training" in config and "compile" in config.training:
            do_compile = config.training.compile
    accelerator.print(f"Model compilation setting: {do_compile}")
    if do_compile:
        diffusion_model = torch.compile(diffusion_model)

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # All of the gradient accumulation steps count as one training step.
            for _ in range(gradient_accumulation_steps):
                with accelerator.accumulate(diffusion_model):
                    # The dataset has images and classes. Let's use the classes,
                    # and convert them into a fixed embedding space.
                    example_data = next(dataloader)

                    if len(example_data) == 2:
                        images, classes = example_data
                        context = {"classes": classes}

                        # Convert the labels to prompts
                        prompts = convert_labels_to_prompts(classes)
                        context["text_prompts"] = prompts

                    else:
                        images, classes, context_data = example_data
                        context = {"classes": classes}
                        context.update(context_data)

                    context["step"] = step
                    context["total_steps"] = num_training_steps

                    if "super_resolution" in config:
                        # First create the low resolution context.
                        low_resolution_spatial_size = (
                            config.super_resolution.low_resolution_size
                        )
                        low_resolution_images = transforms.functional.resize(
                            images,
                            size=(
                                low_resolution_spatial_size,
                                low_resolution_spatial_size,
                            ),
                            antialias=True,
                        )
                        context[config.super_resolution.conditioning_key] = (
                            low_resolution_images
                        )

                    # If the images are not the right shape for the model input, then
                    # we need to resize them too. This could happen if we are the intermediate
                    # super resolution layers of a multi-layer cascade.
                    model_input_spatial_size = config.data.image_size

                    B, C, H, W = images.shape
                    if H != model_input_spatial_size or W != model_input_spatial_size:
                        images = transforms.functional.resize(
                            images,
                            size=(
                                model_input_spatial_size,
                                model_input_spatial_size,
                            ),
                            antialias=True,
                        )

                    # Calculate the loss for each layer
                    with accelerator.autocast():
                        losses = diffusion_model(images=images, context=context)

                    # Calculate the gradients at each step in the network.
                    loss = losses["loss"]
                    accelerator.backward(loss)

                    # Clip the gradients.
                    accelerator.clip_grad_norm_(
                        diffusion_model.parameters(),
                        max_grad_norm,
                    )

                    for optimizer_for_layer, schedule_for_layer in zip(
                        optimizers, learning_rate_schedules
                    ):
                        # Perform the gradient descent step using the optimizer.
                        optimizer_for_layer.step()
                        schedule_for_layer.step()

                        # Reset the gradients for the next step.
                        optimizer_for_layer.zero_grad()

                    # Update the ema parameters for the model if they are supported.
                    # We should only update after each gradient accumulation step however,
                    # because these steps are not sync'd with accelerate
                    if gradient_accumulation_steps == 1 or (
                        step > 0 and step % gradient_accumulation_steps == 0
                    ):
                        accelerator.unwrap_model(diffusion_model).update_ema(
                            step, num_training_steps, ema_scale_fn
                        )

            # On a multi-gpu machine or cluster, wait for all of the workers
            # to finish after all of the gradient accumulation steps.
            accelerator.wait_for_everyone()

            # Show the current loss in the progress bar.
            stage_loss = loss.item() / len(optimizers)
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
                    num_samples=num_samples,
                    output_path=OUTPUT_NAME,
                    convert_labels_to_prompts=convert_labels_to_prompts,
                    sample_with_guidance=sample_with_guidance,
                    validation_dataloader=validation_dataloader,
                    accelerator=accelerator,
                    prompt_encoder=prompt_encoder,
                )

                if accelerator.is_main_process:
                    save(
                        diffusion_model=accelerator.unwrap_model(diffusion_model),
                        step=step,
                        loss=loss,
                        optimizers=optimizers,
                        config=config,
                        output_path=OUTPUT_NAME,
                        save_lora=use_lora_training,
                    )
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Update the final loss
    average_loss = average_loss_cumulative / float(save_and_sample_every_n)
    average_loss_cumulative = 0.0
    progress_bar.set_description(f"loss: {stage_loss:.4f} avg_loss: {average_loss:.4f}")

    # Save and sample the final step.
    sample(
        diffusion_model=diffusion_model,
        step=step,
        config=config,
        num_samples=num_samples,
        output_path=OUTPUT_NAME,
        convert_labels_to_prompts=convert_labels_to_prompts,
        sample_with_guidance=sample_with_guidance,
        validation_dataloader=validation_dataloader,
        accelerator=accelerator,
        prompt_encoder=prompt_encoder,
    )
    if accelerator.is_main_process:
        save(
            diffusion_model=accelerator.unwrap_model(diffusion_model),
            step=step,
            loss=loss,
            optimizers=optimizers,
            config=config,
            output_path=OUTPUT_NAME,
            save_lora=use_lora_training,
        )
    accelerator.end_training()


def sample(
    diffusion_model: DiffusionModel,
    step,
    config: DotConfig,
    validation_dataloader: DataLoader,
    output_path: str,
    convert_labels_to_prompts: Callable[[torch.Tensor], List[str]],
    accelerator: Accelerator,
    num_samples=64,
    sample_with_guidance: bool = False,
    prompt_encoder: Optional[torch.nn.Module] = None,
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
        example_data = next(iter(validation_dataloader))
        classes = example_data[1].to(device)

        prompts = convert_labels_to_prompts(classes)
        context["text_prompts"] = prompts
        context["classes"] = classes

    if sample_with_guidance:
        for guidance in [0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 20.0]:
            samples, intermediate_stage_output = diffusion_model.sample(
                num_samples=num_samples,
                context=context,
                classifier_free_guidance=guidance,
                context_preprocessor=prompt_encoder,
            )

            # Save the samples into an image grid
            utils.save_image(
                samples,
                str(f"{output_path}/sample-{step}-cfg-{guidance}.png"),
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
                            f"{output_path}/sample-{step}-cfg-{guidance}-stage-{layer_idx}.png"
                        ),
                        nrow=int(math.sqrt(num_samples)),
                    )

    else:
        samples, intermediate_stage_output = accelerator.unwrap_model(
            diffusion_model
        ).sample(
            num_samples=num_samples,
            context=context,
            context_preprocessor=prompt_encoder,
        )

        # Save the samples into an image grid
        if accelerator.is_main_process:
            utils.save_image(
                samples,
                str(f"{output_path}/sample-{step}.png"),
                nrow=int(math.sqrt(num_samples)),
            )

            # Save the intermedidate stages if they exist
            if intermediate_stage_output is not None:
                for layer_idx, intermediate_output in enumerate(
                    intermediate_stage_output
                ):
                    utils.save_image(
                        intermediate_output,
                        str(f"{output_path}/sample-{step}-stage-{layer_idx}.png"),
                        nrow=int(math.sqrt(num_samples)),
                    )

    if accelerator.is_main_process:
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
    output_path: str,
    optimizers: List[torch.optim.Optimizer],
    config: DotConfig,
    save_lora: bool = False,
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
        f"{output_path}/diffusion-{step}.pt",
    )

    # Save the lora weights separately
    if save_lora:
        save_lora_weights(diffusion_model, f"{output_path}/diffusion-{step}-lora.pt")
