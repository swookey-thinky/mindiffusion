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

from image_diffusion.utils import cycle, load_yaml, DotConfig
from image_diffusion.ddpm import GaussianDiffusion_DDPM
from image_diffusion.diffusion import DiffusionModel
from image_diffusion.cascade import GaussianDiffusionCascade
from image_diffusion.samplers import ddim, ancestral, base

OUTPUT_NAME = "output/mnist/sample"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sample_model(
    config_path: str,
    num_samples: int,
    guidance: float,
    checkpoint_path: str,
    sampling_steps: int,
    sampler: str,
):
    global OUTPUT_NAME
    OUTPUT_NAME = f"{OUTPUT_NAME}/{str(Path(config_path).stem)}"

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    if "diffusion_cascade" in config:
        diffusion_model = GaussianDiffusionCascade(config)
    else:
        diffusion_model = GaussianDiffusion_DDPM(config=config)

    # Load the model weights if we have them
    if checkpoint_path:
        diffusion_model.load_checkpoint(checkpoint_path)

    # Build context to display the model summary.
    diffusion_model.print_model_summary()

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Move the model and the optimizer to the accelerator as well.
    diffusion_model = accelerator.prepare(diffusion_model)

    if sampler == "ddim":
        sampler = ddim.DDIMSampler()
    elif sampler == "ancestral":
        sampler = ancestral.AncestralSampler()
    else:
        raise NotImplemented(f"Sampler {sampler} not implemented.")

    # Save and sample the final step.
    sample(
        diffusion_model=diffusion_model,
        config=config,
        num_samples=num_samples,
        num_sampling_steps=sampling_steps,
        sampler=sampler,
    )


def sample(
    diffusion_model: DiffusionModel,
    config: DotConfig,
    sampler: base.ReverseProcessSampler,
    num_samples=64,
    num_sampling_steps=1000,
):
    device = next(diffusion_model.parameters()).device

    context = {}
    if "super_resolution" in config:
        assert False, "Not supported yet."

    # Sample from the model to check the quality.
    classes = torch.randint(
        0, config.data.num_classes, size=(num_samples,), device=device
    )
    prompts = convert_labels_to_prompts(classes)
    context["text_prompts"] = prompts
    context["classes"] = classes

    samples, intermediate_stage_output = diffusion_model.sample(
        num_samples=num_samples,
        context=context,
        num_sampling_steps=num_sampling_steps,
        sampler=sampler,
    )

    # Save the samples into an image grid
    utils.save_image(
        samples,
        str(f"{OUTPUT_NAME}/sample.png"),
        nrow=int(math.sqrt(num_samples)),
    )

    # Save the intermedidate stages if they exist
    if intermediate_stage_output is not None:
        for layer_idx, intermediate_output in enumerate(intermediate_stage_output):
            utils.save_image(
                intermediate_output,
                str(f"{OUTPUT_NAME}/sample-stage-{layer_idx}.png"),
                nrow=int(math.sqrt(num_samples)),
            )

    # Save the prompts that were used
    with open(f"{OUTPUT_NAME}/sample.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{context['text_prompts'][i]} ")


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
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--sampling_steps", type=int, default=1000)
    parser.add_argument("--sampler", type=str, default="ancestral")
    args = parser.parse_args()

    sample_model(
        config_path=args.config_path,
        num_samples=args.num_samples,
        guidance=args.guidance,
        checkpoint_path=args.checkpoint,
        sampling_steps=args.sampling_steps,
        sampler=args.sampler,
    )


if __name__ == "__main__":
    main()
