"""Lesson 6 - Denoising Diffusion Implicit Models

This lesson demonstrates deterministic sampling from the paper
"Denoising Diffusion Implicit Models" (https://arxiv.org/pdf/2010.02502).
Importantly, this paper uses a pre-trained DDPM model (in this case, from
Lesson 5e) and uses it to perform sampling in only 50 timesteps, versus the 1000
timesteps from DDPM.
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
from functools import partial
import math
import os
import torch
from torchvision.utils import save_image

import os
from pathlib import Path
import sys

# HACK: Reuse the packages from the parent directory. It will try to import
#       packages in this directory first, then move to the packages in the
#       parent directory.
sys.path.append(
    os.path.join(
        str(Path(os.path.dirname(__file__)).parent.absolute()), "lesson_05/sublesson_d"
    )
)
sys.path.append(
    os.path.join(
        str(Path(os.path.dirname(__file__)).parent.absolute()), "lesson_05/sublesson_e"
    )
)

from ddim import sample_ddim
from diffusion_model import GaussianDiffusion_ConditionalDDPM
from score_network import ConditionalMNistUNet
from utils import (
    extract,
    linear_beta_schedule,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)
from text_encoder import FrozenCLIPEmbedder
from text_embeddings import convert_labels_to_embeddings

OUTPUT_NAME = "output"


def run_lesson_6_sampling(model_checkpoint: str, num_examples: int):
    # Fix the random seed for reproducibility.
    torch.manual_seed(42)

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Context dimension - the dimension of the context conditioning that
    # is passed to the model. During runtime, we project the embedding dimension
    # to the context dimension before passing to the model.
    context_dimension = 768

    # The text encoder generates embeddings of size
    # (B, text_encoder_max_length, context_dimension)
    text_encoder_max_length = 77

    # We will load the diffusion model from lesson 5 and interpolate
    # the latents generated from it.
    diffusion_model = GaussianDiffusion_ConditionalDDPM(
        unet_type=partial(
            ConditionalMNistUNet, dropout=0.1, context_dimension=context_dimension
        )
    )

    # Load the model checkpoint
    checkpoint = torch.load(model_checkpoint)
    diffusion_model.load_state_dict(checkpoint["model_state_dict"])

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        DataLoaderConfiguration(split_batches=False), mixed_precision="no"
    )
    device = accelerator.device

    # Move the model to the accelerator as well.
    diffusion_model = accelerator.prepare(diffusion_model)

    # Unconditionally sample from the DDIM sampler
    num_samples = 64
    unconditional_samples = sample_ddim(
        epsilon_network=diffusion_model._unet,
        image_size=32,
        num_channels=1,
        num_timesteps=diffusion_model._num_timesteps,
        batch_size=num_samples,
        num_sampling_timesteps=50,
        context_dimension=context_dimension,
    )
    save_image(unconditional_samples, str(f"{OUTPUT_NAME}/unconditional_samples.png"))

    # Now create some conditional samples.
    text_encoder = FrozenCLIPEmbedder(max_length=text_encoder_max_length).to(device)

    labels = torch.randint(low=0, high=10, size=(num_samples,), device=device)
    conditional_embeddings, prompts = convert_labels_to_embeddings(
        labels,
        text_encoder,
        return_prompts=True,
    )
    conditional_embeddings = conditional_embeddings.to(device)

    conditional_samples = sample_ddim(
        epsilon_network=diffusion_model._unet,
        image_size=32,
        num_channels=1,
        num_timesteps=diffusion_model._num_timesteps,
        batch_size=num_samples,
        num_sampling_timesteps=50,
        context_dimension=context_dimension,
        y=conditional_embeddings,
    )

    save_image(conditional_samples, str(f"{OUTPUT_NAME}/conditional_samples.png"))
    with open(f"{OUTPUT_NAME}/conditional_sample_prompts.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{prompts[i]} ")


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=64)
    args = parser.parse_args()

    run_lesson_6_sampling(
        model_checkpoint=args.model_checkpoint, num_examples=args.num_examples
    )


if __name__ == "__main__":
    main()
