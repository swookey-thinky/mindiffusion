"""Lesson 17 - DaLL*E 2 Sampling

Implements text-to-image generation using DaLL*E 2 pipeline.

```
python sample_dalle2.py --diffusion_decoder <path to diffusion decoder checkpoint> --diffusion_prior <path to diffusion prior checkpoint>
```
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
import torch
from torchinfo import summary
from torchvision import utils
from typing import List

from clip import FrozenCLIPEmbedder
from diffusion_prior import GaussianDiffusion_DaLLE2_Prior
from diffusion_decoder import GaussianDiffusion_DaLLE2_Decoder
from utils import load_yaml, DotConfig

OUTPUT_NAME = "output/samples_dalle2"


def run_lesson_17_sample(
    diffusion_prior_path: str,
    diffusion_decoder_path: str,
    num_samples: int,
    config_path: str,
):
    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

    # Create the diffusion model we are going to sample, with a UNet
    # specifically for the MNIST dataset.
    diffusion_prior = GaussianDiffusion_DaLLE2_Prior(config=config)
    checkpoint = torch.load(diffusion_prior_path)
    diffusion_prior.load_state_dict(checkpoint["model_state_dict"])
    summary(
        diffusion_prior._score_network,
        [
            (
                128,
                config.diffusion_prior.text_encoder.tokens_in_sequence + 4,
                config.diffusion_prior.model.transformer.context_size,
            ),
        ],
    )

    diffusion_decoder = GaussianDiffusion_DaLLE2_Decoder(
        config=config.diffusion_decoder
    )
    checkpoint = torch.load(diffusion_decoder_path)
    diffusion_decoder.load_state_dict(checkpoint["model_state_dict"])
    summary(
        diffusion_decoder._score_network,
        [
            (
                128,
                config.diffusion_decoder.model.input_channels,
                config.diffusion_decoder.model.input_spatial_size,
                config.diffusion_decoder.model.input_spatial_size,
            ),  # x
            (128,),  # t
            (128, config.diffusion_decoder.model.context_size),  # text_embeddings
            (128, config.diffusion_decoder.model.context_size),  # image_embeddings
        ],
    )

    # Load the frozen CLIP embedder
    clip_embedder = FrozenCLIPEmbedder()

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Move the model and the classifier to the accelerator as well.
    diffusion_prior, diffusion_decoder, clip_embedder = accelerator.prepare(
        diffusion_prior, diffusion_decoder, clip_embedder
    )
    sample(
        diffusion_prior=diffusion_prior,
        diffusion_decoder=diffusion_decoder,
        clip_embedder=clip_embedder,
        config=config,
        num_samples=num_samples,
    )


def sample(
    diffusion_prior: GaussianDiffusion_DaLLE2_Prior,
    diffusion_decoder: GaussianDiffusion_DaLLE2_Decoder,
    clip_embedder: FrozenCLIPEmbedder,
    config: DotConfig,
    num_samples=64,
):
    device = next(diffusion_prior.parameters()).device

    # Sample from the model to check the quality.
    # First generate the text labels for the conditioning.
    classes = torch.randint(
        0, config.data.num_classes, size=(num_samples,), device=device
    )
    prompts = convert_labels_to_prompts(classes)

    # Sample from the prior to generate the image embeddings that
    # we use to condition the decoder.
    image_embeddings, text_embeddings = diffusion_prior.sample(
        prompts=prompts, num_samples=num_samples, clip_embedder=clip_embedder
    )

    # Finally, sample from the decoder network with the image and text embeddings
    # as conditional context.
    images = diffusion_decoder.sample(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        num_samples=num_samples,
    )

    # Save the samples into an image grid
    utils.save_image(
        images,
        str(f"{OUTPUT_NAME}/sample-dalle2.png"),
        nrow=int(math.sqrt(num_samples)),
    )

    # Save the prompts we used.
    with open(f"{OUTPUT_NAME}/sample.txt", "w") as fp:
        for i in range(num_samples):
            if i != 0 and (i % math.sqrt(num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{prompts[i]} ")


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
    parser.add_argument("--diffusion_prior", type=str)
    parser.add_argument("--diffusion_decoder", type=str)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--config", type=str, default="configs/mnist_dalle2.yaml")
    args = parser.parse_args()

    run_lesson_17_sample(
        diffusion_prior_path=args.diffusion_prior,
        diffusion_decoder_path=args.diffusion_decoder,
        num_samples=args.num_samples,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
