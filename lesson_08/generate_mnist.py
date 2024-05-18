"""Generate samples from MNIST using DaLL-E."""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import math
import os
import torch
from torchinfo import summary
from torchvision import utils

from utils import freeze, load_yaml
from dvae import DiscreteVAE
from dall_e import DalleModel
from tokenizer import SimpleTokenizer

OUTPUT_NAME = "output/dall_e_samples"


def run_lesson_08_dall_e_generate(
    dvae_checkpoint: str, dall_e_checkpoint: str, config_path: str
):
    if not dvae_checkpoint:
        print("Argument dvae_checkpoint must be specified!")
        return

    if not dall_e_checkpoint:
        print("Argument dall_e_checkpoint must be specified!")
        return

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Open the model configuration
    config = load_yaml(config_path)

    # Load the dVAE from the checkpoint
    image_spatial_size = config.data.image_size
    dvae = DiscreteVAE(
        num_groups=config.model.vae.num_groups,
        input_channels=config.model.vae.input_channels,
        vocab_size=config.model.vae.vocab_size,
        hidden_size=config.model.vae.hidden_size,
        num_blocks_per_group=config.model.vae.num_blocks_per_group,
    )
    checkpoint = torch.load(dvae_checkpoint)
    dvae.load_state_dict(checkpoint["model_state_dict"])

    summary(
        dvae,
        [
            (
                128,
                config.model.vae.input_channels,
                image_spatial_size,
                image_spatial_size,
            )
        ],
    )

    # BPE text tokenizer
    tokenizer = SimpleTokenizer()

    # Load the DALL*E model we will sample from
    dall_e = DalleModel(
        num_layers=config.model.num_layers,
        hidden_size=config.model.hidden_size,
        num_attention_heads=config.model.num_attention_heads,
        num_text_tokens_in_sequence=config.model.num_text_tokens_in_sequence,
        num_image_tokens_per_dim=config.model.num_image_tokens_per_dim,
        text_vocab_size=config.model.text_vocab_size,
        image_vocab_size=dvae.vocab_size,
    )
    checkpoint = torch.load(dall_e_checkpoint)
    dall_e.load_state_dict(checkpoint["model_state_dict"])

    summary(
        dall_e,
        [
            (
                128,
                config.model.num_text_tokens_in_sequence
                + config.model.num_image_tokens_per_dim**2,
            )
        ],
        dtypes=[torch.int32],
    )

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )

    # Move the model and the optimizer to the accelerator as well.
    dvae, dall_e = accelerator.prepare(dvae, dall_e)
    dvae = freeze(dvae)

    num_samples = 64
    samples = dall_e.sample(
        dvae=dvae,
        num_samples=num_samples,
        batch_size=8,
        tokenizer=tokenizer,
    )

    utils.save_image(
        samples,
        str(f"{OUTPUT_NAME}/sample.png"),
        nrow=int(math.sqrt(num_samples)),
    )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dvae_checkpoint", type=str, default="")
    parser.add_argument("--dall_e_checkpoint", type=str, default="")
    parser.add_argument("--config_path", type=str, default="configs/dall_e_mnist.yaml")
    args = parser.parse_args()

    run_lesson_08_dall_e_generate(
        dvae_checkpoint=args.dvae_checkpoint,
        dall_e_checkpoint=args.dall_e_checkpoint,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
