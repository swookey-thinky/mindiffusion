"""Train a DaLL-E model for MNIST.

Train a model to generate text conditional MNIST samples.
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
from einops import rearrange
import math
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm

from dvae import DiscreteVAE
from dall_e import DalleModel
from tokenizer import SimpleTokenizer
from utils import (
    cycle,
    map_pixels,
    freeze,
    convert_labels_to_tokens,
    load_yaml,
)

OUTPUT_NAME = "output/dall_e"


def run_lesson_08_dall_e(
    num_training_steps: int, batch_size: int, dvae_checkpoint: str, config_path: str
):
    if not dvae_checkpoint:
        print("Argument dvae_checkpoint must be specified!")
        return

    # Ensure the output directories exist
    os.makedirs(OUTPUT_NAME, exist_ok=True)

    # Load the MNIST dataset. This is a supervised dataset so
    # it contains both images and class labels. We will ignore the class
    # labels for now.
    image_spatial_size = 32
    dataset = MNIST(
        ".",
        train=True,
        transform=transforms.Compose(
            [
                # To make the math work out easier, resize the MNIST
                # images from (28,28) to (32, 32).
                transforms.Resize(size=(image_spatial_size, image_spatial_size)),
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

    # Load the dVAE from the checkpoint
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

    # Load the DALL*E model we will train
    dall_e = DalleModel(
        num_layers=config.model.num_layers,
        hidden_size=config.model.hidden_size,
        num_attention_heads=config.model.num_attention_heads,
        num_text_tokens_in_sequence=config.model.num_text_tokens_in_sequence,
        num_image_tokens_per_dim=config.model.num_image_tokens_per_dim,
        text_vocab_size=config.model.text_vocab_size,
        image_vocab_size=dvae.vocab_size,
    )
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
    device = accelerator.device

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader = accelerator.prepare(dataloader)

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Basic optimizer for training.
    optimizer = torch.optim.AdamW(
        dall_e.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_training_steps // 100, eta_min=1e-6
    )

    # Move the model and the optimizer to the accelerator as well.
    dvae, dall_e, optimizer, scheduler = accelerator.prepare(
        dvae, dall_e, optimizer, scheduler
    )
    dvae = freeze(dvae)

    # Step counter to keep track of training
    step = 0

    # We will sample the autoencoder every N steps, to monitor
    # training and see how it improves over time.
    save_and_sample_every_n = 100

    # Clip the gradients for smoother training.
    max_grad_norm = 1.0
    temp = 1.0

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, labels = next(dataloader)

            # Convert the labels to tokens
            with torch.no_grad():
                text_tokens, prompts = convert_labels_to_tokens(
                    labels,
                    tokenizer,
                    text_token_length=config.model.num_text_tokens_in_sequence,
                    return_prompts=True,
                )
                image_tokens = dvae.get_codebook_indices(map_pixels(images))
                input_ids = torch.cat([text_tokens.to(device), image_tokens], dim=1)

            # Calculate the loss on the batch of training data.
            loss, _ = dall_e(input_ids, return_loss=True)

            # Calculate the gradients at each step in the network.
            accelerator.backward(loss)

            # On a multi-gpu machine or cluster, wait for all of the workers
            # to finish.
            accelerator.wait_for_everyone()

            # Clip the gradients.
            accelerator.clip_grad_norm_(dall_e.parameters(), max_grad_norm)

            # Perform the gradient descent step using the optimizer.
            optimizer.step()

            # Resent the gradients for the next step.
            optimizer.zero_grad()

            # Show the current loss in the progress bar.
            progress_bar.set_description(f"loss: {loss.item():.4f}")

            # To help visualize training, periodically sample from the
            # autoencoder to see how well its doing.
            if step % save_and_sample_every_n == 0:
                samples, prompts = dall_e.sample(
                    dvae=dvae,
                    num_samples=config.sampling.num_samples,
                    batch_size=config.sampling.sample_batch_size,
                    tokenizer=tokenizer,
                )

                utils.save_image(
                    samples,
                    str(f"{OUTPUT_NAME}/sample-{step}.png"),
                    nrow=int(math.sqrt(config.sampling.num_samples)),
                )

                with open(f"{OUTPUT_NAME}/sample_prompts-{step}.txt", "w") as fp:
                    for i in range(config.sampling.num_samples):
                        if i != 0 and (i % math.sqrt(config.sampling.num_samples)) == 0:
                            fp.write("\n")
                        fp.write(f"{prompts[i]} ")

                # Save a corresponding model checkpoint.
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": dall_e.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    f"{OUTPUT_NAME}/dall_e-{step}.pt",
                )

                temp = max(temp * math.exp(-1e-6 * step), 0.5)
                scheduler.step()

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save the final output.
    samples, prompts = dall_e.sample(
        dvae=dvae,
        num_samples=config.sampling.num_samples,
        batch_size=config.sampling.sample_batch_size,
        tokenizer=tokenizer,
    )

    utils.save_image(
        samples,
        str(f"{OUTPUT_NAME}/sample-{step}.png"),
        nrow=int(math.sqrt(config.sampling.num_samples)),
    )
    with open(f"{OUTPUT_NAME}/sample_prompts-{step}.txt", "w") as fp:
        for i in range(config.sampling.num_samples):
            if i != 0 and (i % math.sqrt(config.sampling.num_samples)) == 0:
                fp.write("\n")
            fp.write(f"{prompts[i]} ")
    torch.save(
        {
            "step": step,
            "model_state_dict": dall_e.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"{OUTPUT_NAME}/dall_e-{step}.pt",
    )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=60000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--dvae_checkpoint", type=str, default="")
    parser.add_argument("--config_path", type=str, default="configs/dall_e_mnist.yaml")
    args = parser.parse_args()

    run_lesson_08_dall_e(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        dvae_checkpoint=args.dvae_checkpoint,
        config_path=args.config_path,
    )


if __name__ == "__main__":
    main()
