from accelerate import Accelerator, DataLoaderConfiguration
import argparse
from functools import partial
import math
import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm

from utils import cycle, freeze, get_obj_from_str, load_yaml, DotConfig
from text_encoder import FrozenCLIPEmbedder
from diffusion_model import GaussianDiffusion_LatentDiffusion
from score_network import MNistUnet

OUTPUT_NAME = "output"


def run_lesson_11(
    num_training_steps: int,
    batch_size: int,
    autoencoder_checkpoint: str,
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

    # Context dimension - the dimension of the context conditioning that
    # is passed to the model. During runtime, we project the embedding dimension
    # to the context dimension before passing to the model.
    text_encoder_max_length = 77

    # Create and load the VAE
    vae = get_obj_from_str(config.model.vae.target)(config.model.vae.params)
    checkpoint = torch.load(autoencoder_checkpoint, map_location="cpu")
    vae.load_state_dict(checkpoint["model_state_dict"])

    # Freeze the VAE so that we are not updating its weights.
    vae = freeze(vae)

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    diffusion_model = GaussianDiffusion_LatentDiffusion(
        score_network_type=MNistUnet, config=config, vae=vae
    )
    # The text encoder generates embeddings of size (B, text_encoder_max_length, context_dimension)
    summary(
        diffusion_model._score_network,
        [
            (
                128,
                config.model.latent_channels,
                config.model.latent_size,
                config.model.latent_size,
            ),
            (128,),
            (128, text_encoder_max_length, config.model.context_size),
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
    optimizer = Adam(diffusion_model.parameters(), lr=2e-4, betas=(0.9, 0.99))

    # Move the model and the optimizer to the accelerator as well.
    diffusion_model, optimizer = accelerator.prepare(diffusion_model, optimizer)

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

    text_encoder = FrozenCLIPEmbedder(max_length=text_encoder_max_length).to(device)
    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, labels = next(dataloader)

            # Convert the class labels into text embeddings.
            context = convert_labels_to_embeddings(labels, text_encoder)

            # Calculate the loss on the batch of training data.
            loss_dict = diffusion_model.loss_on_batch(images, context)
            loss = loss_dict["loss"]

            # Calculate the gradients at each step in the network.
            accelerator.backward(loss)

            # On a multi-gpu machine or cluster, wait for all of the workers
            # to finish.
            accelerator.wait_for_everyone()

            # Clip the gradients.
            accelerator.clip_grad_norm_(diffusion_model.parameters(), max_grad_norm)

            # Perform the gradient descent step using the optimizer.
            optimizer.step()

            # Resent the gradients for the next step.
            optimizer.zero_grad()

            # Show the current loss in the progress bar.
            progress_bar.set_description(
                f"loss: {loss.item():.4f} avg_loss: {average_loss:.4f}"
            )
            average_loss_cumulative += loss.item()

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step % save_and_sample_every_n == 0:
                sample(
                    diffusion_model=diffusion_model,
                    text_encoder=text_encoder,
                    step=step,
                    config=config,
                    conditional=False,
                )
                sample(
                    diffusion_model=diffusion_model,
                    text_encoder=text_encoder,
                    step=step,
                    config=config,
                    conditional=True,
                )
                save(diffusion_model, step, loss, optimizer)
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save and sample the final step.
    sample(
        diffusion_model=diffusion_model,
        text_encoder=text_encoder,
        step=step,
        config=config,
        conditional=False,
    )
    sample(
        diffusion_model=diffusion_model,
        text_encoder=text_encoder,
        step=step,
        config=config,
        conditional=True,
    )
    save(diffusion_model, step, loss, optimizer)


def sample(
    diffusion_model: GaussianDiffusion_LatentDiffusion,
    text_encoder,
    step,
    config: DotConfig,
    conditional: bool = False,
):
    num_samples = 64
    device = next(diffusion_model.parameters()).device

    if conditional:
        labels = torch.randint(low=0, high=10, size=(num_samples,), device=device)
        conditional_embeddings, prompts = convert_labels_to_embeddings(
            labels,
            text_encoder,
            return_prompts=True,
        )
        context = conditional_embeddings.to(device)
    else:
        context = torch.zeros(
            (num_samples, config.model.context_size),
            dtype=torch.float32,
            device=device,
        )
        labels = None
        prompts = None
    # Sample from the model to check the quality
    output = diffusion_model.sample(
        num_samples=num_samples,
        context=context,
    )

    if diffusion_model._is_class_conditional:
        samples, labels = output
    else:
        samples = output
        labels = None

    # Save the samples into an image grid
    utils.save_image(
        samples,
        str(
            f"{OUTPUT_NAME}/{'conditional' if conditional else 'unconditional'}_sample-{step}.png"
        ),
        nrow=int(math.sqrt(num_samples)),
    )

    # Save the labels if we have them
    if labels is not None:
        with open(
            f"{OUTPUT_NAME}/{'conditional' if conditional else 'unconditional'}_sample-{step}.txt",
            "w",
        ) as fp:
            for i in range(num_samples):
                if i != 0 and (i % math.sqrt(num_samples)) == 0:
                    fp.write("\n")
                fp.write(f"{labels[i]} ")

    if prompts is not None:
        with open(
            f"{OUTPUT_NAME}/{'conditional' if conditional else 'unconditional'}_sample-{step}_prompts.txt",
            "w",
        ) as fp:
            for i in range(num_samples):
                if i != 0 and (i % math.sqrt(num_samples)) == 0:
                    fp.write("\n")
                fp.write(f"{prompts[i]} ")


def save(diffusion_model, step, loss, optimizer):
    # Save a corresponding model checkpoint.
    torch.save(
        {
            "step": step,
            "model_state_dict": diffusion_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"{OUTPUT_NAME}/latent_diffusion-{step}.pt",
    )


def convert_labels_to_embeddings(
    labels: torch.Tensor, text_encoder: torch.nn.Module, return_prompts: bool = False
) -> torch.Tensor:
    """Converts MNIST class labels to embeddings.

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

    # Convert the prompts into context embeddings. Use the text encoder
    # we created earlier to convert the text labels in vector tensors.
    text_embeddings = text_encoder.encode(prompts)

    if return_prompts:
        return text_embeddings, prompts
    else:
        return text_embeddings


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--config_path", type=str, default="configs/mnist_ldm.yaml")
    parser.add_argument("--autoencoder_checkpoint", type=str, required=True)

    args = parser.parse_args()

    run_lesson_11(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config_path,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
    )


if __name__ == "__main__":
    main()
