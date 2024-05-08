from accelerate import Accelerator
import argparse
from functools import partial
import math
import os
from pathlib import Path
import sys
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms, utils
from torchvision.datasets import MNIST
from tqdm import tqdm

# HACK: Reuse the packages from the parent directory. It will try to import
#       packages in this directory first, then move to the packages in the
#       parent directory.
sys.path.append(
    os.path.join(str(Path(os.path.dirname(__file__)).parent.absolute()), "sublesson_d")
)

from utils import cycle
from diffusion_model import GaussianDiffusion_ConditionalDDPM
from score_network import ConditionalMNistUNet
from text_encoder import FrozenCLIPEmbedder
from text_embeddings import convert_labels_to_embeddings

OUTPUT_NAME = "output"


def run_lesson_5e(num_training_steps: int, batch_size: int):
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

    # Context dimension - the dimension of the context conditioning that
    # is passed to the model. During runtime, we project the embedding dimension
    # to the context dimension before passing to the model.
    context_dimension = 768

    # The text encoder generates embeddings of size
    # (B, text_encoder_max_length, context_dimension)
    text_encoder_max_length = 77

    # Create the diffusion model we are going to train, with a UNet
    # specifically for the MNIST dataset.
    diffusion_model = GaussianDiffusion_ConditionalDDPM(
        unet_type=partial(
            ConditionalMNistUNet, dropout=0.1, context_dimension=context_dimension
        )
    )

    # The text encoder generates embeddings of size
    # (B, text_encoder_max_length, context_dimension)
    summary(
        diffusion_model._unet,
        [(128, 1, 32, 32), (128,), (128, text_encoder_max_length, context_dimension)],
    )

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(split_batches=False, mixed_precision="no")
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

    # Create the fozen text encoder for MNIST label embedding.
    text_encoder = FrozenCLIPEmbedder(max_length=text_encoder_max_length).to(device)

    # Start training
    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes. Let's use the classes,
            # and convert them into a fixed embedding space.
            images, labels = next(dataloader)

            # Convert the class labels into text embeddings.
            context = convert_labels_to_embeddings(labels, text_encoder)

            # Calculate the loss on the batch of training data.
            loss = diffusion_model.algorithm1_train_on_batch(images, context)

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
            progress_bar.set_description(f"loss: {loss.item():.4f}")

            # Update the current step.
            step += 1

            # To help visualize training, periodically sample from the
            # diffusion model to see how well its doing.
            if step != 0 and step % save_and_sample_every_n == 0:
                diffusion_model.eval()
                num_samples = 64
                with torch.inference_mode():
                    # Sample from the model to check the quality for
                    # unconditional sampling.
                    unconditional_context = torch.zeros(
                        (num_samples, context_dimension),
                        dtype=torch.float32,
                        device=device,
                    )
                    samples = diffusion_model.algorithm2_sampling(
                        image_size=images.shape[2],
                        num_channels=images.shape[1],
                        batch_size=num_samples,
                        context=unconditional_context,
                    )

                # Ssve the unconditional samples into an image grid
                utils.save_image(
                    samples,
                    str(f"{OUTPUT_NAME}/sample-{step}.png"),
                    nrow=int(math.sqrt(num_samples)),
                )

                # Generate a batch of conditional samples
                labels = torch.randint(
                    low=0, high=10, size=(num_samples,), device=device
                )
                conditional_embeddings, prompts = convert_labels_to_embeddings(
                    labels,
                    text_encoder,
                    return_prompts=True,
                )
                conditional_embeddings = conditional_embeddings.to(device)

                with torch.inference_mode():
                    # Sample from the model to check the quality
                    conditional_samples = diffusion_model.algorithm2_sampling(
                        image_size=images.shape[2],
                        num_channels=images.shape[1],
                        batch_size=num_samples,
                        context=conditional_embeddings,
                    )
                utils.save_image(
                    conditional_samples,
                    str(f"{OUTPUT_NAME}/conditional_sample-{step}.png"),
                    nrow=int(math.sqrt(num_samples)),
                )
                with open(f"{OUTPUT_NAME}/conditional_sample-{step}.txt", "w") as fp:
                    for i in range(num_samples):
                        if i != 0 and (i % math.sqrt(num_samples)) == 0:
                            fp.write("\n")
                        fp.write(f"{labels[i]} ")
                with open(
                    f"{OUTPUT_NAME}/conditional_sample_prompts-{step}.txt", "w"
                ) as fp:
                    for i in range(num_samples):
                        if i != 0 and (i % math.sqrt(num_samples)) == 0:
                            fp.write("\n")
                        fp.write(f"{prompts[i]} ")

                # Save a corresponding model checkpoint.
                torch.save(
                    {
                        "step": step,
                        "model_state_dict": diffusion_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                    },
                    f"{OUTPUT_NAME}/gaussian_diffusion_ddpm-{step}.pt",
                )

            # Update the training progress bar in the console.
            progress_bar.update(1)


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    run_lesson_5e(
        num_training_steps=args.num_training_steps, batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
