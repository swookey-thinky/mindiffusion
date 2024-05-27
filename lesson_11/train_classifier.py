"""Lesson 10 - Guided Diffusion Classifier

Training script for training a classifier for MNIST from
"Diffusion Models Beat GANs on Image Synthesis" (https://arxiv.org/abs/2105.05233).

To run this script, install all of the necessary requirements
and run:

```
python train_classifier.py
```
"""

from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from classifier import GuidedDiffusionClassifier
from diffusion_model import GaussianDiffusion_GuidedDiffusion
from score_network import MNistUnet
from utils import cycle, load_yaml

OUTPUT_NAME = "output/classifier"


def run_lesson_10_classifier(
    num_training_steps: int, batch_size: int, config_path: str
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

    evaluation_dataset = MNIST(
        ".",
        train=False,
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
    evaluation_dataloader = DataLoader(
        evaluation_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Open the model configuration
    config = load_yaml(config_path)

    # Create the diffusion model, and leave it on the CPU.
    # We are using this *only* to noise the input samples for
    # the classifier.
    diffusion_model = GaussianDiffusion_GuidedDiffusion(
        score_network_type=MNistUnet, config=config
    )

    # Create the classifier we are going to train, with a UNet
    # specifically for the MNIST dataset.
    classifier = GuidedDiffusionClassifier(config=config)
    summary(classifier, [(128, 1, 32, 32), (128,)])

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(split_batches=False),
        mixed_precision="no",
    )
    device = accelerator.device

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader, evaluation_dataloader = accelerator.prepare(
        dataloader, evaluation_dataloader
    )

    # We are going to train for a fixed number of steps, so set the dataloader
    # to repeat indefinitely over the entire dataset.
    dataloader = cycle(dataloader)

    # Now create the optimizer. The optimizer choice and parameters come from
    # the DDPM paper:
    # "We tried Adam [31] and RMSProp early on in our experimentation process and chose the
    #  former. We left the hyperparameters to their standard values. We set the learning
    #  rate to 2 × 10−4 without any sweeping, and we lowered it to 2 × 10−5
    #  for the 256 × 256 images, which seemed unstable to train with the larger learning rate."
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_training_steps // 100, eta_min=1e-6
    )

    # Move the model and the optimizer to the accelerator as well.
    classifier, optimizer, scheduler = accelerator.prepare(
        classifier, optimizer, scheduler
    )

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

    with tqdm(initial=step, total=num_training_steps) as progress_bar:
        # Perform gradient descent for the given number of training steps.
        while step < num_training_steps:
            # The dataset has images and classes, but we don't need the class
            images, labels = next(dataloader)
            images = images.to(device)
            labels = labels.to(device)

            # Noise the input images for the classifier
            with torch.no_grad():
                t, _ = diffusion_model._importance_sampler.sample(
                    images.shape[0], device=device
                )

                # The diffusion model is on the CPU.
                noisy_images = diffusion_model._q_sample(images.to("cpu"), t.to("cpu"))

                # Move back to the GPU for training.
                noisy_images = noisy_images.to(device)

            # Calculate the loss on the batch of training data.
            logits = classifier(noisy_images, t)
            loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            loss = loss.mean()

            # Calculate the gradients at each step in the network.
            accelerator.backward(loss)

            # On a multi-gpu machine or cluster, wait for all of the workers
            # to finish.
            accelerator.wait_for_everyone()

            # Clip the gradients.
            accelerator.clip_grad_norm_(classifier.parameters(), max_grad_norm)

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
                evaluate(
                    classifier, step, diffusion_model, evaluation_dataloader, device
                )
                save(classifier, step, loss, optimizer)
                average_loss = average_loss_cumulative / float(save_and_sample_every_n)
                average_loss_cumulative = 0.0
                scheduler.step()

            # Update the current step.
            step += 1

            # Update the training progress bar in the console.
            progress_bar.update(1)

    # Save the final results
    evaluate(classifier, step, diffusion_model, evaluation_dataloader, device)
    save(classifier, step, loss, optimizer)


@torch.inference_mode()
def evaluate(classifier, step, diffusion_model, data_loader, device):
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_entries = 0

    classifier.eval()
    for images, labels in tqdm(data_loader, total=len(data_loader), leave=False):
        # Noise the input images for the classifier
        with torch.no_grad():
            t, _ = diffusion_model._importance_sampler.sample(
                images.shape[0], device=device
            )
            noisy_images = diffusion_model._q_sample(images.to("cpu"), t.to("cpu"))

            # Move back to the GPU for training.
            noisy_images = noisy_images.to(device)

        # Calculate the loss on the batch of training data.
        logits = classifier(noisy_images, t)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        total_loss += loss.mean()
        total_batches += 1

        # Softmax the logits, and calculate the index of the most likely
        # element.
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        total_correct += (predictions == labels).float().sum()
        total_entries += labels.shape[0]

    classifier.train()
    average_loss = total_loss / total_batches
    accuracy = total_correct / total_entries
    print(
        f"Evaluation at step {step}: average loss={average_loss:.4f} accuracy: {accuracy}"
    )


def save(diffusion_model, step, loss, optimizer):
    # Save a corresponding model checkpoint.
    torch.save(
        {
            "step": step,
            "model_state_dict": diffusion_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        f"{OUTPUT_NAME}/guided_diffusion_classifier-{step}.pt",
    )


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_training_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--config", type=str, default="configs/mnist_v_param.yaml")
    args = parser.parse_args()

    run_lesson_10_classifier(
        num_training_steps=args.num_training_steps,
        batch_size=args.batch_size,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
