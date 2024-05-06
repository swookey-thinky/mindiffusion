"""Lesson 5b - Denoising Diffusion Probabilistic Models - Latent Interpolation.

This lesson demonstrates the latent interpolation capabilities of DDPM
models, as demonstrated in the paper.
"""
from accelerate import Accelerator, DataLoaderConfiguration
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm

import os
from pathlib import Path
import sys

# HACK: Reuse the packages from the parent directory. It will try to import
#       packages in this directory first, then move to the packages in the
#       parent directory.
sys.path.append(str(Path(os.path.dirname(__file__)).parent.absolute()))

from diffusion_model import GaussianDiffusion_DDPM
from score_network import MNistUNet
from utils import (
    extract,
    linear_beta_schedule,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)

OUTPUT_NAME = "output"


def run_lesson_5b(model_checkpoint: str, num_examples: int):
    # Fix the random seed for reproducibility.
    torch.manual_seed(42)

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

    # Create the dataloader for the MNIST dataset.
    dataloader = DataLoader(
        dataset, batch_size=num_examples, shuffle=False, num_workers=4
    )

    # We will load the diffusion model from lesson 5 and interpolate
    # the latents generated from it.
    diffusion_model = GaussianDiffusion_DDPM(unet_type=MNistUNet)

    # Load the model checkpoint
    checkpoint = torch.load(model_checkpoint)
    diffusion_model.load_state_dict(checkpoint["model_state_dict"])

    # The accelerate library will handle of the GPU device management for us.
    accelerator = Accelerator(
        DataLoaderConfiguration(split_batches=False), mixed_precision="no"
    )
    device = accelerator.device

    # Prepare the dataset with the accelerator. This makes sure all of the
    # dataset items are placed onto the correct device.
    dataloader = accelerator.prepare(dataloader)

    # Move the model to the accelerator as well.
    diffusion_model = accelerator.prepare(diffusion_model)

    # Grab two batches of data. We will interpolate the latents
    # between the two batches of data.
    dataloader_iter = iter(dataloader)
    x1 = next(dataloader_iter)[0].to(device)
    x2 = next(dataloader_iter)[0].to(device)

    save_image(x1, str(f"{OUTPUT_NAME}/x1_source.png"))
    save_image(x2, str(f"{OUTPUT_NAME}/x2_source.png"))

    # Fully noise both x1 and x2 to create the "latent" representation
    # of each one.
    num_timesteps = 1000

    # The paper demonstrated starting the interpolation with the latents
    # from a given timestep.
    interpolation_timesteps = 250
    t = torch.full(
        (num_examples,), interpolation_timesteps, device=device, dtype=torch.long
    )

    x1_normalized = normalize_to_neg_one_to_one(x1)
    x2_normalized = normalize_to_neg_one_to_one(x2)

    # Noise the samples to the given timestep.
    noise = torch.randn_like(x1_normalized)
    x1_latents = q_sample(x1_normalized, t, num_timesteps, device, noise)
    x2_latents = q_sample(x2_normalized, t, num_timesteps, device, noise)
    save_image(
        unnormalize_to_zero_to_one(x1_latents), str(f"{OUTPUT_NAME}/x1_latents.png")
    )
    save_image(
        unnormalize_to_zero_to_one(x2_latents), str(f"{OUTPUT_NAME}/x2_latents.png")
    )

    # Fix the noise for each reconstruction so that it matches the corresponding
    # lambda parameter.
    torch.manual_seed(42)
    x1_reconstruction = p_sample_loop(
        diffusion_model, x1_latents, num_timesteps, interpolation_timesteps, device
    )
    save_image(x1_reconstruction, str(f"{OUTPUT_NAME}/x1_reconstruction.png"))

    # Fix the noise for each reconstruction so that it matches the corresponding
    # lambda.
    torch.manual_seed(42)
    x2_reconstruction = p_sample_loop(
        diffusion_model, x2_latents, num_timesteps, interpolation_timesteps, device
    )
    save_image(x2_reconstruction, str(f"{OUTPUT_NAME}/x2_reconstruction.png"))

    num_lambdas = 11
    lambdas = torch.linspace(0.0, 1.0, num_lambdas, device=device)
    outputs = []
    for i in tqdm(range(num_lambdas)):
        # Fix the noise for each lambda. From the paper,
        # "We fixed the noise for different values of λ so xt and xt'
        # remain the same."
        torch.manual_seed(42)

        # Interpolate the latents
        x_interpolated = x1_latents * (1 - lambdas[i]) + lambdas[i] * x2_latents

        # Sample starting with the interpolated latents
        x_interpolated_denoised = p_sample_loop(
            diffusion_model,
            x_interpolated,
            num_timesteps,
            interpolation_timesteps,
            device,
        )

        # Save the intermediates
        save_image(
            x_interpolated_denoised, str(f"{OUTPUT_NAME}/lambda-{lambdas[i]:.2f}.png")
        )
        outputs.append(x_interpolated_denoised)

    # Now save the outputs so we can visualize them.
    # We have a list of (num_examples, 1, 32, 32) images,
    # so save them in a grid. Conver the list into a batch dimension
    # size (num_examples * num_lambdas, ...)
    samples = torch.cat([x1] + outputs + [x2], dim=0)
    assert samples.shape[0] == num_examples * (num_lambdas + 2)

    save_image(
        samples,
        str(f"{OUTPUT_NAME}/interpolation_t_{interpolation_timesteps}.png"),
        nrow=num_examples,
    )


@torch.inference_mode()
def q_sample(x_0, t, num_timesteps, device, noise=None):
    """Forward process for DDPM.

    Noise the initial sample x_0 to the timestep t.
    """
    betas = linear_beta_schedule(num_timesteps).to(torch.float32).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    if noise is None:
        noise = torch.randn_like(x_0)

    return (
        extract(sqrt_alphas_cumprod, t, x_0.shape) * x_0
        + extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
    )


@torch.inference_mode()
def p_sample_loop(
    diffusion_model, x_t, total_timesteps, interpolation_timesteps, device
):
    """Reverse process for DDPM.

    Denoise the sample x_t for the number of timesteps.
    """
    batch_size = x_t.shape[0]

    # Line 4, constants that we will use below.
    # Fixed linear schedule for Beta
    betas = linear_beta_schedule(total_timesteps).to(torch.float32).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alpha = torch.sqrt(1.0 / alphas)

    # Line 2, repeat from T=num_timesteps (fully noised image) to
    # t=0 (fully unnoised image).
    for t in tqdm(
        reversed(range(0, interpolation_timesteps)),
        desc="sampling loop time step",
        total=interpolation_timesteps,
    ):
        # Add a batch dimension to the integer time (all items
        # in the batch are sampled at the same time).
        batched_times = torch.full((batch_size,), t, device=device, dtype=torch.long)

        # Line 3, if t > 1 (1-indexed) then sample from a Gaussian,
        # else t is zero.
        z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

        # Line 4, lets start calculating all of the values we need. First
        # we need to calulate epsilon_theta, at time t.
        epsilon_theta = diffusion_model._unet(x_t, batched_times)

        # Line 4, calculate the constant in front of epsilon_theta, which
        # corresponds to Beta / sqrt (1 - cumprod(alpha))
        epsilon_theta_coeff = (
            sqrt_recip_alpha * (1.0 - alphas) / sqrt_one_minus_alphas_cumprod
        )
        # Grab only the timesteps in the batch.
        epsilon_theta_coeff = extract(
            epsilon_theta_coeff, batched_times, epsilon_theta.shape
        )

        # Line 4, calculate the coefficent for x_t
        x_t_coeff = sqrt_recip_alpha
        # Grab only the values for the timesteps in the batch.
        x_t_coeff = extract(x_t_coeff, batched_times, x_t.shape)

        # Line 4, calculate the z coefficient. We use the 'fixedlarge'
        # calculation for the variance here, which corresponds to
        # beta.
        z_coeff_squared = betas
        # Grab only the values for the timesteps in the batch.
        z_coeff_squared = extract(z_coeff_squared, batched_times, z.shape)

        # Line 4, put it all together.
        x_t_minus_1 = (
            x_t_coeff * x_t
            - epsilon_theta_coeff * epsilon_theta
            + torch.sqrt(z_coeff_squared) * z
        )

        # Loop to the next timestep, setting x(t-1) to the
        # current time
        x_t = x_t_minus_1
    # Unnormalize the sample images from (-1, 1) back to (0,1)
    return unnormalize_to_zero_to_one(x_t)


def main(override=None):
    """
    Main entrypoint for the standalone version of this package.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True)
    parser.add_argument("--num_examples", type=int, default=8)
    args = parser.parse_args()

    run_lesson_5b(
        model_checkpoint=args.model_checkpoint, num_examples=args.num_examples
    )


if __name__ == "__main__":
    main()
