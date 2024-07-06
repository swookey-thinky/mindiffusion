# Lesson 17 - Progressive Distillation for Fast Sampling of Diffusion Models

In this lesson we are going to learn about distilling a diffusion model into a samller number of sampling steps, from the paper [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512).

This paper is very cool because you can take a pretrained diffusion model, that required N steps of sampling, and distill it into another, same sized diffusion model that only takes M steps of sampling, where M << N. For example, in the steps below, we take a diffusion model that originally required 1024 sampling steps and distill it into one that only requires 8 sampling steps, at a similar quality level as the original umber of steps!

Another innovation from this paper was a new parameterization of the score network, called the *v-parameterization*, which is used pretty extensively in most modern models.

The authors released their original JAX source code at [Diffusion Distillation](https://github.com/google-research/google-research/tree/master/diffusion_distillation) that you can use to follow along if you want.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

First, we need a pretrained diffusion model that we are going to distill. To train a standard DDPM, v-parameterized, continuous time formulation diffusion model, use:

```
> python train.py --config_path configs/ddpm_32x32_v_continuous.yaml
```

This will generate model checkpoints in the `output/ddpm_32x32_v_continuous` directory.

Next, we want to distill that model into a smaller number of steps. We will distill it for 7 iterations, taking it from 1024 sampling steps down to 8.

```
> python distill.py --config_path configs/ddpm_32x32_v_continuous.yaml --teacher_model_checkpoint output/ddpm_32x32_v_continuous/diffusion-10000.pt --initial_sampling_steps 512 --distillation_iterations 7
```

Generated samples will be saved to `output/distilled`.

## Results

After training the initial model for 10k steps, and each distillation iteration for 5k steps, the model generates the following output:

| Original Model (1024 steps) | Original Model (8 Steps) | Distilled Model (8 steps)
| ---- | ---- | ----
| ![Original 1024](https://drive.google.com/uc?export=view&id=1FRaNRkEyg0xMvn0dH6MW5PcXCj3BTj-k) | ![Original 8](https://drive.google.com/uc?export=view&id=1hLpFWceM_7ni5s1sdKSsms20W9uHXhog) | ![Distilled 8](https://drive.google.com/uc?export=view&id=1UVcvEptEg8D7pPp_2HWm3bizAajAvYIY)

