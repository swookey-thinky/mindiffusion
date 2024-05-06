# Lesson 5c - Denoising Diffusion Probabilistic Models - Basic Class Conditioning

This lesson explores using control mechanisms to conditionally generate samples of the class we want. So far, all of the diffusion models we have explored have been *unconditional* diffusion models - you have no control over what the final image looks like. However, most of the popular image generators you have used are *conditional* diffusion models - you can control what images are generated through a conditioning mechanism, typically a text caption to describe the output or a reference image to generate a similar image. In this lesson, we are going to add conditional control, specifically class conditioning, to the diffusion model, to demonstrate one approach to controlling the output of the image generation process. In later lessons, we will extend this conditioning to other modalities, until we have replicated the text-to-image functionality present in public image generation applications. 

In this lesson, we will begin to introduce advanced conditioning using cross attention, from the paper "Attention Is All You Need" (https://arxiv.org/abs/1706.03762). This lesson will use cross attention in a naive way, but inspired by the attention mechanism in "High-Resolution Image Synthesis with Latent Diffusion Models" (https://arxiv.org/abs/2112.10752), which introduced the world to *latent* diffusion models and Stable Diffusion, an implementation of latent diffusion models.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

To train the model, simply run the `train_mnist.py` script under this directory in your virtual environment:

```
> python train_mnist.py
```

Output files (including sample generated images of the model in progress) are stored by timestep in the `output` directory.

## Results

TODO!!!
