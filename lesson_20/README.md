# Lesson 20 - Elucidating the Design Space of Diffusion-Based Generative Models

In this lesson we are going to learn about an improved and unified diffusion formulation from [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364).

This paper is a fascinating look at ablating a lot of the design decisions from previous work, including DDPM++ from [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), NCSN++ from [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), ADM from [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233), and iDDPM from [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). Using the architectures from the above papers, the authors propose a new loss formulation, model preconditioning, augmentations, and sampling algorithm to improve upon the results of the previous authors.

The original source code for the paper was published at [EDM](https://github.com/NVlabs/edm/).

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

Run the training script at `train_mnist.py` like:

```
> python train_mnist.py --config_path configs/edm.yaml
```

Generated samples will be saved to `output/image/mnist/edm`.

## Results

After training the network for 20k steps, the full unconditional EDM is able to generate samples like the below, using only **18** steps:

![EDM](https://drive.google.com/uc?export=view&id=1yUeR5ep9mK1IwMsTyHwhyAqlftFwBNYz)

