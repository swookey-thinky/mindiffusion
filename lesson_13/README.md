# Lesson 13 - Variational Diffusion Models

In this lesson we are going to learn about the variational diffusion models from [Variational Diffusion Models](https://arxiv.org/abs/2107.00630).

The most important innovation of this paper is a new, simpler continuous time formulation of the diffusion process.

The authors released their original JAX code at [VDM](https://github.com/google-research/vdm) so you can double check your work with their work.


In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

To train the diffusion model, simply run the `train.py` script under this directory in your virtual environment:

```
> python train.py
```

Output files (including sample generated images of the model in progress) are stored by timestep in the `output` directory.

## Results

After training the network for 10k steps, the model is able to generate samples like the below :

![VDM](https://drive.google.com/uc?export=view&id=1FRaNRkEyg0xMvn0dH6MW5PcXCj3BTj-k)
