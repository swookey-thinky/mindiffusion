# Lesson 5a - Denoising Diffusion Probabilistic Models - Latent Interpolation

This lesson explores a cool feature of the DDPM, which is interpolating between latents to generate intermediate representations of the samples. The reverse process produces high-quality reconstructions, and plausible interpolations that smoothly
vary attributes from the source samples.

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

Here is an example of interpolating between samples x1 and x2 in latent space, starting at fixed timestep 250.

![Interpolation](https://drive.google.com/uc?export=view&id=1vyNRZ06mTy3-_jqzbevm_Iz0fnE7PUND)
