# Lesson 12 - Cascaded Diffusion Models for High Fidelity Image Generation

In this lesson we are going to learn about using incorporating super-resolution models into a cascade of diffusion models, called [Cascaded Diffusion Models for High Fidelity Image Generation](https://arxiv.org/abs/2106.15282). 

The basic idea here is to use a low resolution diffusion model as the first stage, and then subsequently apply a bunch of super-resolution models to progressively higher resolutions. In this lesson, we will create an unconditional base diffusion model at 8x8 resolution, and then utilize a single cascade of super-resolution models at 32x32 resolution to generate the final output. We train the base model and the first stage cascade in parallel, and demonstrate sampling from the final cascaded model.

Unfortunately the authors never released their source code, so we have no pointers for that.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

To train the diffusion model, simply run the `train_mnist.py` script under this directory in your virtual environment:

```
> python train_mnist.py
```

Output files (including sample generated images of the model in progress) are stored by timestep in the `output` directory.

## Results

After training the network for 15k steps, the model is able to generate samples like the below:

| High Resolution Sample | Base Stage Output 
| ----- | ----- 
| ![CDM](https://drive.google.com/uc?export=view&id=13Ii508SuNxIyVD8tirihxOe87mQhrgYn) | ![CDM Base Stage](https://drive.google.com/uc?export=view&id=1D8VTcd0jVxVMZAApzLE6LscQu_VGXPmX) 
