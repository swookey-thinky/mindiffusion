# Lesson 2 - Diffusion Probabilistic Models

In this lesson, we are going to implement the diffusion model presented in the 2015 paper [Deep Unsupervised Learning using Nonequilibrium Thermodynamics
](https://arxiv.org/abs/1503.03585), the groundbreaking research paper from Sohl-Dickstein and others. This paper was the first to introduce the concept of a **diffusion model** for use in generative imagery.

The original code for this paper can be found [here](https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/tree/master). The original repository is written in Theano, and given the age of the repository, and more importantly the age of its dependencies, can be challenging to get running. But fret not, we will refer to the code where necessary, and will reimplement everything we need in PyTorch to see how it works.

In this colab, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

To train the model, simply run the `train_mnist.py` script under this directory in your virtual environment:

```
> python train_mnist.py
```

Output files (including sample generated images of the model in progress) are stored by timestep in the `output` directory.