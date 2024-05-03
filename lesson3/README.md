# Lesson 3 - Noise Conditioned Score Networks

In this lesson we are going to implement the diffusion model presented in the 2019 paper [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) which introduced Noise Conditioned Score Networks (NCSN).

If you want to look at the author's original codebase, you can find it [here](https://github.com/ermongroup/ncsn).

This model is interesting in that it presents an alternative, arguably simpler, theoretical framework for diffusion models, allowing us to use an improved timestep conditioning and arbitrary model architectures. It starts to present a  unified theoretical understanding around stochastic differential equations, although it won't be until later that this is fully fleshed out.

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

After training the network for 30k steps, the model is able to generate samples like:

![NCSN](https://drive.google.com/uc?export=view&id=17BD8rurt5CL_NPIYjnkgSyhAdYg_eLqV)
