# Lesson 5 - Denoising Diffusion Probabilistic Models

In this lesson we going to dive into a seminal paper, [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). This paper really laid the foundation for all of the generative models to come, and was the first to bring together the theory from our previous lessons and give it a more modern and flexible framework. Importantly, this paper introduced the usage of sinusoidal position embeddings for conditioning on the timestep (based on modern transformers), and adding attention into the noise prediction network. 
If you want to look at the author's original codebase, you can find it [here](https://github.com/hojonathanho/diffusion). The original code is written in Tensorflow, but it easy to follow along with.

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

TODO!!!
![DDPM](https://drive.google.com/uc?export=view&id=1bkSdBlvli5U2Lle9ELd2mXqZ6T0hxj6k)
