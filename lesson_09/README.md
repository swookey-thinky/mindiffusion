# Lesson 9 - Improved Denoising Diffusion Probabilistic Models

In this lesson we are going to dive into an improvement to the [DDPM](https://arxiv.org/abs/2006.11239) model we learned about in [Lesson 5](https://github.com/swookey-thinky/mindiffusion/lesson_05), [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). This paper introduced a number of improvements that have become standard techniques in later models, so it's interesting to see their origin and see how they were applied to the base model. In particular, some of the improvements introduced here include:

1. A new parameterization called *v-param* which produces an estimate of the variance at each noise step.
2. A cosine rather than linear beta schedule.
3. The use of importance sampling during training to reduce the gradient noise and yields better log-likelihoods.
4. A different timestep embedding at the group normalization stage (scale and shift). 

You can find the authors original codebase [here](https://github.com/openai/improved-diffusion/tree/main).

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

After training the network for 10k steps, the model is able to generate samples like:

![IDDPM](https://drive.google.com/uc?export=view&id=1vHDkYDEpaIReqedT9GFJprZiiVliuX81)
