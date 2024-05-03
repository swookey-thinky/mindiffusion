# Lesson 4 - Noise Conditioned Score Networks v2

In this lesson we are going to implement the diffusion model presented in the 2020 paper [Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011) which was a follow on to the authors 2019 paper [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600). This paper introduces Noise Conditioned Score Networks v2 (NCSNv2) with several improved techniques to get better image quality at higher resolutions.

If you want to look at the author's original codebase, you can find it [here](https://github.com/ermongroup/ncsnv2).

This paper is ostensibly an incremental improvement to the Noise Conditioned Score Networks we learned about in Lesson 3. It doesn't change the theory at all, but provides some more formal estimates of hyperparameters and some advanced tuning techniques. The biggest takeways are twofold. First, the method of conditioning the score network on the variance in the original paper was unnecessary, and a much simpler improvement is introduced here. Second, exponential moving averages was introduced to the sampling model, and we will see this used again and again in future work.

In summary, this paper introduced 5 improved techniques for training the model from Lesson 3:

1. Choose a larger initial noise scale for better sampling diversity.
2. Choose a smarter set of noise scales $\{\sigma_i\}_{i=1}^L$ to match the larger initial noise scale.
3. Remove the noise conditioning from the score network $s_\theta(\textbf{x}, \sigma)$ and instead parameterize the model with the unconditional score network $s_\theta(\textbf{x}, \sigma) = s_\theta(\textbf{x}) / \sigma$
4. Choose T (the number of sampling steps for each noise scale) as large as computationally
   feasible and then solve for the optimal learning rate $\epsilon$
5. Use Exponential Moving Averages (EMA) when sampling

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

![NCSNv2](https://drive.google.com/uc?export=view&id=1bkSdBlvli5U2Lle9ELd2mXqZ6T0hxj6k)
