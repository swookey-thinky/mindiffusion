# Lesson 7 - Score-Based Generative Modeling through Stochastic Differential Equations

In this lesson, we are going to look at a generalization to the theory of all prior lessons, [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456). The authors will even derive the results for both [NSCN](https://arxiv.org/abs/1907.05600) and [DDPM](https://arxiv.org/abs/2006.11239) from this new unified theory, which is pretty cool!

The author's approach this problem as the solution to a generalized stochastic differential equation (SDE), and in particular, the reverse time solution to that same SDE. They propose several different SDE's for the diffusion process, as well as new samplers based on the long history of SDE solvers in other domains.

You can find the original FLAX source code from the authors is [here](https://github.com/yang-song/score_sde/tree/main). They also have a PyTorch implementation [here](https://github.com/yang-song/score_sde_pytorch).

In this lesson, we will be re-implementing DDPM according to the Score-SDE paper, to show how it fits into the authors expanded theoretical framework.

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

Here are the results of sampling the model after 10k steps:

![Score SDE](https://drive.google.com/uc?export=view&id=1MeQidUiVZWQJKhWlibxWvrjbijQx4ORc)
