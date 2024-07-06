# Lesson 14 - Classifier Free Guidance

In this lesson we are going to learn about an alternative to classifier guidance (which we learned about in [Lesson 11 - Guided Diffusion](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_11)) called [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). Classifier-Free Guidance is a technique used in later models like Stable Diffusion to trade-off image diversity with image fidelity. In this lesson, we use it in a class-conditional manner, as was done in the paper. But the technique applies equally for other conditioning mechanisms, like text-to-image.

Unfortunately the authors never released their source code, so we have no pointers for that. Phil Wang has an implementation of it [here](https://github.com/lucidrains/classifier-free-guidance-pytorch) and while we did not use it for this lesson, his respositories are well written and follow the papers decently so you can look there for additional examples.

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

After training the network for 10k steps, the model is able to generate samples like the below with a guidance scale of 4.0:

<!-- ![CFG](https://drive.google.com/uc?export=view&id=1_kh2eBgzMNOf1GdeJ-jpRr-6baUrgX2t) -->

