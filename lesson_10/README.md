# Lesson 10 - Image Super-Resolution via Iterative Refinement

In this lesson we are going to learn about using diffusion models to perform super-resolution from the paper [Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636). 

Why do we care about super-resolution? This technique has become a cornerstone of modern diffusion model pipelines, especially in the context of "cascaded" diffusion models, which are introduced in this paper but refined in a later paper and lesson that we will explore. The idea is that you can synthesize a lower resolution image to get the semantic details of the scene correct, and then use a super-resolution model to fill in the high level details. The cool part about using a diffusion model to do this is that it really requires minimal changes to the pipeline and the model itself in order to work! In fact, we reuse most of the previous lessons, and the only real change to the diffusion model is an additional channel of input - the low resolution image that we want to upsample.

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

After training the network for 18k steps, the model is able to generate samples like the below:

https://drive.google.com/file/d/1CaRU1TXoZQEHz7xiHqguMYQqNoURU5Yn/view?usp=sharing

| High Resolution Target | Low Resolution Context | SR3 Upsampled
| ----- | ----- | -----
| ![HR](https://drive.google.com/uc?export=view&id=1_kh2eBgzMNOf1GdeJ-jpRr-6baUrgX2t) | ![LR](https://drive.google.com/uc?export=view&id=1CaRU1TXoZQEHz7xiHqguMYQqNoURU5Yn) | ![SR3](https://drive.google.com/uc?export=view&id=1EkcuWrND6oT86n1khKIsZJb3L6KmcgMo)
