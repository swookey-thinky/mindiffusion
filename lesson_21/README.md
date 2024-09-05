# Lesson 20 - Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

In this lesson we are going to learn about a rectified flow diffusion model from [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003).

Rectified flows are an alternative ODE formulation that is simple and yet yields surprisingly good results. For diffusion models, the code is formulated based on the Score SDE formulation from [Score-Based Generative Modeling through Stochastic Differential Equations] (https://arxiv.org/abs/2011.13456). For our purposes, this is an exciting paper to study because it is used as part of the innovation in the Stable Diffusion 3.0 model from the paper [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

The original source code for the paper was published at [RectifiedFlow](https://github.com/gnobitab/RectifiedFlow).

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

Run the training script at `train_mnist.py` like:

```
> python train_mnist.py --config_path configs/rectified_flow_32x32.yaml
```

Generated samples will be saved to `output/mnist/rectified_flow_32x32`.

## Results

After training the network for 10k steps, the full unconditional rectified flow model is able to generate samples like the below:

![Rectified Flow](https://drive.google.com/uc?export=view&id=14TOqFXSWiFpeUVnDuMfLRcRDUV5onuKQ)

