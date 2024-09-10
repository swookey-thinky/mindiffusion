# Lesson 26 - Consistency Models

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn new class of generative models (in addition to a new distillation technique) called Consistency Models from [Consistency Models](https://arxiv.org/abs/2303.01469).

Consistency models are both a distillation technique as well as a new class of generative models. The concept is fairly simple. Using the SDE/ODE formulation of diffusion models, whereby any SDE can be interpreted as a probability flow ODE, the consistency model formulation enforces the *consistency* property during training/distillation, which states that all points along the same probability flow ODE trajectory map to the same initial point. Given this stipulation, this naturally leads to a **one-step** sampling method with these models, which is incredibly cool!

In this repository, we demonstrate training a consistency model from scratch, although in the code you will also see the loss function for distillation, and you can distill a trained model using consistency distillation over at the  [xdiffusion](https://github.com/swookey-thinky/xdiffusion/) repository..

The original source code for the paper was published at [Consistency Models](https://github.com/openai/consistency_models/).

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

PixArt-α is a three stage training process, and we implement the first two stages here. First, we train a class conditional diffusion model without text captions, to create a strong pixel generation baseline.

Run the training script at `train_mnist.py` like:

```
> python train_mnist.py --config_path "configs/consistency_model.yaml" --num_training_steps 100000 --batch_size 64
```

Generated samples will be saved to `output/consistency_model`. On a T4 instance, we trained the model using a batch size of 64. Also, the model requires a much larger number of steps compared to a DDPM based model.


## Results

After training the network for 100k steps, the model is able to generate examples like the below. Note that we give results or different numbers of sampling steps:

| Config | Checkpoint | Num Sampling Steps | Results
| ------ | ---------- | ------- | -------
| [Consistency Model](configs/consistency_model.yaml) | [Google Drive](https://drive.google.com/file/d/1iT2RxA7yJs2udO2qQDv8fkSe5JcwARTn/view?usp=sharing) | 1 | ![1](https://drive.google.com/uc?export=view&id=12hMpGtyLrfTy0BdJMSQ4GmPdyFd4ceE4)
| [Consistency Model](configs/consistency_model.yaml) | [Google Drive](https://drive.google.com/file/d/1iT2RxA7yJs2udO2qQDv8fkSe5JcwARTn/view?usp=sharing) | 3 | ![3](https://drive.google.com/uc?export=view&id=12wVUP7Gid2-mzHj0gAgPpuVgsOvIQahU)
| [Consistency Model](configs/consistency_model.yaml) | [Google Drive](https://drive.google.com/file/d/1iT2RxA7yJs2udO2qQDv8fkSe5JcwARTn/view?usp=sharing) | 40 | ![40](https://drive.google.com/uc?export=view&id=1Zgj38dDdEwGvHKFMJ0zgR37zrA5fQ-vx)
