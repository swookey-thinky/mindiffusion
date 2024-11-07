# Lesson 35 - AuraFlow

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to build the AuraFlow diffusion model from Fal.ai. There is no technical report yet, but they released a blog post at [AuraFlow](https://blog.fal.ai/auraflow/) which has some technical details. Unfortunately there is no training code yet, and the only inference code released is inside the diffusers repository.

## Introduction 

AuraFlow is a fun model combining a few different recent research results. It was released prior to Flux being released, and uses a similar Double and Single DiT stream block architecture, to a much lower extent that Flux though. The released AuraFlow model uses 4 MMDiT blocks followed by 32 DiT blocks, where Flux uses a 1:2 ratio of MMDiT to DiT. Similarly, AuraFlow uses a logit-normal schedule rectified flow training target, and they utilized "wider" transformers, at a ratio of hidden_dim / num_layers = 85 (however they found ratios 20 < R < 100 worked well). For training, they recaptioned all instances, and their blog details the recaptioning strategy. Interestingly for conditioning they appear to only use a T5 text embedder, so it will be interesting to see how the text adherence compares to Flux and others. They also used the maximal update parameterization from [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) ([github](https://github.com/microsoft/mup)) to find the optimal set of training hyperparameters.

In our implementation here, we train in pixel space rather than the latent space of the original model, and we scale down the transformer network to ~300m parameters from the original models 6.8b parameters. We did not perform hyperparameter transfer with maximal update parameterization. We successfully trained this model on a T4 instance with a batch size of 64.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Configuration File

The configuration file is located in [AuraFlow](https://github.com/swookey-thinky/mindiffusion/blob/main/lesson_35/configs/auraflow.yaml).

## Training

To train the auraflow model, use:

```
> python train_mnist.py --config_path configs/auraflow.yaml
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 32.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/mindiffusion/blob/main/lesson_35/configs/auraflow.yaml) | [google drive](https://drive.google.com/file/d/1p7LXrEVD3tTH5RE4Dj3hNxx44Cr4ob8I/view?usp=sharing) | ![AuraFlow](https://drive.google.com/uc?export=view&id=1_qODQDHpEZrverYHqa6_uH0QnLy7Tf4f)


After training the network for 10k steps, the auraflow model is able to generate samples like the below:

![AuraFlow](https://drive.google.com/uc?export=view&id=1_qODQDHpEZrverYHqa6_uH0QnLy7Tf4f)

The prompts we used for generation above were:

<pre>
four one two five 4 nine seven seven 
3 zero 2 5 8 nine two 1 
8 nine 5 0 four 7 0 4 
2 two one zero 1 three 2 5 
9 nine two two 3 two six four 
zero 8 8 9 nine one nine one 
2 nine 8 one nine eight eight three 
2 1 three 6 zero 3 nine 4 
</pre>

It hasn't quite learned the text adherence yet at 10k steps, but its clear that another 5k steps will improve this dramatically, in line with other models.
