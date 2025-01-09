# Stable Diffusion 3.5 - Scaling Rectified Flow Transformers for High-Resolution Image Synthesis - Part 2

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn about the Stable Diffusion 3.5 diffusion model from Stability AI. This model was introduced in a blog post [Introducing Stable Diffusion 3.5](https://stability.ai/news/introducing-stable-diffusion-3-5)


## Introduction

Stable Diffusion 3.5 is an extension to the Stable Diffusion 3 model from [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206). In particular, it contains a number of architectural and training changes, especially in the Medium size model, in order to improve semantic and artistic fidelity. From the [model card](), the main changes in 3.5 over 3 include:

- MMDiT-X: Introduces self-attention modules in the first 13 layers of the transformer, enhancing multi-resolution generation and overall image coherence.

- QK Normalization: Implements the QK normalization technique to improve training Stability.

- Mixed-Resolution Training:
    - Progressive training stages: 256 → 512 → 768 → 1024 → 1440 resolution
    - The final stage included mixed-scale image training to boost multi-resolution generation performance
    - Extended positional embedding space to 384x384 (latent) at lower resolution stages
    - Employed random crop augmentation on positional embeddings to enhance transformer layer robustness across the entire range of mixed resolutions and aspect ratios. For example, given a 64x64 latent image, we add a randomly cropped 64x64 embedding from the 192x192 embedding space during training as the input to the x stream.

Note for our training, we incorporated both the MMDiT-X and QK Normalization architectural changes. We did not implement the mixed resolution training however, since we are already training at such a low resolution. And since we did not implement the mixed precision training, we did not implement the largest positional embedding space, since the random cropping strategy would yield random positional embeddings in the smaller space without the larger resolutions.

In our implementation here, we train in pixel space rather than the latent space of the original model, and we scale down the transformer network to ~70m parameters, rather than the ~2b parameters of the Medium model. Note we are using the architectural changes introduced in the Medium model, namely the improved MMDitX blocks with the additional self-attention channels.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements) to set up your environment.

## Configuration File

The configuration file is located in [SD3.5](https://github.com/swookey-thinky/mindiffusion/blob/main/lesson_38/config/sd3.5.yaml).

## Training

To train the SD3.5 model, use:

```
> python train.py --config_path config/sd3.5.yaml
```

We successfully tested training on a single A10 instance (24GB VRAM) using a batch size of 64. It took around 3hrs20 at a total cost of $2.50.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/mindiffusion/blob/main/lesson_38/config/sd3.5.yaml) | [Google Drive](https://drive.google.com/file/d/1bf4Jdk0OUt63XB2-5fzBdwQOzcMtkyYe/view?usp=sharing) | ![SD3.5](https://drive.google.com/uc?export=view&id=1_6GKNeTazoZ2RqEyxN2ta8B9UAdBPffB)

After training the network for 20k steps at batch size 64, the sd3.5 model pipeline is able to generate samples like the below:

![SD3.5](https://drive.google.com/uc?export=view&id=1_6GKNeTazoZ2RqEyxN2ta8B9UAdBPffB)

The prompts we used for generation above were:

<pre>
1 five 3 6 eight 0 3 4 
0 four seven nine 3 nine four 9 
three zero one 0 three two 6 3 
5 1 six three six six 1 two 
3 two 1 9 nine 8 zero zero 
6 six 3 eight 5 6 seven six 
4 three five one one 7 three 1 
one zero 3 two one four 7 six
</pre>
