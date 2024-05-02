# Zero-to-Hero - Diffusion Models
![Zero to Hero](https://drive.google.com/uc?export=view&id=1FfH6643JCYjnCq2JKFPs9wH64W_Rm_B2)

Repository of lessons exploring image diffusion models, focused on understanding and education.

## Introduction

This series is heavily inspired by Andrej Karpathy's [Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series of videos. Well, actually, we are straight out copying that series, because they are so good. Seriously, if you haven't followed his videos, go do that now - lot's of great stuff in there!

Each lesson contains both an explanatory video which walks you through the lesson and the code, a colab notebook that corresponds to the video material, and a a pointer to the runnable code in github. All of the code is designed to run on a minimal GPU. We test everything on T4 instances, since that is what colab provides at the free tier, and they are cheap to run on AWS as stand alone instances. Theoretically each of the lessons should be runnable on any 8GB or greater GPU, as they are all designed to be trained in real time on minimal hardware, so that we can really dive into the code.

Each lesson is in its own subdirectory, and we have ordered the lessons in historical order (from oldest to latest) so that its easy to trace the development of the research and see the historical progress of this space.

Since every lesson is meant to be trained in real time with minimal cost, most of the lessons are restricted to training on the MNIST dataset, simply because it is quick to train and easy to visualize.

## Requirements for All Lessons

All lessons are built using PyTorch and written in Python 3. To setup an environment to run all of the lessons, we suggest using conda or venv:

```
> python3 -m venv mindiffusion_env
> source mindiffusion_env/bin/activate
> pip install -upgrade pip
> pip install -r requirements.txt
```

All lessons are designed to be run *in the lesson directory*, not the root of the repository.

## Table of Lessons


Lesson | Date | Name |Title
:---- | :---- | :---- | ----
1 |  | | Introduction to Diffusion Models
2 | March 2015 | DPM | [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)
3 | July 2019 | NCSN | [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)
4 | June 2020 | NCSNv2 | [Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011)
5 | June 2020 | DDPM | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
6 | October 2020 | DDIM | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
7 | November 2020 | Score SDE | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
8 | February 2021 | DaLL-E | [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)
9 | December 2021 | Stable Diffusion | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
10 | April 2022 | DaLL-E 2| [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125)
11 | May 2022 | Imagen | [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)
12 | October 2022 | | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
13 | October 2022 | ERNIE-ViLG 2.0 | [ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts](https://arxiv.org/abs/2210.15257)
14 | December 2022 | DiT | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
15 | May 2023 | RAPHAEL | [RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths](https://arxiv.org/abs/2305.18295)
16 | June 2023 | Wuerstchen | [Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models](https://arxiv.org/abs/2306.00637)
17 | July 2023 | SDXL | [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)
18 | September 2023 | PixArt-α | [PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426)
19 | October 2023 | DaLL-E 3 | [Improving Image Generation with Better Captions](https://cdn.openai.com/papers/dall-e-3.pdf)
20 | January 2024 | PIXART-δ | [PIXART-δ: Fast and Controllable Image Generation with Latent Consistency Models](https://arxiv.org/abs/2401.05252)
21 | March 2024 | Stable Diffusion 3 | [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)
