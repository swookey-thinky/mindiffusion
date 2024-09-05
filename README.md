# Zero-to-Hero - Diffusion Models
![Zero to Hero](https://drive.google.com/uc?export=view&id=1FfH6643JCYjnCq2JKFPs9wH64W_Rm_B2)

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

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
> pip install --upgrade pip
> pip install -r requirements.txt
```

All lessons are designed to be run *in the lesson directory*, not the root of the repository.

## Table of Lessons


Lesson | Date | Name |Title|Video|Colab|Code
:---- | :---- | :---- | ---- | ---- | ----| ----
1 |  | | Introduction to Diffusion Models | | [colab](https://colab.research.google.com/drive/1BPSG4SA21T8yx8oYqtsx_cYBwNvBtrvt#scrollTo=wAZDhI2JSPnV) |
2 | March 2015 | DPM | [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) | | [colab](https://colab.research.google.com/drive/115GjoxpHCTi-MCBF4_j7Vqhf3xfyZeGp?usp=sharing) | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_02)
3 | July 2019 | NCSN | [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_03)
4 | June 2020 | NCSNv2 | [Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_04)
5 | June 2020 | DDPM | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_05)
5a |  |  | DDPM with Dropout | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_05/sublesson_a)
5b |  |  | Interpolation in Latent Space | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_05/sublesson_b)
5c |  |  | Adding Control - Basic Class Conditioning with Cross-Attention | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_05/sublesson_c)
5d |  |  | Adding Control - Extended Class Conditioning | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_05/sublesson_d)
5e |  |  | Adding Control - Text-to-Image | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_05/sublesson_e)
6 | October 2020 | DDIM | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_06)
7 | November 2020 | Score SDE | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_07)
8 | February 2021 | DaLL-E | [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_08)
9 | February 2021 | IDDPM | [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_09)
10 | April 2021 | SR3 | [Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_10)
11 | May 2021 | Guided Diffusion (ADM) | [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_11)
12 | May 2021 | CDM | [Cascaded Diffusion Models for High Fidelity Image Generation](https://arxiv.org/abs/2106.15282) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_12)
13 | July 2021 | VDM | [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_13)
14 | December 2021 | Latent Diffusion | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_14)
14a | | Stable Diffusion v1 | | | |
14b | | Stable Diffusion v2 | | | |
15 | December 2021 | CFG| [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_15)
16 | December 2021 | GLIDE| [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_16)
17 | February 2022 | | [Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_17)
18 | April 2022 | DaLL-E 2| [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_18)
19 | May 2022 | Imagen | [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_19)
20 | June 2022 | EDM | [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_20)
21 | September 2022 | | [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_21)
22 | October 2022 | ERNIE-ViLG 2.0 | [ERNIE-ViLG 2.0: Improving Text-to-Image Diffusion Model with Knowledge-Enhanced Mixture-of-Denoising-Experts](https://arxiv.org/abs/2210.15257) | | |
23 | December 2022 | DiT | [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_23)
24 | January 2023 | Simple Diffusion | [Simple diffusion: End-to-end diffusion for high resolution images](https://arxiv.org/abs/2301.11093) | | |
25 | February 2023 | ControlNet | [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) | | |
26 | May 2023 | RAPHAEL | [RAPHAEL: Text-to-Image Generation via Large Mixture of Diffusion Paths](https://arxiv.org/abs/2305.18295) | | |
27 | June 2023 | Wuerstchen | [Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models](https://arxiv.org/abs/2306.00637) | | |
28 | July 2023 | SDXL | [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) | | |
29 | September 2023 | PixArt-α | [PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_29)
30 | October 2023 | DaLL-E 3 | [Improving Image Generation with Better Captions](https://cdn.openai.com/papers/dall-e-3.pdf) | | |
31 | January 2024 | PIXART-δ | [PIXART-δ: Fast and Controllable Image Generation with Latent Consistency Models](https://arxiv.org/abs/2401.05252) | | |
32 | March 2024 | Stable Diffusion 3 | [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_32)
33 | March 2024 | PixArt-Σ | [PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://arxiv.org/abs/2403.04692) | | |
34 | August 2024 | Flux | [Flux Announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) | | | [code](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_34)

## Lessons to Add

- [ ] Emu ([abstract](https://ai.meta.com/research/publications/emu-enhancing-image-generation-models-using-photogenic-needles-in-a-haystack/))
- [ ] CogView ([abstract](https://arxiv.org/abs/2105.13290))
- [ ] CogView 2 ([abstract](https://arxiv.org/abs/2204.14217))
- [ ] CogView 3 ([abstract](https://arxiv.org/abs/2403.05121))
- [ ] Consistency Models ([abstract](https://arxiv.org/abs/2303.01469))
- [ ] Latent Consistency Models ([abstract](https://arxiv.org/abs/2310.04378))
- [ ] Scalable Diffusion Models with State Space Backbone ([abstract](https://arxiv.org/abs/2402.05608))
- [ ] Palette: Image-to-Image Diffusion Models ([abstract](https://arxiv.org/abs/2111.05826))
- [ ] MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation ([abstract](https://arxiv.org/abs/2302.08113))
- [ ] Matryoshka Diffusion Models ([abstract](https://arxiv.org/abs/2310.15111))
- [ ] On the Importance of Noise Scheduling for Diffusion Models ([abstract](https://arxiv.org/abs/2301.10972))
- [ ] Analyzing and Improving the Training Dynamics of Diffusion Models ([abstract](https://arxiv.org/abs/2312.02696))
- [x] Elucidating the Design Space of Diffusion-Based Generative Models ([abstract](https://arxiv.org/abs/2206.00364))
- [ ] Flow Matching for Generative Modeling ([abstract](https://arxiv.org/abs/2210.02747))
- [ ] U-ViT: All are Worth Words: A ViT Backbone for Diffusion Models ([abstract](https://arxiv.org/abs/2209.12152))
- [ ] MDTv2: Masked Diffusion Transformer is a Strong Image Synthesizer ([abstract](https://arxiv.org/abs/2303.14389))
- [ ] DiffiT: Diffusion Vision Transformers for Image Generation ([abstract](https://arxiv.org/abs/2312.02139))
- [ ] Scaling Vision Transformers to 22 Billion Parameters ([abstract](https://arxiv.org/abs/2302.05442))
- [ ] DiG: Scalable and Efficient Diffusion Models with Gated Linear Attention ([abstract](https://arxiv.org/abs/2405.18428))

## Resources

Most of the implementations have been consolidated into a single [image and video diffusion](https://github.com/swookey-thinky/xdiffusion) repository, which is configurable through YAML files.

If you are interested Video Diffusion Models, take a look through [video diffusion models](https://github.com/swookey-thinky/xdiffusion) where we are adding all of the latest video diffusion model paper implementations, on an equivalent MNIST dataset for video.
