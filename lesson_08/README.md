# Lesson 8 - DALL\*E (Zero-Shot Text-to-Image Generation)

In this lesson, we are going to look at one of the first image generation models to come out of OpenAI - DALL\*E - from the paper [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092). But wait a minute you say! DALL\*E is not a diffusion model!!! You are correct, DALL\*E is not a diffusion model, it's an autoregressive variational model. However, later versions of DALL\*E have become the standard image generation models produced by OpenAI, so I think its important to at least understand where they come from, so let's take a brief aside here to learn about an important, non-diffusion based image generation model. This model is important because it really brought the *text-conditional* image generation models to the public attention. We briefly saw in [Lesson 5e](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_05/sublesson_e) what a text-conditional (eg text-to-image) model looks like, so this is a good opportunity to study another variation of them. 

The authors never released the original source code or training algorithm to DALL\*E. They did release one piece of it, the discrete VAE used to encode the image embedding, [here](https://github.com/openai/DALL-E/tree/master). For other references, there is an excellent PyTorch implementation from the prolific Phil Wang [here](https://github.com/lucidrains/DALLE-pytorch/tree/main).


In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

First, we need to train the Discrete VAE that is used to tokenize the image data.

```
> python train_dvae.py
```

Model checkpoints will be saved into `outut/dvae`. Once we have a trained dVAE, we can now train the DaLL-E model.

```
> python train_mnist.py --dvae_checkpoint="output/dvae/dvae-30000.pt"
```

Output files (including sample generated images of the model in progress) are stored by timestep in the `output/dall_e` directory.

You can also generate samples once you have both a trained dVAE and DaLL-E model.

```
> python generate_mnist.py --dvae_checkpoint="output/dvae/dvae-30000.pt" --dall_e_checkpoint="output/dall_e/dall_e-60000.pt"
```

The generated samples will be written to `output/dall_e_samples`.

## Results

Here are the results of the dVAE, with the original samples and the reconstructed samples:

| Original | Reconstructed |
| -------- | -------- |
|![Original](https://drive.google.com/uc?export=view&id=1whSORJMoEJDdC_L_R7W_I6ygo1FYvS2K) | ![Reconstructed](https://drive.google.com/uc?export=view&id=1u8i70csgMka7Ka0qrd0FcBweYjWz4pPn) |

Here are the results of sampling the DaLL-E model after 60k steps. For conditioning,
we used the following text prompts:
<pre>
eight 5 7 three 6 3 six seven 
4 7 0 2 1 1 six seven 
four nine nine 9 9 five 2 three 
four 7 3 nine 6 six 4 2 
zero eight five four 6 2 5 nine 
0 1 5 eight eight 1 five 1 
0 8 one 8 eight 1 6 3 
zero two 2 eight 6 3 9 1 
</pre>

![DaLL-E](https://drive.google.com/uc?export=view&id=1LCPtLocm8EQACZGeaYdkX7socMd5SRYS)
