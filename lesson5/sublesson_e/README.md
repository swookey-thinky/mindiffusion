# Lesson 5e - Denoising Diffusion Probabilistic Models - Text-to-Image

This lesson finally gives us our first text-to-image model! In the last lesson, we implemented the conditioning architecture from latent diffusion and the paper [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). However, we still used basic class information for the conditioning signal. In this lesson, we are going to finally add a text based conditioning mechanism, so that we can generate images using text like "zero" or "three". Arguably, this is not that interesting. However, this is the same mechanism that all of the modern text-to-image diffusion models use, so now we are really starting to see how things are put together.

In this lesson, we use a frozen CLIP embedder from [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) as introduced in Stable Diffusion. The original LDM paper chose a BERT embedder from [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). There is not much context as to why Stable Diffusion v1 moved to CLIP embeddings over BERT embeddings, but most likely it was due to Google's [Imagen](https://arxiv.org/abs/2205.11487) also choosing a frozen CLIP embedder. For this lesson, the choice of text embedding model is not very important, but it is important to realize the architecture we use for projecting the text embeddings into the model is general enough to support all of them.

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

Here is an example of generating only the class `5`, to show how we can control the image generation process through class conditioning.

![Class Conditioning](https://drive.google.com/uc?export=view&id=1vyNRZ06mTy3-_jqzbevm_Iz0fnE7PUND)
