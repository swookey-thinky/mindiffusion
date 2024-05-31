# Lesson 15 - GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models

In this lesson we are going to learn about the GLIDE diffusion model from [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741).

The GLIDE diffusion model is very similar to DDPM, which two important improvements. First,
it uses [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) to improve the sampling process (and hence necessarily trains a joint text-conditional/unconditional score network) and it uses a cross-attention based text conditioning in the score network, similar to Latent Diffusion Models, but with a single shared transformer projection across layers, rather than the per-layer transformer projection in LDM. GLIDE also experimented with CLIP guidance rather than just classifier-free guidance, but found that classifier-free guidance performed better, so we implement that here. Otherwise, GLIDE is based on the ADM architecture that we learned about in [Lesson 9](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_09) with [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). 

The authors released a smaller version of their model [here](https://github.com/openai/glide-text2im/tree/main) so you can double check your work with their work.


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

After training the network for 10k steps, the model is able to generate samples like the below with different guidance scales:


| CFG = 0.0 | CFG = 1.0 | CFG = 2.0 | CFG = 4.0 | CFG = 7.0 | CFG = 10.0 | CFG = 20.0
| ---- | ---- | ---- | ---- | ---- | ---- | ---- 
| ![GLIDE 0.0](https://drive.google.com/uc?export=view&id=10LyT0Ynsn3ti1wOEqdG4jnW_k3NamwZl) | ![GLIDE 1.0](https://drive.google.com/uc?export=view&id=1q46FWPWWYQPTv-8wYlZp55RQeIBI0dad) | ![GLIDE 2.0](https://drive.google.com/uc?export=view&id=1Pq7mlhJam8ARx87_rjzy8LTgFd0WNbDD) | ![GLIDE 4.0](https://drive.google.com/uc?export=view&id=1gYByHVKAqJuxALn2kfGNeuj-dNUJ85-I) | ![GLIDE 7.0](https://drive.google.com/uc?export=view&id=16wXV6gn8hPAoKEheRVgU3JrKllf8sek0) | ![GLIDE 10.0](https://drive.google.com/uc?export=view&id=1yGuVixiUr4JPVPsYx5UVdoQZ65Dahl7v) | ![GLIDE 20.0](https://drive.google.com/uc?export=view&id=1trNGDUg637gJfBR_qgfbjdtUzeNwovdv)

