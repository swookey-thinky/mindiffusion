# Lesson 21 - Scalable Diffusion Models with Transformers (DiT)

In this lesson we are going to learn about the DiT diffusion model from [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).

This model is interesting because it was the first to replace the UNet architecture in the diffusion score network with a transformer based architecture, from [Attention is all you need](https://arxiv.org/abs/1706.03762), and in particular uses the Vision Transformer architecture from [An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/abs/2010.11929) as the backbone for the score network.

The original DiT paper used transformers to train a latent diffusion model, but for simplicity, we will be training in pixel space rather than latent space. Note that this is a class conditional model as well, without text alignment (for now). Future research uses transformer backbones with text alignment so we will explore that there.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

Run the training script at `train_mnist.py` like:

```
> python train_mnist.py
```

Generated samples will be saved to `output/dit`.

## Results

After training the prior network for 10k steps, the full Dit pipeline is able to generate samples like the below:

<!--
 ![Imagen](https://drive.google.com/uc?export=view&id=1SVWvGD0FhakjL2G9QCyi0TbiaZ_6ILKM)

 The prompts we used for generation above were:

<pre>
8 one seven 1 7 six 6 two 
1 8 4 six 3 9 8 6 
five three eight 2 1 9 seven 7 
two 8 9 three 3 0 3 6 
two two 7 two 0 three nine nine 
five six one 1 0 seven six 3 
0 three 2 one 3 nine six 0 
2 9 zero 4 7 two 9 eight 
</pre>
-->
