# Lesson 14 - Latent Diffusion Models

In this lesson, we are going to explore one of the key optimizations that allowed larger models like Stable Diffusion to come into existence - latent space training. Namely, every lesson we have looked at until now has built a diffusion model in *pixel* space. That is, the forward and reverse diffusion processes (and training) operate at the same resolution as the images themselves. For MNIST, this isn't that big of a deal, since that resolution is 1x28x28. But for high resolution images, the amount of data is orders of magnitude larger!

Training at higher resolutions is the primary motivation of the paper we are exploring in this lesson, [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). Their key insight was that instead of training the diffusion models in *pixel* space, they would train the models in a much lower resolution *latent* space. So for example if the training data we are using is at 3x512x512 resolution, what if we could instead downsample that data to a lower dimensional intermediate space (say 3x64x64), and learn a model at this reduced dimensionality, before converting back to the original higher resolution? Even better, if we could losslessly convert from the higher resolution to the lower resolution and back, then we would be able to learn the higher dimensionality data distributions using only a fraction of computational resources without any quality loss!

This is exactly what the authors of the above paper did. They use a [Variational Auto-Encoder](https://arxiv.org/abs/1312.6114) to convert the input data distribution from the higher dimensionality into an intermediate representation at a much lower dimensionality (typically a downsampling factor of 8). They then learn the forward/reverse process diffusion model at the lower dimensionality, before using the VAE to convert back to the original resolution and into pixel space for the generated data distribution. The VAE the authors used was an encoder-decoder network trained with a perceptual loss, a patch-based adversarial objective, and penalized with a KL-divergence regularization term. For this lesson, we will use a much simpler autoencoder, because we are just interested in seeing how it works. But feel free to explore using more complicated autoencoders!

The authors original Autoencoder model is defined [here](https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/autoencoder.py#L285). The loss it was trained with is defined [here](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/losses/contperceptual.py#L7)



In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

This lesson contains two parts. First, we need to train the VAE that will be used to compress the input data distribution into a lower dimension latent space:

```
> python autoencoders/train.py
```

The saved checkpoints, and sampled images, are saved into the `output/autoencoderskl` directory.

Second, with the above trained VAE, we can now train the diffusion model. To train the diffusion model, simply run the `train_mnist.py` script under this directory in your virtual environment, and point the script to the trained VAE checkpoint:

```
> python train_mnist.py --autoencoder_checkpoint output/autoencoderkl/vae-29900.pt
```

Output files (including sample generated images of the model in progress) are stored by timestep in the `output` directory.

## Results

After training the network for 30k steps, the model is able to generate samples like:

![LDM Unconditional](https://drive.google.com/uc?export=view&id=1PN7t8qNKOMbqvNqM3L6EeQThUxrW6L2n)

And here is the result of conditionally sampling the model, after 10k steps of training. For conditioning,
we used the following text prompts:

<pre>
one nine two 3 five 0 3 one 
8 zero seven 0 nine 9 nine two 
7 7 eight 7 0 seven one eight 
7 six 0 0 nine 0 7 one 
9 8 seven 3 3 two two 6 
seven 1 6 4 eight eight three seven 
4 three 2 6 one five eight seven 
7 nine 4 8 0 0 zero six 
</pre>

![LDM Conditional](https://drive.google.com/uc?export=view&id=1l0l-BV7aNXEhYyeT0a2ItodEHVuPVV-L)

As you can see, the model has successfully learned how to generate instances of the requested classes.
