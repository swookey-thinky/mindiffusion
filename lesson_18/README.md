# Lesson 18 - Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)

In this lesson we are going to learn about the Imagen diffusion model from [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487).

Imagen posits that a larger language model (deep language understanding) yields better results than scaling other factors (model size, embedding size, etc). In Imagen, the authors used a T5 XXL language model, in conjunction with a cascade of diffusion models - a base  64x64 resolution diffusion model followed by a 64->256 diffusion super-resolution model and a final 256->1024 diffusion super-resolution model. Some of the attributes of Imagen include:

1. T5 XXL text encoder
2. Cascade of diffusion models
3. Classifier free guidance with large guidance weights
4. Static/dynamic thresholding of x_hat predictions in sampling
5. Gaussian conditioning augmentation and conditioning on the augmentation level in the super-resolution models
6. Base network uses the Improved DDPM architecture
7. Text embeddings are added to the timestep conditioning via a pooled embedding vector, as well as at multiple resolution using cross attention from Latent Diffusion Models. LayerNorm at the attention and pooling layers helped as well.
8. Improved UNet architecture for the super-resolution models (Efficient UNet)

We've also introduced a new code structure in this lesson. Since all of the lessons are building off each, and are essentially "picking and choosing" different pieces, we merged 
the implementations of all of the lessons and made it easier to configure them through YAML files. So now you will see the main details of the lesson in the YAML files, and the individual additional pieces added in code.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

DaLL\*E 2 training involves 3 steps:

1.) Train a CLIP model to generate image and text embeddings
2.) Train the prior network to generate image embeddings given text embeddings
3.) Train the decoder network to generate images from predicted image embeddings.

For step 1.), we use a pretrained CLIP model from OpenAI. To train the prior network, use:

```
> python train_prior_mnist.py
```

This will generate model checkpoints in the `output/prior` directory. To train the decoder network, run:

```
> python train_decoder_mnist.py
```

This will save the decoder model checkpoints into `output/decoder`. In order to sample from the full pipeline (`text prompts -> image embeddings -> images`), you can run:

```
> python sample_dalle2.py --diffusion_prior <path to prior checkpoint> --diffusion_decoder <path to decoder checkpoint>
```
Generated samples will be saved to `output/samples_dalle2`.

## Results

After training the prior network for 10k steps, the full Imagen pipeline is able to generate samples like the below:

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
