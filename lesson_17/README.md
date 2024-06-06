# Lesson 17 - Hierarchical Text-Conditional Image Generation with CLIP Latents (DaLL\*E 2)

In this lesson we are going to learn about the DaLL\*E 2 diffusion model from [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125).

DaLL\*E 2 is an interesting follow on to the original DaLL\*E text-to-image model, which as you recall from [Lesson 8](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_08) was actually a non-diffusion, transformer based text-to-image generation model. In contrast, DaLL\*E 2 is a multi-stage diffusion model whose main contribution is using [CLIP](https://arxiv.org/abs/2103.0002) image embeddings as a source of conditioning in the diffusion process. In order to accomplish this, DaLL\*E 2 first introduces a diffusion prior networkm which learns to predict the CLIP image embeddings from the CLIP text embeddings of the generation prompts. Then, DaLL\*E 2 uses a diffusion decoder network to predict a generated image conditioned on the CLIP image embeddings from the prior network, and text embeddings from the given prompts.

The authors did not release any code for their model. However Phil Wang has an implementation at [DaLL\*E 2 PyTorch](https://github.com/lucidrains/DALLE2-pytorch) that you can use to follow along if you want.

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

After training the prior network for 30k steps and the decoder network for 10k steps, the model is able to generate samples like the below:

![DaLL\*E 2](https://drive.google.com/uc?export=view&id=1SVWvGD0FhakjL2G9QCyi0TbiaZ_6ILKM)

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

