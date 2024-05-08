# Lesson 6 - Denoising Diffusion Implicit Models

In this lesson, we are going to look into an improvement to [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) called [Denoising Diffusion Implicit Model](https://arxiv.org/abs/2010.02502). This paper is fascinating in that it dramatically speeds up the sampling process of DDPM, from 1000 timesteps down to only 50 timesteps for equivalent quality. Importantly, it also provides for deterministic sampling, such that starting with the same latents will always return the same results, which is not the case for DDPM.

You can find the original source code from the authors [here](https://github.com/ermongroup/ddim).

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

This repository requires a trained model from [Lesson 5e](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_05/sublesson_e), and uses that model to perform efficient sampling. If you don't already have a trained model, you can download this checkpoint [here](https://drive.google.com/file/d/1SGk-pqZK6VmuuB4RgpjPf56uZIzHxaTK/view?usp=drive_link) and use it below.

To sample from the model above, simply run the `sample_ddim.py` script under this directory pointing to the model checkpoint downloaded above, in your virtual environment:

```
> python sample_ddim.py --model_checkpoint <path to checkpoint file>
```

The output directory will include unconditional samples generated from the model, as well as conditional samples and their prompts. These samples are generated using only 50 inference steps!

## Results

Here are the unconditional samples generated using only 50 steps:

![DDIM Unconditional](https://drive.google.com/uc?export=view&id=1SSVGEOvO9Pmz-SdWLFFTVduzwF8gAR8_)

Here are the prompts, and the samples generated, to demonstrate conditional sampling in only 50 steps:

<pre>
one 7 5 6 7 7 seven 5 
0 1 7 1 5 8 nine 1 
0 2 4 9 3 one 6 2 
0 seven 9 2 eight 9 four three 
8 five 0 three 9 four 5 one 
3 0 6 four 0 nine four 7 
9 0 1 0 2 6 two seven 
2 6 two zero 9 five 0 nine 
</pre>

![DDIM Conditional](https://drive.google.com/uc?export=view&id=1kLKufiSWTzVW54b-bvQZD17TcJcWFUSC)
