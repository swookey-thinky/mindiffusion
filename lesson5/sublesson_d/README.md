# Lesson 5d - Denoising Diffusion Probabilistic Models - Extended Class Conditioning

This lesson explores using control mechanisms to conditionally generate samples of the class we want. So far, all of the diffusion models we have explored have been *unconditional* diffusion models - you have no control over what the final image looks like. However, most of the popular image generators you have used are *conditional* diffusion models - you can control what images are generated through a conditioning mechanism, typically a text caption to describe the output or a reference image to generate a similar image. In this lesson, we are going to add conditional control, specifically class conditioning, to the diffusion model, to demonstrate one approach to controlling the output of the image generation process. In later lessons, we will extend this conditioning to other modalities, until we have replicated the text-to-image functionality present in public image generation applications. 

In the last lesson, we added basic class conditioning inspired by the use of cross attention in [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752). In this lesson, we are going to implement the actual conditioning mechanism from that paper, to give us a general framework for how we will eventually implement text-to-image and other conditioning mechanisms in later lessons. This conditioning architecture uses a more gneral transformer-like architecture for the conditional embeddings, in addition to cross attention as introduced in the last lesson.

For reference, you can find the original LDM implementation of this transformer architecture [here](https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/attention.py#L218).

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
