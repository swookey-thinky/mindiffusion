# Lesson 33 - FLUX

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to build the Flux diffusion model from Black Forest Labs. There is no technical report yet, but they released their model and inference code [here](https://github.com/black-forest-labs/flux) and we can infer a lot about how the model is built.

Flux is a very cool model that combines a lot of recent research about image diffusion models. It extends the diffusion transformer network from [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) with parallel transformer blocks from [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442). It uses the rectified flow formulation from [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) which builds on flow matching from [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). It also uses an improved positional embedding in the transformer blocks call rotart positional embedding, from [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). This also sounds a lot like what Stable Diffusion 3 is built upon (minus the rotary position embedding), which makes sense since the authors were all key contributors to the stable diffusion series of models.

In our implementation here, we train in pixel space rather than the latent space of the original model, and we scale down the transformer network to ~50m parameters from the original models 12b parameters. We successfully trained this model on a T4 instance with a batch size of 64.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

Run the training script at `train_mnist.py` like:

```
> python train_mnist.py --config_path "configs/flux.yaml" --num_training_steps 30000
```

Generated samples will be saved to `output/flux`.

## Results

After training the network for 30k steps, the flux model pipeline is able to generate samples like the below:

![Flux](https://drive.google.com/uc?export=view&id=1_r8poe1SJxf8UtT4mmQaTT378m26hD-F)

The prompts we used for generation above were:

<pre>
3 5 six 5 six 9 six four 
5 zero eight one six 9 1 three 
two 2 8 8 zero six seven 5 
4 6 0 3 three 6 six four 
one 5 nine 2 5 6 2 three 
six four 0 4 one 8 eight five 
0 8 5 two four nine seven 1 
seven four four 2 eight 9 8 4 
</pre>
