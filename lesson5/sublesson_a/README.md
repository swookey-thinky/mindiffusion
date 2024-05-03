# Lesson 5a - Denoising Diffusion Probabilistic Models with Dropout

This lesson explores an improvement to the training of DDPM's, which is the addition
of dropout to the score network. Dropout was introduced in the paper
[Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580) to prevent overfitting of the network during training. This functionality was not talked about in the DDPM paper, but it was utilized in the original implementation, so we have added it here in an incremental manner, so that we can see how to perform these incremental additions to a model.

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

After training the network for 10k steps, the model is able to generate samples like:

TODO!!!
![DDPM](https://drive.google.com/uc?export=view&id=1bkSdBlvli5U2Lle9ELd2mXqZ6T0hxj6k)
