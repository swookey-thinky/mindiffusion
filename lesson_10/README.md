# Lesson 10 - Guided Diffusion

In this lesson we are going to learn about Guided Diffusion from the paper [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233). This paper introduced the concept of *classifier guidance* to guide the diffusion model during sampling.


You can find the authors original codebase [here](https://github.com/openai/guided-diffusion).

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

To train the classifier model, simply run the `train_classifier.py` script under this directory in your virtual environment:

```
> python train_classifier.py
```

To train the diffusion model, simply run the `train_mnist.py` script under this directory in your virtual environment:

```
> python train_mnist.py
```

Output files (including sample generated images of the model in progress) are stored by timestep in the `output` directory.

To sample from the model using classifier guidance, run the `sample_with_guidance.py` script under this directory in your virtual environment:

```
> python sample_with_guidance.py --diffusion_checkpoint <path to diffusion model> --classifier_checkpoint <path to classifier checkpoint>
```

Samples with be saved to the `output/samples_with_guidance` directory.

## Results

After training the network for 10k steps, the model is able to conditionally generate samples like:

Labels Used
```
4 7 3 9 0 9 2 5 
5 8 5 4 9 8 8 5 
0 2 1 2 8 1 5 8 
6 9 0 5 7 2 3 5 
4 1 1 7 5 0 3 5 
3 5 7 1 8 4 4 4 
8 9 6 9 8 4 9 4 
5 5 3 9 5 1 4 0 
```

![Guided Diffusion](https://drive.google.com/uc?export=view&id=1uBIHLW-COW_DzblGaePOlHXUMSQ1Pp1T)

Using classifier guidance, the model is able to generate samples like (using classifier scale=10.0):

Labels Used:

```
7 1 5 8 8 9 5 4 
7 1 0 5 5 2 8 2 
4 6 1 5 2 2 2 3 
9 7 1 2 5 1 5 1 
8 1 1 5 9 9 5 5 
2 8 2 4 3 0 5 3 
3 8 4 1 0 8 8 8 
4 6 3 3 0 5 8 6 
```

| Unguided | Classifier Guided 
| -------- | --------
| ![Unguided Diffusion](https://drive.google.com/uc?export=view&id=1jYNr0wTFQHzgXj-o0M5UA263O0NwZqXG) | ![Classifier Guided Diffusion](https://drive.google.com/uc?export=view&id=14_GsXX6n6SfESMdBa-s7FhTtdpq3l2kh)

