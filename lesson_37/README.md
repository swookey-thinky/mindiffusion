# Lesson 37 - SANA

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to build the Sana diffusion model from NVidia. The training, inference, and model code are released [here](https://github.com/NVlabs/Sana), and there is a technical report published [here](https://arxiv.org/abs/2410.10629) as well.

## Introduction 

Sana is an improvement on the other [transformer](https://arxiv.org/abs/2212.09748) based diffusion models like [Pixart-Alpha](https://arxiv.org/abs/2310.00426) and [SD3](https://arxiv.org/abs/2403.03206). It uses the rectified flow formulation from [SD3](https://arxiv.org/abs/2403.03206) which builds on flow matching from [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). Interestingly, it uses what the authors call NoPE positional embedding - which just means they did not use any positional embedding and hypothesize that adding the 3x3 convolution in the Mix-FFN layer of each block (which replaces the MLP-FFN of traditional transformers) implicitly adds a positional bias for each token.

Other improvement include the use of Linear Attention from [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236), and a much more aggressive latent VAE, to encode the pixel space images into a much smaller (32x downsampling vs the traditional 8x) latent space. They also use [Gemma 2](https://arxiv.org/abs/2408.00118), a decoder only GPT style large language model, to embed the captions for each image. Theoretically this provides an embedding model for the text with improved semantic understanding. However, I question this assumption, since decoder only language models like Gemma are trained for next token prediction tasks, while text embedding models like T5 are trained for essentially text understanding. It seems implausible to me that a next token prediction model generates a better embedding for semantic understanding than a dedicated embedding model, and potentially a better, modern embedding model like ModernBERT might be a better caption embedding model here.

In our implementation here, we train in pixel space rather than the latent space of the original model, and we scale down the transformer network to ~220m parameters from the original models ~600m parameters. We successfully trained this model on a A10 instance with a batch size of 16, 4 gradient accumulation steps, and 10000 total steps, with a total cost of $8.

Training this model has some unique differences compared to the other models we have seen so far. First off, the Gemma 2 language model, while small compared to other large language models as ~2b parameters, is still much larger than the text embedding models we have seen so like the T5 base model, which clocks in at around ~220m parameters. Also, even on GPU, the Gemma language model is considerably slower at inference than the embedding models, and especially on batched inference. This slows down training step times considerably.

So to improve the training throughput, we have created a "pre-embedded" version of MNIST, where we have stored the text embeddings for all of the examples. This way, we do not need to run the language model during training, and this saves us a considerable amount of time at each training step. So you will notice the dataset being downloaded before training starts.

Second, since we still need to run the language model at inference time (since we do no know the captions we are generating with in general) we need to do some more intelligent model-device mapping than usual. Every time we create the text embeddings during sampling, we need to move the trained model to the CPU, move the text model to the GPU, create the embedding, and then move everything back to where it was. This is essentially a version of "cpu offloading", to help us better manage the memory.

Both of the above approaches are common approaches when training larger models.

Another thing to note is that we have reduced the width of the inner dimension of the transformer blocks from 2240 -> 1152. This is still overkill for this model, and we should reduce this even further for faster training. For example, in the Flux model we trained here, we reduced the inner dimension to 384.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Configuration File

The configuration file is located in [Sana](https://github.com/swookey-thinky/mindiffusion/blob/main/config/sana.yaml).

## Training

To train the sana model, use:

```
> python train.py --config_path config/sana.yaml
```

We successfully tested training on a single A10 instance (24GB VRAM) using a batch size of 16 with a gradient accumulation steps of 4 (for an effective batch size of 64). It took about two hours to train for 10000 steps. This only used ~8gb of VRAM, so this should be trainable on a T4 as well.

NOTE: The Gemma 2 family of language models is protected behind a gated repo on Huggingface, so make sure to get an access token and set the HF_TOKEN environment variable before running the above command.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/mindiffusion/blob/main/config/sana.yaml) | [google drive](https://drive.google.com/file/d/1ksqyJPn25QjftDk_g0sOS3Qqv2p1N8Jz/view?usp=sharing) | ![Sana](https://drive.google.com/uc?export=view&id=1_sUwEAQkN58xtGwJP69q94WO9NxmOWDO)

After training the network for 10k steps, the sana model pipeline is able to generate samples like the below:

![Sana](https://drive.google.com/uc?export=view&id=1_sUwEAQkN58xtGwJP69q94WO9NxmOWDO)

The prompts we used for generation above were:

<pre>
4 two 4 six 4 five 4 7 
3 1 three 3 two 3 8 zero 
6 six six 5 2 8 3 five 
zero 1 eight nine 8 eight two eight 
3 5 4 3 6 9 four nine 
0 8 8 one 2 4 eight 7 
nine 9 two 3 2 9 9 9 
four 9 1 seven 1 5 seven five
</pre>

## Other Resources

The Sana team released their training and inference code at [github](https://github.com/NVlabs/Sana).
