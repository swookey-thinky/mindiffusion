# Lesson 27 - PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis

In this lesson we are going to build the PixArt-α diffusion model from [](). PixArt-α extends the DiT diffusion model from [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) (see [Lesson 21](https://github.com/swookey-thinky/mindiffusion/tree/main/lesson_21)) by adding text alignment using cross attention at each transformer block.

PixArt-α proposes a 3-stage pipeline to training their model. The first stage is "Pixel dependency learning", where they train a class conditional model on Imagenet to learn the "pixel dependencies" of the image generation task within the transformer framework. This model serves as the initialized weights for Stage 2.

Stage 2 is "Text-image alignment learning", where the model uses the pretrained weights from Stage 1 to learn the text alignment from the captions paired with each image.

Stage 3 is "High-resolution and aesthetic image generation", where the model created in Stage 2 is fine tuned using a high-quality, aesthetic image dataset, used to condition the model towards higher quality output.

Since our dataset is so small and Stage 3 requires only a dataset change, we only implement Stages 1 and 2 here.

There are two architectural changes in PixArt-α over DiT:

1. There is a multi-head cross-attention layer in each transformer block. It is positioned between the self-attention layer and feed-forward layer so that the model can flexibly interact with the text embedding extracted from the language model. 
2. Instead of the per-layer adaLN modulation used in DiT, there is a single adaLN modulation in the first block of the network, shared across all blocks (adaLN-single). Each layer contains a trainable embedding, which still gives the model per-layer adaptability of the timestep embedding. This changed is motivated by the fact that the class embeddings are not used in PixArt-α, so the additional control provided by the per-layer adaLN is not needed.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements-for-all-lessons) to set up your environment.

## Running the Lesson

PixArt-α is a three stage training process, and we implement the first two stages here. First, we train a class conditional diffusion model without text captions, to create a strong pixel generation baseline.

Run the training script at `train_mnist.py` like:

```
> python train_mnist.py --config_path "configs/pixart_alpha_class_conditional.yaml" --num_training_steps 30000
```

Generated samples will be saved to `output/pixart_alpha_class_conditional`.

To train the second stage, run the same training script passing in the saved checkpoint from the first stage:

```
> python train_mnist.py --config_path "configs/pixart_alpha.yaml" --num_training_steps 30000 --load_model_weights_from_checkpoint output/pixart_alpha_class_conditional/diffusion-30000.pt
```

Generated samples and model checkpoints will be saved to `output/pixart_alpha`.

## Results

After training the prior network for 30k steps, the full PixArt-α pipeline is able to generate samples like the below:

 ![DiT](https://drive.google.com/uc?export=view&id=1J6ktzFr7iqgWcf23JpgVaM81Z7sUUcmj)

 The prompts we used for generation above were:

<pre>
2 4 one nine 2 eight 7 9 
two two 2 eight nine 6 nine one 
nine 4 seven 2 one two four 1 
eight two two two six six eight nine 
five 1 seven 0 4 seven four 2 
one 0 five 9 five 4 5 four 
1 two five 4 9 0 zero one 
1 4 eight seven 8 eight zero 8 
</pre>
