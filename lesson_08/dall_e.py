"""DaLL-E text-image transformer.

Implements the DaLL-E transformer model from "Zero-Shot Text-to-Image Generation"
(https://arxiv.org/abs/2102.12092). DaLL-E is a transformer model which predicts
image tokens given a sequence of text tokens.
"""

from einops import rearrange
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dvae import DiscreteVAE
from tokenizer import SimpleTokenizer
from transformer import Transformer
from utils import exists, is_empty, convert_labels_to_tokens, top_k


class DalleModel(torch.nn.Module):
    """DaLL-E Model Implementation."""

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        text_vocab_size: int,
        num_text_tokens_in_sequence: int,
        image_vocab_size: int,
        num_image_tokens_per_dim: int,
        embedding_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        output_dropout_prob: float = 0.1,
        loss_img_weight: int = 7,
        mlp_activation: str = "gelu_jit",
    ):
        """Initialize a new instance of DalleModel.

        Args:
            num_layers: The number of transformer layers to use. DaLL-E used 64.
            hidden_size: The hidden size of the transformer layers. DaLL-E used
                3968 (num_attention_heads * 64)
            num_attention_heads: The number of attention heads to use for attention
                layer. DaLL-E used 62.
            text_vocab_size: The vocabulary size of the text tokenizer. DaLL-E used
                a tokenizer with a vocabulary size of 16384.
            num_text_tokens_in_sequence: The number of text tokens in a sequence. DaLL-E
                used 256.
            image_vocab_size: The vocabulary size of an image token, from the discrete
                VAE. DaLL-E used 8192.
            num_image_tokens_per_dim: The number of image tokens per spatial dimension,
                from the discrete VAE. DaLL-E used 32.
            embedding_dropout_prob: Dropout probability for the text/image embeddings layer.
            attention_dropout_prob: Dropout probability for the attention layers.
            output_dropout_prob: Dropout probability for the final output layer.
            loss_img_weight: Weight for the image loss. DaLL-E used 7, corresponding
                to 7/8 for the image and 1/8 for the text.
            mlp_activation: Activation function for the MLP layer.
        """
        super(DalleModel, self).__init__()
        self.image_tokens_per_dim = num_image_tokens_per_dim
        self.num_image_tokens_in_sequence = num_image_tokens_per_dim**2
        self.num_text_tokens_in_sequence = num_text_tokens_in_sequence
        self.total_sequence_length = (
            self.num_text_tokens_in_sequence + self.num_image_tokens_in_sequence
        )
        self.total_vocab_size = text_vocab_size + image_vocab_size
        self.text_vocab_size = text_vocab_size
        self.image_vocab_size = image_vocab_size
        self.loss_img_weight = loss_img_weight

        # Embeddings to go from discrete tokens to vectors.
        self.text_embeddings = torch.nn.Embedding(text_vocab_size, hidden_size)
        self.image_embeddings = torch.nn.Embedding(image_vocab_size, hidden_size)

        # Position embedding (serial).
        self.text_pos_embeddings = torch.nn.Embedding(
            num_text_tokens_in_sequence + 1, hidden_size
        )
        # Image positional embeddings. It wasn't clear from the DaLL-E paper
        # which type of positional embedding was used. "Axial Attention in Multidimensional Transformers"
        # and "Reformer: The Efficient Transformer" both used axial positional embeddings,
        # similar to https://github.com/lucidrains/axial-positional-embedding. Here,
        # we just use a learned embedding of the rows and columns of each token.
        self.image_row_embeddings = torch.nn.Embedding(
            num_image_tokens_per_dim, hidden_size
        )
        self.image_col_embeddings = torch.nn.Embedding(
            num_image_tokens_per_dim, hidden_size
        )

        # Converts transformer output to per-token logits of
        # the total vocabulary size.
        self.to_logits = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, self.total_vocab_size),
        )

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Transformer
        self.transformer = Transformer(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_text_tokens_in_sequence=num_text_tokens_in_sequence,
            num_image_tokens_per_dim=num_image_tokens_per_dim,
            mlp_activation=mlp_activation,
            attention_dropout_prob=attention_dropout_prob,
            output_dropout_prob=output_dropout_prob,
        )

    @torch.inference_mode()
    def sample(
        self,
        num_samples: int,
        batch_size: int,
        dvae: DiscreteVAE,
        tokenizer: SimpleTokenizer,
        temperature=1.0,
    ):
        """Generate random text-conditional samples from the model.

        Args:
            num_samples: Total number of samples to generate.
            batch_size: Batch size to use for sample generation.
            dvae: The DiscreteVAE to use for image token generation.
            tokenizer: Text tokenizer to use for text token generation.
            temperature: Logit scaling factor.

        Returns:
            Tensor batch of num_samples images of shape (B, 1, 32, 32).
        """
        # Use the device that the current model is on.
        # Assumes all of the parameters are on the same device
        device = next(self.parameters()).device

        # Create a random set of text tokens to generate images from.
        labels = torch.randint(low=0, high=10, size=(num_samples,), device=device)
        text_tokens, prompts = convert_labels_to_tokens(
            labels, tokenizer, self.num_text_tokens_in_sequence, return_prompts=True
        )
        text_tokens = text_tokens.to(device)

        # To save memory, generate the samples in batches.
        total_samples = 0
        images_list = []
        while total_samples < num_samples:
            # Grab the tokens for the batch.
            batch_text_tokens = text_tokens[total_samples : total_samples + batch_size]

            # We start with just the text tokens. out is shape
            # [B, num_text_tokens_in_sequence]
            out = batch_text_tokens

            # We are starting with the text tokens, and are going to sequentially
            # generate the image tokens that correspond to that text.
            for idx in tqdm(
                range(out.shape[1], self.total_sequence_length), leave=False
            ):
                # idx marks the image token that we are currently generating,
                # ranging from [0, num_image_tokens_in_sequence]
                idx -= self.num_text_tokens_in_sequence

                # Predict the next set of text tokens.
                logits = self(
                    out,
                    return_loss=False,
                )

                # Grab the logits for the next image token. This is the
                # last token in the sequence of predicted logits.
                logits = logits[:, -1, self.text_vocab_size :]

                # Temperature scaling
                logits /= temperature

                # Filters the logits to push their values to the boundaries
                filtered_logits = top_k(logits)

                # Convert the logits into probabilities
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)

                # Sample from the probabilities to generate a token from
                # the image vocabulary.
                sample = torch.multinomial(probs, 1)

                # Add the token the output and keep sequencing.
                out = torch.cat((out, sample), dim=-1)

            # Grab the codebook (the image tokens) from the sequenced output
            codebooks = out[:, -self.num_image_tokens_in_sequence :]
            out = out.contiguous().long()

            # Decode the codebook into an image
            images = dvae.decode(codebooks)
            images_list.append(images.detach().to("cpu"))
            total_samples += batch_size
        return torch.cat(images_list, dim=0), prompts

    def forward(
        self,
        text_and_image_tokens: torch.Tensor,
        return_loss: bool = False,
    ):
        """Calculates the loss of the model.

        Args:
            text_and_image_tokens: Tensor batch of text and image tokens.
            return_loss: If True, returns the loss of the model. Otherwise,
                returns the logits of the model, with the token predictions.

        Returns:
            Loss of the model, or the logits of the token predictions.
        """
        text_tokens = text_and_image_tokens[:, : self.num_text_tokens_in_sequence]
        text_range = torch.arange(self.num_text_tokens_in_sequence)
        text_range += self.text_vocab_size - self.num_text_tokens_in_sequence
        text_range = text_range.to(text_and_image_tokens.device)
        text_tokens = torch.where(text_tokens == 0, text_range, text_tokens)

        text_embeddings = self.text_embeddings(text_tokens) + self.text_pos_embeddings(
            torch.arange(text_tokens.shape[1], device=text_and_image_tokens.device)
        )

        image_tokens = text_and_image_tokens[:, self.num_text_tokens_in_sequence :]

        if exists(image_tokens) and not is_empty(image_tokens):
            image_embeddings = self.image_embeddings(
                image_tokens
            ) + self._get_image_pos_embeddings(image_tokens)
            embeddings = torch.cat((text_embeddings, image_embeddings), dim=1)
        else:
            embeddings = text_embeddings

        # Run the embeddings through the transformer.
        transformer_output = self.transformer(embeddings)

        logits = self.to_logits(transformer_output)
        if return_loss is False:
            return logits

        # The transformer predicts the next token, so skip the first token
        # to create the labels.
        labels = (
            torch.cat((text_tokens[:, 1:], image_tokens), dim=1).contiguous().long()
        )
        logits = rearrange(logits, "b n c -> b c n")

        text_logits = (
            logits[:, : self.text_vocab_size, : self.num_text_tokens_in_sequence]
            .contiguous()
            .float()
        )
        image_logits = (
            logits[:, self.text_vocab_size :, self.num_text_tokens_in_sequence : -1]
            .contiguous()
            .float()
        )

        loss_text = F.cross_entropy(
            text_logits, labels[:, : self.num_text_tokens_in_sequence]
        )
        loss_img = F.cross_entropy(
            image_logits, labels[:, self.num_text_tokens_in_sequence :]
        )

        loss = (loss_text + self.loss_img_weight * loss_img) / (
            self.loss_img_weight + 1
        )
        return loss, {
            "text": loss_text.data.detach().float(),
            "image": loss_img.data.detach().float(),
        }

    def _get_image_pos_embeddings(self, image_tokens):
        """Gets the image position embeddings.

        Embeds the current row and column for each spatial image
        token.
        """
        input_shape = image_tokens.size()

        row_ids = (
            torch.arange(
                0,
                input_shape[-1],
                dtype=torch.long,
                device=image_tokens.device,
            )
            // self.image_tokens_per_dim
        )
        row_ids = row_ids.unsqueeze(0).view(-1, input_shape[-1])
        col_ids = (
            torch.arange(
                0,
                input_shape[-1],
                dtype=torch.long,
                device=image_tokens.device,
            )
            % self.image_tokens_per_dim
        )
        col_ids = col_ids.unsqueeze(0).view(-1, input_shape[-1])
        return self.image_row_embeddings(row_ids) + self.image_col_embeddings(col_ids)
