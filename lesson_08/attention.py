"""Attention modules for use with DaLL-E."""

import math
import torch

from utils import split_tensor_along_last_dim


class SelfAttention(torch.nn.Module):
    """Masked Self Attention

    Self-attention layer takes input with size [B, sequence_length, hidden_size]
    and creates output of the same size. Uses layer specific attention masks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout_prob: float = 0.1,
        output_dropout_prob: float = 0.1,
    ):
        """Initialize a new instance of SelfAttention.

        Args:
            hidden_size: Total hidden size of the layer.
            num_attention_heads: Number of attention heads N. Note that we
                                require N to be divisible by number of GPUs
                                used to parallelize the model. Also, we
                                require hidden size to be divisible by N.
            attention_dropout_prob: Dropout probability for the attention scores.
            output_dropout_prob: Dropout probability for the output.

        We use the following notation:
        h: hidden_size
        n: num_attention_heads
        p: number of partitions
        np: n/p
        hp: h/p
        hn: h/n
        b: batch size
        s: sequence length
        """
        super().__init__()

        assert (
            hidden_size % num_attention_heads == 0
        ), f"Hidden size {hidden_size} must be divisible by num_attention_heads {num_attention_heads}"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads

        self.query_key_value = torch.nn.Linear(hidden_size, 3 * hidden_size)
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

    def _transpose_for_scores(self, tensor):
        """Rearrange QKV tensors for score calculation.

        Transposes a 3D tensor [B, sequence_length, num_attention_heads*hidden_size_per_attention_head]
        into a 4D tensor with size [B, num_attention_heads, sequence_length, hidden_size_per_attention_head].

        Args:
            tensor: Tensor batch of QKV value.
        """
        new_tensor_shape = tensor.size()[:-1] + (
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
        )
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def _calculate_attention_scores(self, query_layer, key_layer, mask):
        """Calculate the attention scores.

        Calculates the affinities amongst tokens, using the passed in mask.
        These affinities will be used the the attention values.

        Args:
            query_layer:
            key_layer:
            mask:
        """
        key_t = key_layer.transpose(-1, -2)
        attention_scores = torch.matmul(query_layer, key_t) / math.sqrt(
            self.hidden_size_per_attention_head
        )
        mask = mask[:, :, -attention_scores.shape[-2] :]

        # Mask value just needs to generate a large negative number,
        # for the softmax calculation later.
        mask_value = 10000.0
        attention_scores = torch.mul(attention_scores, mask) - mask_value * (1.0 - mask)
        return attention_scores

    def forward(
        self,
        x,
        mask,
    ):
        """Calculates masked self-attention.

        Args:
            x: Tensor batch of shape (B, sequence_length, hidden_size)
            mask: Mask used or attention calculation, of shape (1, 1, sequence_length, sequence_length)
        """
        mixed_x_layer = self.query_key_value(x)

        (mixed_query_layer, mixed_key_layer, mixed_value_layer) = (
            split_tensor_along_last_dim(mixed_x_layer, 3)
        )

        # Split out the QKV
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        # Calculate the attention scores.
        attention_scores = self._calculate_attention_scores(
            query_layer=query_layer, key_layer=key_layer, mask=mask
        )

        # Attention probabilities.
        # [B, num_attention_heads, sequence_length, sequence_length]
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Context layer. This is the affinities between the queries/keys
        # and the values.
        context_layer = torch.matmul(attention_probs, value_layer)

        # Move the sequence_length channel back to the original position.
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Reshape for the output layer
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Final dense layer.
        output = self.dense(context_layer)
        # output = torch.clamp(output, min=-65504, max=65504)

        # Final dropout
        output = self.output_dropout(output)
        return output


def _init_mask(text_tokens, image_tokens):
    """Creates a causal attention mask.

    Args:
        text_tokens: The number of text tokens
        image_tokens: The number of image tokens

    Returns:
        Returns a mask of shape [text_tokens+image_tokens, text_tokens+image_tokens]
    """
    attn_size = text_tokens + image_tokens
    mask = torch.tril(torch.ones(attn_size, attn_size, dtype=torch.float32))
    return mask


def get_row_mask(text_tokens, image_tokens_per_dim):
    """Creates a causal row attention mask.

    Mask corresponds to Figure 11a in the paper.

    Args:
        text_tokens: The number of text tokens in a sequence.
        image_tokens_per_dim: The number of image tokens per spatial dimension.
            The total number of image tokens in a sequence is image_token_per_dim**2.

    Returns:
        Returns a mask of shape [text_tokens+image_tokens_per_dim**2, text_tokens+image_tokens_per_dim**2]
    """
    mask = _init_mask(text_tokens, image_tokens_per_dim**2)
    step = image_tokens_per_dim + 1
    for col in range(text_tokens, mask.size(1)):
        mask[col + step :, col] = 0.0
    return mask


def get_col_mask(text_tokens, image_tokens_per_dim):
    """Creates a causal column attention mask.

    Mask corresponds to Figure 11b in the paper.

    Args:
        text_tokens: The number of text tokens in a sequence.
        image_tokens_per_dim: The number of image tokens per spatial dimension.
            The total number of image tokens in a sequence is image_token_per_dim**2.

    Returns:
        Returns a mask of shape [text_tokens+image_tokens_per_dim**2, text_tokens+image_tokens_per_dim**2]
    """
    mask = _init_mask(text_tokens, image_tokens_per_dim**2)
    step = image_tokens_per_dim - 1
    for col in range(text_tokens, mask.size(1)):
        for i in range(1, mask.size(0), step + 1):
            mask[col + i : col + i + step, col] = 0.0
    return mask


def get_conv_mask(
    text_tokens,
    image_tokens_per_dim,
    kernel=11,
):
    """Creates a causal convolutional attention mask.

    Mask corresponds to Figure 11d in the paper.

    Args:
        text_tokens: The number of text tokens in a sequence.
        image_tokens_per_dim: The number of image tokens per spatial dimension.
            The total number of image tokens in a sequence is image_token_per_dim**2.

    Returns:
        Returns a mask of shape [text_tokens+image_tokens_per_dim**2, text_tokens+image_tokens_per_dim**2]
    """
    mask = _init_mask(text_tokens, image_tokens_per_dim**2)
    shift = kernel // 2

    # Starting at the first pixel, create the initial convolutional
    # mask
    c = set()
    for pixel_row in range(shift + 1):
        pixel_idx = pixel_row * image_tokens_per_dim
        # Update to the right
        c.update(range(pixel_idx, pixel_idx + shift + 1))
        # Update to the left (wraparound)
        c.update(
            range(
                pixel_idx + image_tokens_per_dim - 1,
                pixel_idx + (image_tokens_per_dim - 1 - shift),
                -1,
            )
        )
    c = list(c)

    for col in range(text_tokens, mask.size(1)):
        mask[col:, col] = 0.0

    for col in range(text_tokens, mask.size(1)):
        rc = [x + col for x in c]
        rc = list(filter(lambda x: x < (text_tokens + image_tokens_per_dim**2), rc))
        mask[rc, col] = 1.0

    return mask
