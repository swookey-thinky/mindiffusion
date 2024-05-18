"""Transformer blocks used in DaLL-E.

Implementation inspired by:

https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
and
https://github.com/ai-forever/ru-dalle/blob/master/rudalle/dalle/model.py
"""

import torch

from attention import get_col_mask, get_conv_mask, get_row_mask, SelfAttention


class Transformer(torch.nn.Module):
    """Base Transformer class.

    This module takes input from the embedding layer and it's output can
    be used directly by a logit layer (dVAE decoder). It consists of N (num-layers)
    blocks of transformer layers, followed by a final layer norm.

    Arguments:
        num_layers: Number of transformer layers.
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self attention.
        num_text_tokens_in_sequence: The number of text tokens in a sequence.
        num_image_tokens_per_dim: The number of image tokens for spatial dimension.
        attention_dropout_prob: dropout probability of the attention score in self attention.
        output_dropout_prob: dropout probability for the outputs after self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid division by zero.
        mlp_activate: Activation function to use in layers.
    """

    def __init__(
        self,
        num_layers,
        hidden_size,
        num_attention_heads,
        num_text_tokens_in_sequence,
        num_image_tokens_per_dim,
        attention_dropout_prob=0.1,
        output_dropout_prob=0.1,
        layernorm_epsilon=1.0e-5,
        mlp_activation="gelu_jit",
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_text_tokens_in_sequence = num_text_tokens_in_sequence
        self.num_image_tokens_per_dim = num_image_tokens_per_dim

        # Transformer layers.
        self.layers = torch.nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size,
                    num_attention_heads,
                    attention_dropout_prob,
                    output_dropout_prob,
                    layernorm_epsilon,
                    mlp_activation=mlp_activation,
                )
                for i in range(num_layers)
            ]
        )

        # Mask from Figure 11a
        row_mask = get_row_mask(
            num_text_tokens_in_sequence,
            num_image_tokens_per_dim,
        )
        # Mask from Figure 11b
        col_mask = get_col_mask(
            num_text_tokens_in_sequence,
            num_image_tokens_per_dim,
        )
        # Mask from Figure 11c
        conv_mask = get_conv_mask(
            num_text_tokens_in_sequence,
            num_image_tokens_per_dim,
        )
        # The full causal attention mask
        full_mask = torch.tril(
            torch.ones(
                (
                    1,
                    1,
                    self.num_text_tokens_in_sequence + self.num_image_tokens_per_dim**2,
                    self.num_text_tokens_in_sequence + self.num_image_tokens_per_dim**2,
                )
            )
        )

        self.register_buffer("row_mask", row_mask)
        self.register_buffer("col_mask", col_mask)
        self.register_buffer("conv_mask", conv_mask)
        self.register_buffer("full_mask", full_mask)

        # Final layer norm before output.
        self.final_layernorm = torch.nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

    def _get_layer_mask(self, layer_id) -> torch.Tensor:
        """Gets the per-layer self attention mask.

        Implements the per-layer self attention masks,
        as desribed in B.1.
        """
        if (layer_id - 1) % 4 == 0:
            layer_mask = self.col_mask
        elif layer_id != self.num_layers - 1:
            layer_mask = self.row_mask
        else:
            layer_mask = self.conv_mask
        return layer_mask

    def forward(
        self,
        x,
    ):
        """Run the transformer.

        Args:
            x: Tensor batch of shape (B, sequence_length, hidden_dim).
        """
        # Shape the full attention mask to the size of the input. At inference,
        # there may only be text embeddings, hence the size of the mask may not be
        # the full sequence length.
        attention_mask = torch.tile(self.full_mask, (x.shape[0], 1, 1, 1))
        attention_mask = attention_mask[:, :, : x.shape[1], : x.shape[1]]

        for i, layer in enumerate(self.layers):
            layer_mask = self._get_layer_mask(i)[
                : attention_mask.size(2), : attention_mask.size(3)
            ]
            mask = torch.mul(attention_mask, layer_mask)

            x = layer(x, mask)

        output = self.final_layernorm(x)
        return output


class TransformerLayer(torch.nn.Module):
    """
    A single transformer layer. Its consists of the following layers:
        layer norm
        self attention
        residual connection
        layer norm
        mlp
        residual connection

    Transformer layer takes input with size [b, sequence_length, hidden_dim] and
    returns an output of the same size.

    Arguments:
        hidden_size: The hidden size of the self attention.
        num_attention_heads: number of attention head in the self attention.
        attention_dropout_prob: dropout probability of the attention score
            in self attention.
        output_dropout_prob: dropout probability for the outputs after
            self attention and final output.
        layernorm_epsilon: epsilon used in layernorm to avoid division by zero.
        mlp_activation: Activation function.
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_dropout_prob,
        output_dropout_prob,
        layernorm_epsilon,
        mlp_activation="gelu_jit",
    ):
        super().__init__()

        # Layernorm on the input data.
        self.input_layernorm = torch.nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = torch.nn.LayerNorm(
            hidden_size, eps=layernorm_epsilon
        )

        # MLP
        self.mlp = MLP(hidden_size, output_dropout_prob, activation=mlp_activation)

    def forward(self, x, layer_mask):
        """Run the layer.

        Args:
            x: Tensor batch of shape (B, sequence_length, hidden_dim)
            layer_mask: Attention mask of shape [1, 1, sequence_length, sequence_length]
        """
        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(x)

        # Self attention.
        attention_output = self.attention(
            layernorm_output,
            layer_mask,
        )

        # Residual connection.
        layernorm_input = x + attention_output

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output = self.mlp(layernorm_output)

        # Second residual connection.
        output = layernorm_input + mlp_output
        return output


class MLP(torch.nn.Module):
    """Final MLP layer in DaLL-E

    Takes the input from the last transformer layer, of shape [B, sequence_length, hidden_dim],
    project it to 4*hidden_dim, perform gelu transformation, and project the
    state back into hidden_dim dimension. At the end, dropout is also
    applied.

    Args:
        hidden_size: The hidden size of the self attention.
        output_dropout_prob: dropout probability for the outputs
                             after self attention and final output.
    """

    def __init__(self, hidden_size, output_dropout_prob, activation="gelu_jit"):
        super().__init__()

        if activation == "gelu_jit":
            self.activation = gelu_jit
        elif self.activation == "gelu":
            self.activation = gelu
        else:
            raise NotImplementedError("Used MLP activation is not implemented.")

        # Project to 4h.
        self.dense_h_to_4h = torch.nn.Linear(hidden_size, 4 * hidden_size)
        # Project back to h.
        self.dense_4h_to_h = torch.nn.Linear(4 * hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(output_dropout_prob)

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.activation(x)
        x = self.dense_4h_to_h(x)
        output = self.dropout(x)
        return output


def gelu(x):
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


@torch.jit.script
def gelu_jit(x):
    """OpenAI's gelu implementation."""
    return gelu(x)
