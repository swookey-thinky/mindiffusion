"""Discrete VAE implementation.

DALL*E implementation of a discrete VAE. This implementation is based
on the original DaLL-E Discrete VAE implementation here:

https://github.com/openai/DALL-E
"""

from collections import OrderedDict
from einops import rearrange
import math
import torch
from typing import Optional

from utils import unmap_pixels


class EncoderBlock(torch.nn.Module):
    """Basic encoder block."""

    def __init__(self, n_in: int, n_out: int, n_layers: int):
        """Initialize a new instance of EncoderBlock.

        Args:
            n_in: The number of input channels for the block.
            n_out: The number of output channels for the block.
            n_layers: The total number of layers in the VAE.
        """
        super().__init__()

        hidden_size = n_out // 4
        self._post_gain = 1 / (n_layers**2)

        self._residual_path = (
            torch.nn.Conv2d(n_in, n_out, kernel_size=1)
            if n_in != n_out
            else torch.nn.Identity()
        )
        self._resnet_path = torch.nn.Sequential(
            OrderedDict(
                [
                    ("relu_1", torch.nn.ReLU()),
                    (
                        "conv_1",
                        torch.nn.Conv2d(n_in, hidden_size, kernel_size=3, padding=1),
                    ),
                    ("relu_2", torch.nn.ReLU()),
                    (
                        "conv_2",
                        torch.nn.Conv2d(
                            hidden_size, hidden_size, kernel_size=3, padding=1
                        ),
                    ),
                    ("relu_3", torch.nn.ReLU()),
                    (
                        "conv_3",
                        torch.nn.Conv2d(
                            hidden_size, hidden_size, kernel_size=3, padding=1
                        ),
                    ),
                    ("relu_4", torch.nn.ReLU()),
                    ("conv_4", torch.nn.Conv2d(hidden_size, n_out, 1)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._residual_path(x) + self._post_gain * self._resnet_path(x)


class DecoderBlock(torch.nn.Module):
    """Basic decoder block."""

    def __init__(self, n_in: int, n_out: int, n_layers: int):
        """Initialize a new instance of DecoderBlock.

        Args:
            n_in: The number of input channels for the block.
            n_out: The number of output channels for the block.
            n_layers: The total number of layers in the VAE.
        """
        super().__init__()

        hidden_size = n_out // 4
        self._post_gain = 1 / (n_layers**2)

        self._residual_path = (
            torch.nn.Conv2d(n_in, n_out, kernel_size=1)
            if n_in != n_out
            else torch.nn.Identity()
        )
        self._resnet_path = torch.nn.Sequential(
            OrderedDict(
                [
                    ("relu_1", torch.nn.ReLU()),
                    (
                        "conv_1",
                        torch.nn.Conv2d(n_in, hidden_size, kernel_size=1),
                    ),
                    ("relu_2", torch.nn.ReLU()),
                    (
                        "conv_2",
                        torch.nn.Conv2d(
                            hidden_size, hidden_size, kernel_size=3, padding=1
                        ),
                    ),
                    ("relu_3", torch.nn.ReLU()),
                    (
                        "conv_3",
                        torch.nn.Conv2d(
                            hidden_size, hidden_size, kernel_size=3, padding=1
                        ),
                    ),
                    ("relu_4", torch.nn.ReLU()),
                    (
                        "conv_4",
                        torch.nn.Conv2d(hidden_size, n_out, kernel_size=3, padding=1),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._residual_path(x) + self._post_gain * self._resnet_path(x)


class Encoder(torch.nn.Module):
    """Encoder module of the VAE."""

    def __init__(
        self,
        num_groups: int = 4,
        hidden_size: int = 256,
        num_blocks_per_group: int = 2,
        input_channels: int = 3,
        vocab_size: int = 8192,
    ):
        """Initialize a new instance of Encoder.

        Args:
            num_groups: The number of groups in the encoder.
            hidden_size: The dimensions of the blocks in the encoder.
            num_blocks: The number of encoder blocks in each group.
            input_channels: The number of channels in the input.
            vocab_size: Vocabulary size of input tokens.
        """
        super().__init__()

        num_layers = num_groups * num_blocks_per_group
        self._vocab_size = vocab_size

        # Padding is always: (kernel_size - 1) // 2)
        self._blocks = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "input",
                        torch.nn.Conv2d(
                            input_channels,
                            1 * hidden_size,
                            kernel_size=7,
                            padding=(7 - 1) // 2,
                        ),
                    ),
                ]
                + [
                    (
                        f"group_{group_idx}",
                        torch.nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            EncoderBlock(
                                                (
                                                    (2 ** (group_idx - 1)) * hidden_size
                                                    if group_idx > 0 and i == 0
                                                    else (
                                                        1 * hidden_size
                                                        if group_idx == 0
                                                        else (2**group_idx)
                                                        * hidden_size
                                                    )
                                                ),
                                                (2**group_idx) * hidden_size,
                                                n_layers=num_layers,
                                            ),
                                        )
                                        for i in range(num_blocks_per_group)
                                    ],
                                    (
                                        "pool",
                                        (
                                            torch.nn.MaxPool2d(kernel_size=2)
                                            if group_idx < (num_groups - 1)
                                            else torch.nn.Identity()
                                        ),
                                    ),
                                ]
                            )
                        ),
                    )
                    for group_idx in range(num_groups)
                ]
                + [
                    (
                        "output",
                        torch.nn.Sequential(
                            OrderedDict(
                                [
                                    ("relu", torch.nn.ReLU()),
                                    (
                                        "conv",
                                        torch.nn.Conv2d(
                                            (2 ** (num_groups - 1)) * hidden_size,
                                            vocab_size,
                                            kernel_size=1,
                                        ),
                                    ),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder a tensor into a discrete codebook.

        Args:
            x: Tensor batch of image data, of shape (B, input_channels, H, W)

        Returns:
            Discrete encoding of image data, of shape (B, vocab_size, H, W).
        """
        z_logits = self(x)
        z = torch.argmax(z_logits, dim=1)
        z = (
            torch.nn.functional.one_hot(z, num_classes=self._vocab_size)
            .permute(0, 3, 1, 2)
            .float()
        )
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._blocks(x)


class Decoder(torch.nn.Module):
    def __init__(
        self,
        num_groups: int = 4,
        hidden_size: int = 256,
        num_init: int = 128,
        num_blocks_per_group: int = 2,
        output_channels: int = 3,
        vocab_size: int = 8192,
    ):
        """Initialize a new instance of Decoder.

        Args:
            num_groups: The number of groups in the decoder.
            hidden_size: The dimensions of the blocks in the decoder.
            num_init: The number of output channels in the initial convolution.
            num_blocks: The number of decoder blocks in each group.
            output_channels: The number of channels in the output.
            vocab_size: Vocabulary size of input tokens.
        """
        super().__init__()

        num_layers = num_groups * num_blocks_per_group
        self._output_channels = output_channels

        # Padding is always: (kernel_size - 1) // 2)
        self._blocks = torch.nn.Sequential(
            OrderedDict(
                [
                    (
                        "input",
                        torch.nn.Conv2d(
                            vocab_size,
                            num_init,
                            kernel_size=1,
                        ),
                    ),
                ]
                + [
                    (
                        f"group_{group_idx}",
                        torch.nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            DecoderBlock(
                                                (
                                                    num_init
                                                    if group_idx == (num_groups - 1)
                                                    and i == 0
                                                    else (
                                                        (2**group_idx) * hidden_size
                                                        if group_idx == (num_groups - 1)
                                                        else (
                                                            (2 ** (group_idx + 1))
                                                            * hidden_size
                                                            if i == 0
                                                            else (2**group_idx)
                                                            * hidden_size
                                                        )
                                                    )
                                                ),
                                                (2**group_idx) * hidden_size,
                                                n_layers=num_layers,
                                            ),
                                        )
                                        for i in range(num_blocks_per_group)
                                    ],
                                    (
                                        "upsample",
                                        (
                                            torch.nn.Upsample(
                                                scale_factor=2, mode="nearest"
                                            )
                                            if group_idx > 0
                                            else torch.nn.Identity()
                                        ),
                                    ),
                                ]
                            )
                        ),
                    )
                    for group_idx in reversed(range(num_groups))
                ]
                + [
                    (
                        "output",
                        torch.nn.Sequential(
                            OrderedDict(
                                [
                                    ("relu", torch.nn.ReLU()),
                                    (
                                        "conv",
                                        torch.nn.Conv2d(
                                            1 * hidden_size,
                                            2 * output_channels,
                                            kernel_size=1,
                                        ),
                                    ),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes images logits into pixel data.

        Args:
            z: Tensor batch of image logits, of shape (B, vocab_size, H, W)

        Returns:
            Tensor batch of image data, of shape (B, output_channels, H, W)
        """
        x_stats = self(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, : self._output_channels]))
        return x_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._blocks(x)


class DiscreteVAE(torch.nn.Module):
    """Discrete VAE used in DaLL-E."""

    def __init__(
        self,
        num_groups: int = 4,
        hidden_size: int = 256,
        num_blocks_per_group: int = 2,
        input_channels: int = 3,
        vocab_size: int = 8192,
    ):
        """Initialize a new instance of DiscreteVAE.

        Args:
            num_groups: The number of groups in the encoder/decoder.
            hidden_size: The dimensions of the blocks in the encoder/decoder.
            num_blocks_per_group: The number of blocks in each encoder/decoder group.
            input_channels: The number of channels in the input.
            vocab_size: Vocabulary size of input tokens.
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.num_groups = num_groups
        self._temperature = 0.9
        self._kl_div_loss_weight = 0.0

        self.encoder = Encoder(
            num_groups=num_groups,
            hidden_size=hidden_size,
            num_blocks_per_group=num_blocks_per_group,
            input_channels=input_channels,
            vocab_size=vocab_size,
        )

        self.decoder = Decoder(
            num_groups=num_groups,
            hidden_size=hidden_size,
            num_init=hidden_size // 2,
            num_blocks_per_group=num_blocks_per_group,
            output_channels=input_channels,
            vocab_size=vocab_size,
        )

    def get_codebook_indices(self, images) -> torch.Tensor:
        """Encodes image data into discrete codebook indices.

        Encodes an image into a set of discrete indices (image tokens) which
        can be passed to the DaLL-E model.

        Args:
            images: Tensor batch of image data, of shape (B, input_channels, H, W)

        Returns:
            Tensor batch of discrete codebook indices, of shape (B, 1, H, W)
        """
        logits = self.encoder(images)
        return logits.argmax(dim=1).flatten(1)

    def decode(self, img_seq):
        """Decodes a sequence of image tokens.

        Args:
            img_seq: Sequence of image_tokens, of shape (B, num_tokens)

        Returns:
            Tensor batch of image data.
        """
        b, n = img_seq.shape
        one_hot_indices = torch.nn.functional.one_hot(
            img_seq, num_classes=self.vocab_size
        ).float()
        z = rearrange(one_hot_indices, "b (h w) c -> b c h w", h=int(math.sqrt(n)))
        img = self.decoder.decode(z)
        return img

    def forward(
        self,
        x: torch.Tensor,
        temperature: Optional[float] = None,
        kl_weight: Optional[float] = None,
    ):
        logits = self.encoder(x)

        tau = temperature if temperature is not None else self._temperature
        one_hot = torch.nn.functional.gumbel_softmax(logits, tau=tau, dim=1, hard=True)
        sampled = self.decoder.decode(one_hot)

        # Reconstruction loss
        assert x.shape[1] == sampled.shape[1], f"{x.shape} {sampled.shape}"
        # reconstruction_loss = torch.nn.functional.mse_loss(x, sampled)
        reconstruction_loss = torch.nn.functional.mse_loss(x, sampled)

        # KL divergence
        logits = rearrange(logits, "b n h w -> b (h w) n")
        log_qy = torch.nn.functional.log_softmax(logits, dim=-1)
        log_uniform = torch.log(torch.tensor([1.0 / self.vocab_size], device=x.device))
        kl_div = torch.nn.functional.kl_div(
            log_uniform, log_qy, None, None, "batchmean", log_target=True
        )

        loss = reconstruction_loss + (kl_div * self._kl_div_loss_weight)
        loss = loss / x.shape[1] * x.shape[2] * x.shape[3]
        return loss, sampled
