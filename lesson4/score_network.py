"""Implementation of Noise Conditioned Score Networks (v2).

As defined in "Improved Techniques for Training Score-Based Generative Models"
(https://arxiv.org/abs/2006.09011), and based on the original source code
from https://github.com/ermongroup/ncsnv2.
"""

import torch
from layers import InstanceNorm2dPlus, ResidualBlock, RefineBlock


class NCSNv2(torch.nn.Module):
    """A RefineNet with Timestep Conditioning and Dilated Convolutions

    Implements a Noise Conditioned Score Network (v2) as defined in
    "Improved Techniques for Training Score-Based Generative Models"
    (https://arxiv.org/abs/2006.09011), and based on the original source code
    from https://github.com/ermongroup/ncsnv2.
    """

    def __init__(self, sigmas):
        super().__init__()

        # For MNIST, the paper uses half the normal number of convolutional
        # filters (128 originally, so 64 here).
        self.num_filters = 64
        # Remember the sigmas
        self.register_buffer("sigmas", sigmas)
        # MNIST only has 1 input channel
        self.num_channels = 1
        self.act = act = torch.nn.ELU()

        self.begin_conv = torch.nn.Conv2d(
            self.num_channels, self.num_filters, 3, stride=1, padding=1
        )
        self.normalizer = InstanceNorm2dPlus(self.num_filters)

        self.end_conv = torch.nn.Conv2d(
            self.num_filters, self.num_channels, 3, stride=1, padding=1
        )

        # We use a 4 cascade RefineNet, so first setup the 4 residual layers.
        self.res1 = torch.nn.Sequential(
            ResidualBlock(
                self.num_filters,
                self.num_filters,
                resample=None,
                act=act,
                normalization=InstanceNorm2dPlus,
            ),
            ResidualBlock(
                self.num_filters,
                self.num_filters,
                resample=None,
                act=act,
                normalization=InstanceNorm2dPlus,
            ),
        )

        self.res2 = torch.nn.Sequential(
            ResidualBlock(
                self.num_filters,
                2 * self.num_filters,
                resample="down",
                act=act,
                normalization=InstanceNorm2dPlus,
            ),
            ResidualBlock(
                2 * self.num_filters,
                2 * self.num_filters,
                resample=None,
                act=act,
                normalization=InstanceNorm2dPlus,
            ),
        )

        self.res3 = torch.nn.Sequential(
            ResidualBlock(
                2 * self.num_filters,
                2 * self.num_filters,
                resample="down",
                act=act,
                normalization=InstanceNorm2dPlus,
                dilation=2,
            ),
            ResidualBlock(
                2 * self.num_filters,
                2 * self.num_filters,
                resample=None,
                act=act,
                normalization=InstanceNorm2dPlus,
                dilation=2,
            ),
        )

        self.res4 = torch.nn.Sequential(
            ResidualBlock(
                2 * self.num_filters,
                2 * self.num_filters,
                resample="down",
                act=act,
                normalization=InstanceNorm2dPlus,
                adjust_padding=True,
                dilation=4,
            ),
            ResidualBlock(
                2 * self.num_filters,
                2 * self.num_filters,
                resample=None,
                act=act,
                normalization=InstanceNorm2dPlus,
                dilation=4,
            ),
        )

        # Now match the 4 residual layers with 4 RefineNet blocks.
        self.refine1 = RefineBlock(
            [2 * self.num_filters],
            2 * self.num_filters,
            act=act,
            start=True,
        )
        self.refine2 = RefineBlock(
            [2 * self.num_filters, 2 * self.num_filters],
            2 * self.num_filters,
            act=act,
        )
        self.refine3 = RefineBlock(
            [2 * self.num_filters, 2 * self.num_filters],
            self.num_filters,
            act=act,
        )
        self.refine4 = RefineBlock(
            [self.num_filters, self.num_filters],
            self.num_filters,
            act=act,
            end=True,
        )

    def forward(self, x, y):
        # Normalize to (-1, 1)
        x = 2 * x - 1.0

        output = self.begin_conv(x)

        # Run through the residual layers
        layer1 = self.res1(output)
        layer2 = self.res2(layer1)
        layer3 = self.res3(layer2)
        layer4 = self.res4(layer3)

        # Run through the RefineNet layers
        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        output = output / used_sigmas
        return output
