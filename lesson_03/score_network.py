"""Implementation of Noise Conditioned Score Networks.

As defined in "Generative Modeling by Estimating Gradients of the Data Distribution"
(https://arxiv.org/abs/1907.05600), and based on the original source code
from https://github.com/ermongroup/ncsn.
"""

import torch

from layers import (
    ConditionalInstanceNorm2dPlus,
    ConditionalResidualBlock,
    CondRefineBlock,
)


class NCSN(torch.nn.Module):
    """Noise Conditioned Score Network

    Implements a Noise Conditioned Score Network as defined in
    "Generative Modeling by Estimating Gradients of the Data Distribution"
    (https://arxiv.org/abs/1907.05600), and based on the original source code
    from https://github.com/ermongroup/ncsn.
    """

    def __init__(self, num_variances: int = 10):
        super().__init__()

        self.norm = ConditionalInstanceNorm2dPlus
        self.ngf = ngf = 64
        self.num_variances = num_variances
        self.num_channels = 1
        self.act = act = torch.nn.ELU()

        self.begin_conv = torch.nn.Conv2d(
            self.num_channels, ngf, 3, stride=1, padding=1
        )
        self.normalizer = self.norm(ngf, self.num_variances)
        self.end_conv = torch.nn.Conv2d(ngf, self.num_channels, 3, stride=1, padding=1)

        # The main network consists of 4 residual blocks
        # followed by 4 refinenet blocks.
        self.res1 = torch.nn.ModuleList(
            [
                ConditionalResidualBlock(
                    self.ngf,
                    self.ngf,
                    self.num_variances,
                    resample=None,
                    act=act,
                    normalization=self.norm,
                ),
                ConditionalResidualBlock(
                    self.ngf,
                    self.ngf,
                    self.num_variances,
                    resample=None,
                    act=act,
                    normalization=self.norm,
                ),
            ]
        )

        self.res2 = torch.nn.ModuleList(
            [
                ConditionalResidualBlock(
                    self.ngf,
                    2 * self.ngf,
                    self.num_variances,
                    resample="down",
                    act=act,
                    normalization=self.norm,
                ),
                ConditionalResidualBlock(
                    2 * self.ngf,
                    2 * self.ngf,
                    self.num_variances,
                    resample=None,
                    act=act,
                    normalization=self.norm,
                ),
            ]
        )

        self.res3 = torch.nn.ModuleList(
            [
                ConditionalResidualBlock(
                    2 * self.ngf,
                    2 * self.ngf,
                    self.num_variances,
                    resample="down",
                    act=act,
                    normalization=self.norm,
                    dilation=2,
                ),
                ConditionalResidualBlock(
                    2 * self.ngf,
                    2 * self.ngf,
                    self.num_variances,
                    resample=None,
                    act=act,
                    normalization=self.norm,
                    dilation=2,
                ),
            ]
        )

        self.res4 = torch.nn.ModuleList(
            [
                ConditionalResidualBlock(
                    2 * self.ngf,
                    2 * self.ngf,
                    self.num_variances,
                    resample="down",
                    act=act,
                    normalization=self.norm,
                    adjust_padding=True,
                    dilation=4,
                ),
                ConditionalResidualBlock(
                    2 * self.ngf,
                    2 * self.ngf,
                    self.num_variances,
                    resample=None,
                    act=act,
                    normalization=self.norm,
                    dilation=4,
                ),
            ]
        )

        # Define the 4 refinenet blocks.
        self.refine1 = CondRefineBlock(
            [2 * self.ngf],
            2 * self.ngf,
            self.num_variances,
            self.norm,
            act=act,
            start=True,
        )
        self.refine2 = CondRefineBlock(
            [2 * self.ngf, 2 * self.ngf],
            2 * self.ngf,
            self.num_variances,
            self.norm,
            act=act,
        )
        self.refine3 = CondRefineBlock(
            [2 * self.ngf, 2 * self.ngf],
            self.ngf,
            self.num_variances,
            self.norm,
            act=act,
        )
        self.refine4 = CondRefineBlock(
            [self.ngf, self.ngf],
            self.ngf,
            self.num_variances,
            self.norm,
            act=act,
            end=True,
        )

    def _compute_cond_module(self, module, x, y):
        for m in module:
            x = m(x, y)
        return x

    def forward(self, x, y):
        # Normalize to (-1, 1)
        x = 2 * x - 1.0

        output = self.begin_conv(x)

        layer1 = self._compute_cond_module(self.res1, output, y)
        layer2 = self._compute_cond_module(self.res2, layer1, y)
        layer3 = self._compute_cond_module(self.res3, layer2, y)
        layer4 = self._compute_cond_module(self.res4, layer3, y)

        ref1 = self.refine1([layer4], y, layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], y, layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], y, layer2.shape[2:])
        output = self.refine4([layer1, ref3], y, layer1.shape[2:])

        output = self.normalizer(output, y)
        output = self.act(output)
        output = self.end_conv(output)
        return output
