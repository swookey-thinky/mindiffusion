"""Utility layers used in defining a Noise Conditioned Score Network."""

import torch
from functools import partial


def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """Helper function for 3x3 convolution with padding

    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        stride: Convolutional stride, default to 1.
        bias: True to use bias, False otherwise.

    Returns
        Instance of torch.nn.Conv2d.
    """
    return torch.nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )


def dilated_conv3x3(in_planes, out_planes, dilation, bias=True):
    """Helper function for 3x3 convolution with dilation.

    Args:
        in_planes: Number of input channels.
        out_planes: Number of output channels.
        dilation: Size of the convolutional dilation.
        bias: True to use bias, False otherwise.

    Returns
        Instance of torch.nn.Conv2d.
    """
    return torch.nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        padding=dilation,
        dilation=dilation,
        bias=bias,
    )


class ConvMeanPool(torch.nn.Module):
    """Performs a convolution followed by Mean Pooling."""

    def __init__(
        self, input_dim, output_dim, kernel_size=3, biases=True, adjust_padding=False
    ):
        super().__init__()
        if not adjust_padding:
            self.conv = torch.nn.Conv2d(
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=biases,
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.ZeroPad2d((1, 0, 1, 0)),
                torch.nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=biases,
                ),
            )

    def forward(self, inputs):
        output = self.conv(inputs)
        output = (
            sum(
                [
                    output[:, :, ::2, ::2],
                    output[:, :, 1::2, ::2],
                    output[:, :, ::2, 1::2],
                    output[:, :, 1::2, 1::2],
                ]
            )
            / 4.0
        )
        return output


class ConditionalInstanceNorm2dPlus(torch.nn.Module):
    """Conditional Instance Normalization++ from Section A.1

    This layer extends traditional instance normalization with
    support for the timestep conditional, as well as an additional
    mean/variance term to prevent color shifting.
    """

    def __init__(self, num_features, num_variances):
        super().__init__()
        self.num_features = num_features

        # Instance norm calculates the original formulation with gamma and beta set
        # to 1 and 0, since affine=False
        self.instance_norm = torch.nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )

        # Embedding calculates a vector for the integer timestep. We have one vector
        # each for alpha, gamma, and beta.
        self.embed = torch.nn.Embedding(num_variances, num_features * 3)
        self.embed.weight.data[:, : 2 * num_features].normal_(
            1, 0.02
        )  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, 2 * num_features :].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        # Calculate the mean of x over the spatial dimensions
        mu = torch.mean(x, dim=(2, 3))

        # Calculate the mean and stdev of mu
        m = torch.mean(mu, dim=-1, keepdim=True)
        v = torch.sqrt(torch.var(mu, dim=-1, keepdim=True) + 1e-5)

        # Calculate the coefficient for alpha
        alpha_coeff = (mu - m) / v

        # Calculate the coefficient for gamma
        gamma_coeff = self.instance_norm(x)

        # Pull alpha, gamma, and beta from the embedding on the conditioning.
        gamma, alpha, beta = self.embed(y).chunk(3, dim=-1)

        z = (
            gamma.view(-1, self.num_features, 1, 1) * gamma_coeff
            + beta.view(-1, self.num_features, 1, 1)
            + alpha_coeff[..., None, None] * alpha[..., None, None]
        )
        return z


class ConditionalResidualBlock(torch.nn.Module):
    """Basic residual block from ResNet.

    Extends the basic residual block with support for passing
    along the conditional information.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_variances,
        resample=None,
        act=torch.nn.ELU(),
        normalization=ConditionalInstanceNorm2dPlus,
        adjust_padding=False,
        dilation=None,
    ):
        super().__init__()
        self.non_linearity = act
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.resample = resample
        if resample == "down":
            if dilation is not None:
                self.conv1 = dilated_conv3x3(input_dim, input_dim, dilation=dilation)
                self.normalize2 = normalization(input_dim, num_variances)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
            else:
                self.conv1 = torch.nn.Conv2d(
                    input_dim, input_dim, 3, stride=1, padding=1
                )
                self.normalize2 = normalization(input_dim, num_variances)
                self.conv2 = ConvMeanPool(
                    input_dim, output_dim, 3, adjust_padding=adjust_padding
                )
                conv_shortcut = partial(
                    ConvMeanPool, kernel_size=1, adjust_padding=adjust_padding
                )

        elif resample is None:
            if dilation is not None:
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
                self.conv1 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                self.normalize2 = normalization(output_dim, num_variances)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = torch.nn.Conv2d
                self.conv1 = torch.nn.Conv2d(
                    input_dim, output_dim, kernel_size=3, stride=1, padding=1
                )
                self.normalize2 = normalization(output_dim, num_variances)
                self.conv2 = torch.nn.Conv2d(
                    output_dim, output_dim, kernel_size=3, stride=1, padding=1
                )
        else:
            raise Exception("invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim, num_variances)

    def forward(self, x, y):
        output = self.normalize1(x, y)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output, y)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class CondCRPBlock(torch.nn.Module):
    """Chained Residual Pooling (CRP) Block from RefineNet.

    Extends the RefineNet CRP Block with support for passing along
    the conditional information.
    """

    def __init__(
        self, features, n_stages, num_variances, normalizer, act=torch.nn.ReLU()
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for i in range(n_stages):
            self.norms.append(normalizer(features, num_variances))
            self.convs.append(conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        self.maxpool = torch.nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x, y):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.norms[i](path, y)
            path = self.maxpool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class CondRCUBlock(torch.nn.Module):
    """Residual Convolutional Unit (RCU) from RefineNet.

    Extends the RefineNet RCU block with the conditional timestep
    information.
    """

    def __init__(
        self,
        features,
        n_blocks,
        n_stages,
        num_variances,
        normalizer,
        act=torch.nn.ReLU(),
    ):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(
                    self,
                    "{}_{}_norm".format(i + 1, j + 1),
                    normalizer(features, num_variances),
                )
                setattr(
                    self,
                    "{}_{}_conv".format(i + 1, j + 1),
                    conv3x3(features, features, stride=1, bias=False),
                )

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x, y):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = getattr(self, "{}_{}_norm".format(i + 1, j + 1))(x, y)
                x = self.act(x)
                x = getattr(self, "{}_{}_conv".format(i + 1, j + 1))(x)
            x += residual
        return x


class CondMSFBlock(torch.nn.Module):
    """Multi-Resolution Fusion (MSF) Block from RefineNet.

    Extends the RefineNet MSF Block with the conditional timestep information.
    """

    def __init__(self, in_planes, features, num_variances, normalizer):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True))
            self.norms.append(normalizer(in_planes[i], num_variances))

    def forward(self, xs, y, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.norms[i](xs[i], y)
            h = self.convs[i](h)
            h = torch.nn.functional.interpolate(
                h, size=shape, mode="bilinear", align_corners=True
            )
            sums += h
        return sums


class CondRefineBlock(torch.nn.Module):
    """Basic RefineNet block.

    Extends the basic RefineNet block with the conditional timestep
    information.

    RefineNet's were introduced in the paper "RefineNet: Multi-Path Refinement
    Networks for High-Resolution Semantic Segmentation" (https://arxiv.org/abs/1611.06612).
    """

    def __init__(
        self,
        in_planes,
        features,
        num_variances,
        normalizer,
        act=torch.nn.ReLU(),
        start=False,
        end=False,
    ):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = torch.nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(
                CondRCUBlock(in_planes[i], 2, 2, num_variances, normalizer, act)
            )

        self.output_convs = CondRCUBlock(
            features, 3 if end else 1, 2, num_variances, normalizer, act
        )

        if not start:
            self.msf = CondMSFBlock(in_planes, features, num_variances, normalizer)

        self.crp = CondCRPBlock(features, 2, num_variances, normalizer, act)

    def forward(self, xs, y, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i], y)
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, y, output_shape)
        else:
            h = hs[0]

        h = self.crp(h, y)
        h = self.output_convs(h, y)

        return h
