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


class InstanceNorm2dPlus(torch.nn.Module):
    """Instance Normalization++ from Section B.1

    This layer extends traditional instance normalization with
    an additional mean/variance term to prevent color shifting.
    """

    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.instance_norm = torch.nn.InstanceNorm2d(
            num_features, affine=False, track_running_stats=False
        )
        self.alpha = torch.nn.Parameter(torch.zeros(num_features))
        self.gamma = torch.nn.Parameter(torch.zeros(num_features))
        self.alpha.data.normal_(1, 0.02)
        self.gamma.data.normal_(1, 0.02)
        self.beta = torch.nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        means = torch.mean(x, dim=(2, 3))
        m = torch.mean(means, dim=-1, keepdim=True)
        v = torch.var(means, dim=-1, keepdim=True)
        means = (means - m) / (torch.sqrt(v + 1e-5))
        h = self.instance_norm(x)

        h = h + means[..., None, None] * self.alpha[..., None, None]
        out = self.gamma.view(-1, self.num_features, 1, 1) * h + self.beta.view(
            -1, self.num_features, 1, 1
        )
        return out


class ConvMeanPool(torch.nn.Module):
    """Mean Pooling Layer

    Mean pooling layer combined with a convolution, used in the resnet blocks.
    """

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


class ResidualBlock(torch.nn.Module):
    """Basic ResNet block."""

    def __init__(
        self,
        input_dim,
        output_dim,
        resample=None,
        act=torch.nn.ELU(),
        normalization=InstanceNorm2dPlus,
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
                self.normalize2 = normalization(input_dim)
                self.conv2 = dilated_conv3x3(input_dim, output_dim, dilation=dilation)
                conv_shortcut = partial(dilated_conv3x3, dilation=dilation)
            else:
                self.conv1 = torch.nn.Conv2d(
                    input_dim, input_dim, 3, stride=1, padding=1
                )
                self.normalize2 = normalization(input_dim)
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
                self.normalize2 = normalization(output_dim)
                self.conv2 = dilated_conv3x3(output_dim, output_dim, dilation=dilation)
            else:
                conv_shortcut = torch.nn.Conv2d
                self.conv1 = torch.nn.Conv2d(
                    input_dim, output_dim, kernel_size=3, stride=1, padding=1
                )
                self.normalize2 = normalization(output_dim)
                self.conv2 = torch.nn.Conv2d(
                    output_dim, output_dim, kernel_size=3, stride=1, padding=1
                )
        else:
            raise Exception("invalid resample value")

        if output_dim != input_dim or resample is not None:
            self.shortcut = conv_shortcut(input_dim, output_dim)

        self.normalize1 = normalization(input_dim)

    def forward(self, x):
        output = self.normalize1(x)
        output = self.non_linearity(output)
        output = self.conv1(output)
        output = self.normalize2(output)
        output = self.non_linearity(output)
        output = self.conv2(output)

        if self.output_dim == self.input_dim and self.resample is None:
            shortcut = x
        else:
            shortcut = self.shortcut(x)

        return shortcut + output


class CRPBlock(torch.nn.Module):
    """Chained Residual Pooling (CRP) Block from RefineNet"""

    def __init__(self, features, n_stages, act=torch.nn.ReLU()):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for i in range(n_stages):
            self.convs.append(conv3x3(features, features, stride=1, bias=False))
        self.n_stages = n_stages
        self.maxpool = torch.nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.act = act

    def forward(self, x):
        x = self.act(x)
        path = x
        for i in range(self.n_stages):
            path = self.maxpool(path)
            path = self.convs[i](path)
            x = path + x
        return x


class RCUBlock(torch.nn.Module):
    """Residual Convolutional Unit (RCU) from RefineNet"""

    def __init__(self, features, n_blocks, n_stages, act=torch.nn.ReLU()):
        super().__init__()

        for i in range(n_blocks):
            for j in range(n_stages):
                setattr(
                    self,
                    "{}_{}_conv".format(i + 1, j + 1),
                    conv3x3(features, features, stride=1, bias=False),
                )

        self.stride = 1
        self.n_blocks = n_blocks
        self.n_stages = n_stages
        self.act = act

    def forward(self, x):
        for i in range(self.n_blocks):
            residual = x
            for j in range(self.n_stages):
                x = self.act(x)
                x = getattr(self, "{}_{}_conv".format(i + 1, j + 1))(x)
            x += residual
        return x


class MSFBlock(torch.nn.Module):
    """Multi-Resolution Fusion (MSF) Block from RefineNet"""

    def __init__(self, in_planes, features):
        super().__init__()
        assert isinstance(in_planes, list) or isinstance(in_planes, tuple)

        self.convs = torch.nn.ModuleList()
        self.features = features

        for i in range(len(in_planes)):
            self.convs.append(conv3x3(in_planes[i], features, stride=1, bias=True))

    def forward(self, xs, shape):
        sums = torch.zeros(xs[0].shape[0], self.features, *shape, device=xs[0].device)
        for i in range(len(self.convs)):
            h = self.convs[i](xs[i])
            h = torch.nn.functional.interpolate(
                h, size=shape, mode="bilinear", align_corners=True
            )
            sums += h
        return sums


class RefineBlock(torch.nn.Module):
    """Basic RefineNet Block."""

    def __init__(
        self,
        in_planes,
        features,
        act=torch.nn.ReLU(),
        start=False,
        end=False,
    ):
        super().__init__()

        assert isinstance(in_planes, tuple) or isinstance(in_planes, list)
        self.n_blocks = n_blocks = len(in_planes)

        self.adapt_convs = torch.nn.ModuleList()
        for i in range(n_blocks):
            self.adapt_convs.append(RCUBlock(in_planes[i], 2, 2, act))

        self.output_convs = RCUBlock(features, 3 if end else 1, 2, act)

        if not start:
            self.msf = MSFBlock(in_planes, features)

        self.crp = CRPBlock(features, 2, act)

    def forward(self, xs, output_shape):
        assert isinstance(xs, tuple) or isinstance(xs, list)
        hs = []
        for i in range(len(xs)):
            h = self.adapt_convs[i](xs[i])
            hs.append(h)

        if self.n_blocks > 1:
            h = self.msf(hs, output_shape)
        else:
            h = hs[0]

        h = self.crp(h)
        h = self.output_convs(h)

        return h
