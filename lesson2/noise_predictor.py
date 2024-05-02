import math
import torch

from utils import extract, generate_beta_array


class MeanAndCovarianceNetwork(torch.nn.Module):
    def __init__(
        self, num_timesteps: int, spatial_width: int = 28, input_channels: int = 1
    ):
        super().__init__()

        # Helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        # The paper used a fixed number of temporal basis functions,
        # and in the MNIST case, this was 10
        self._n_temporal_basis = 10
        self._num_timesteps = num_timesteps

        # The forward variance schedule is fixed, so precalculate it.
        register_buffer(
            "beta",
            generate_beta_array(
                num_timesteps=self._num_timesteps,
            ),
        )

        # The temporal basis bump functions
        register_buffer(
            "temporal_basis",
            generate_temporal_basis(
                num_timesteps=self._num_timesteps,
                num_temporal_basis=self._n_temporal_basis,
            ),
        )

        # The input dimensions for the dense layers
        n_input = input_channels * spatial_width**2

        # The output dimensions for the dense layers in the lower network.
        self._n_dense_lower_output = 2
        self._n_hidden_dense_lower = 500
        self._n_dense_lower = 4

        # The input/output dimensions of the dense layers in the lower
        # portion of the network.
        dense_lower_layer_dims = (
            [n_input]
            + [self._n_hidden_dense_lower] * (self._n_dense_lower - 1)
            + [self._n_dense_lower_output * spatial_width**2]
        )

        # The dense pathway from the input sample.
        self._dense_lower = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=dense_lower_layer_dims[i],
                        out_features=dense_lower_layer_dims[i + 1],
                    ),
                    torch.nn.LeakyReLU(),
                )
                for i in range(len(dense_lower_layer_dims) - 1)
            ]
        )

        # According to the paper, the convolutional channel is a downsample
        # by a factor of 2, then a convolution, then an upsample by the same factor.
        # However, this multi-scale channel was skipped for the MNIST dataset,
        # so we only have a convolution here. Note that including the multi-scale
        # channel for MNIST led to unstable training, so it is required to leave
        # out for this demonstration.
        self._n_hidden_conv = 20
        self._n_conv_lower = self._n_dense_lower
        self._conv_lower = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    # Convolution
                    torch.nn.Conv2d(
                        input_channels if idx == 0 else self._n_hidden_conv,
                        self._n_hidden_conv,
                        kernel_size=3,
                        padding=1,
                    ),
                    # Activation - Different than the paper, but
                    # present in the source code. The paper had one
                    # activation at the end of the layer set.
                    torch.nn.LeakyReLU(),
                )
                for idx in range(self._n_conv_lower)
            ]
        )

        # The upper layers are applied to each pixel independently, with the concatenated
        # input from the lower layers (dense and convolutional).
        self._n_dense_upper = 2
        self._n_hidden_dense_upper = 20

        # The output dimensions of the upper layers are a per-pixel weighting
        # for the temporal basis functions, for both mu and signma
        n_output_channels = input_channels * self._n_temporal_basis * 2
        upper_dense_layer_dims = (
            [self._n_hidden_conv + self._n_dense_lower_output]
            + [self._n_hidden_dense_upper] * (self._n_dense_upper - 1)
            + [n_output_channels]
        )
        self._dense_upper = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    # 1x1 Convolution (equivalent to pixel-wise linear tranformation)
                    torch.nn.Conv2d(
                        in_channels=upper_dense_layer_dims[i],
                        out_channels=upper_dense_layer_dims[i + 1],
                        kernel_size=1,
                    ),
                    (
                        torch.nn.LeakyReLU()
                        if i + 1 != len(upper_dense_layer_dims) - 1
                        else torch.nn.Identity()
                    ),
                )
                for i in range(len(upper_dense_layer_dims) - 1)
            ]
        )

    def forward(self, x, t):
        B, C, H, W = x.shape

        # Apply the lower convolutions
        Y_conv = x
        for layer in self._conv_lower:
            Y_conv = layer(Y_conv)

        # Apply the the lower dense layers
        Y_dense = torch.reshape(x, (B, C * H * W))
        for layer in self._dense_lower:
            Y_dense = layer(Y_dense)
        Y_dense = torch.reshape(Y_dense, (B, self._n_dense_lower_output, H, W))

        # Concatenate the dense and convolutional layers together
        Y = torch.cat(
            [
                Y_conv / math.sqrt(self._n_hidden_conv),
                Y_dense / math.sqrt(self._n_dense_lower_output),
            ],
            dim=1,
        )

        # Now run Y through the upper layers. These are the 1x1
        # pixel-wise convolutions.
        for layer in self._dense_upper:
            Y = layer(Y)

        # Convert the convolutional output (which are the weights for
        # the temporal basis "bump" functions) into the mean and covariance
        # predictions.
        mu, sigma = self._temporal_readout(x, Y, t)
        return mu, sigma

    def _temporal_readout(self, x, Y, t):
        B, C, H, W = x.shape

        # Go from the top layer of the multilayer perceptron to coefficients for
        # mu and sigma for each pixel.
        Y = Y.permute((0, 2, 3, 1))
        Y = torch.reshape(Y, (B, H, W, C, 2, self._n_temporal_basis))

        # Temporal basis bump functions
        g = self.temporal_basis.to(Y.device)
        g = torch.tile(g[None, ...], (B, 1, 1))

        # Gather the bump functions for each timestep in the batch
        coeff_weights = g.gather(
            -1, torch.tile(t[:, None, None], (1, g.shape[1], 1))
        ).squeeze()
        Z = torch.einsum("abcdef,af->abcde", Y, coeff_weights)

        # Move back to (B,C,H,W)
        Z_mu = Z[:, :, :, :, 0].permute((0, 3, 1, 2))
        Z_sigma = Z[:, :, :, :, 1].permute((0, 3, 1, 2))

        # Extract the beta coefficients for each timestep
        beta = self.beta
        beta_forward = extract(beta.to(Z_sigma.device), t, Z_sigma.shape)

        # Scale sigma according to the number of timesteps.
        # Equation 64 for Sigma
        Z_sigma_scaled = Z_sigma / math.sqrt(self._num_timesteps)
        Z_sigma = torch.nn.functional.sigmoid(
            Z_sigma_scaled + torch.log(beta_forward / (1.0 - beta_forward))
        )
        sigma = torch.sqrt(Z_sigma)

        # Equation 65 for mu (with the definition from the code, NOT the paper)
        mu = x * torch.sqrt(1.0 - beta_forward) + Z_mu * torch.sqrt(beta_forward)
        return mu, sigma


def generate_temporal_basis(num_timesteps: int, num_temporal_basis: int):
    """Generates the temporal basis "bump" functions.

    Outputs a (num_temporal_basis, num_timesteps) matrix of temporal
    basis coefficents.
    """
    # (T, B) array
    temporal_basis = torch.zeros((num_timesteps, num_temporal_basis))

    # Bump centers evenly spaced between (-1, 1)
    tau = torch.linspace(-1, 1, num_temporal_basis)

    # Widths are the spacing between bump centers.
    w = (tau[1] - tau[0]) / 2.0

    # Calculate evenly over all timesteps
    t = torch.linspace(-1, 1, num_timesteps)

    # Calculate over each basis function
    for ii in range(num_temporal_basis):
        temporal_basis[:, ii] = torch.exp(-((t - tau[ii]) ** 2) / (2 * w**2))

    # Normalize each one
    temporal_basis /= torch.sum(temporal_basis, dim=1).reshape((-1, 1))

    # Transpose from (T, B) to (B, T)
    temporal_basis = temporal_basis.T
    return temporal_basis
