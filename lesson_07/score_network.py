"""DDPM++ model from Score-SDE.

This package implements the DDPM++ model from the paper
"Score-Based Generative Modeling through Stochastic Differential Equations"
(https://arxiv.org/abs/2011.13456). This model has five improvements
over the DDPM model introduced in "Denoising Diffusion Probabilistic Models"
(https://arxiv.org/abs/2006.11239).

(Improvements listed in Appendix H)

1.) Upsampling and downsampling images with anti-aliasing based on
    Finite Impulse Response (FIR) from "Making convolutional networks shift-invariant again."
    (https://arxiv.org/abs/1904.11486). They follow the same implementation and
    hyper-parameters in StyleGAN-2 from "Analyzing and Improving the Image Quality of StyleGAN"
    (https://arxiv.org/abs/1912.04958).
2.) Rescaling all skip connections by $\frac{1}{\sqrt{2}}$.
3.) Replacing the original residual blocks in DDPM with residual blocks from the BigGAN
    paper "Large scale gan training for high fidelity natural image synthesis."
    (https://arxiv.org/abs/1809.11096).
4.) Increasing the number of residual blocks per resolution from 2 to 4.
5.) Incorporating progressive growing architectures. We consider two progressive
    architectures for input: “input skip” and “residual”, and two progressive
    architectures for output: “output skip” and “residual".

However, for CIFAR-10, with DDPM++, they authors used no FIR, and no progessive upsampling
(Appendix H.2) We will include all of the above improvements for demonstration, but limit
the instantiation to the DDPM++ configuration.
"""

import functools
import numpy as np
import torch
from typing import Dict

from layers import blocks
import utils


class NCSNpp(torch.nn.Module):
    """Base architecture for NSCN++"""

    def __init__(self, config: utils.DotConfig):
        super().__init__()

        self._config = config
        self.register_buffer("_sigmas", torch.tensor(utils.get_sigmas(config)))

        channel_multipliers = config.model.channel_multipliers
        self._num_resolutions = len(channel_multipliers)
        self._all_resolutions = [
            config.data.image_size // (2**i) for i in range(self._num_resolutions)
        ]

        # Validate some of the parameter values for embeddings and progressive
        # growing.
        assert config.model.progressive_output in ["none", "output_skip", "residual"]
        assert config.model.progressive_input in ["none", "input_skip", "residual"]
        assert config.model.embedding_type in ["fourier", "positional"]
        assert config.model.resnet_block_type in ["biggan", "ddpm"]
        assert config.model.progressive_combine in ["sum", "cat"]

        combine_method = config.model.progressive_combine
        combiner = functools.partial(blocks.Combine, method=combine_method)

        # Keep track of all of the modules we are creating, in order
        modules = []

        # Timestep/noise_level embedding; only for continuous training
        if config.model.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            assert (
                config.training.continuous
            ), "Fourier features are only used for continuous training."

            modules.append(
                blocks.GaussianFourierProjection(
                    embedding_size=config.model.num_features,
                    scale=config.model.fourier_scale,
                )
            )
            embed_dim = 2 * config.model.num_features

        elif config.model.embedding_type == "positional":
            embed_dim = config.model.num_features

        else:
            raise ValueError(f"embedding type {config.model.embedding_type} unknown.")

        # Project the timestep into the embedding space, using initialization
        # from DDPM paper.
        modules.append(torch.nn.Linear(embed_dim, config.model.num_features * 4))
        modules[-1].weight.data = blocks.default_init()(modules[-1].weight.shape)
        torch.nn.init.zeros_(modules[-1].bias)
        modules.append(blocks.get_activation(config))
        modules.append(
            torch.nn.Linear(
                config.model.num_features * 4, config.model.num_features * 4
            )
        )
        modules[-1].weight.data = blocks.default_init()(modules[-1].weight.shape)
        torch.nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(
            blocks.AttnBlockpp,
            init_scale=config.model.init_scale,
            skip_rescale=config.model.skip_rescale,
        )

        Upsample = functools.partial(
            blocks.Upsample,
            with_conv=config.model.resamp_with_conv,
            fir=config.model.fir,
            fir_kernel=config.model.fir_kernel,
        )

        if config.model.progressive_output == "output_skip":
            self._pyramid_upsample = blocks.Upsample(
                fir=config.model.fir,
                fir_kernel=config.model.fir_kernel,
                with_conv=False,
            )
        elif config.model.progressive_output == "residual":
            pyramid_upsample = functools.partial(
                blocks.Upsample,
                fir=config.model.fir,
                fir_kernel=config.models.fir_kernel,
                with_conv=True,
            )

        Downsample = functools.partial(
            blocks.Downsample,
            with_conv=config.model.resamp_with_conv,
            fir=config.model.fir,
            fir_kernel=config.model.fir_kernel,
        )

        if config.model.progressive_input == "input_skip":
            self._pyramid_downsample = blocks.Downsample(
                fir=config.model.fir,
                fir_kernel=config.model.fir_kernel,
                with_conv=False,
            )
        elif config.model.progressive_input == "residual":
            pyramid_downsample = functools.partial(
                blocks.Downsample,
                fir=config.model.fir,
                fir_kernel=config.model.fir_kernel,
                with_conv=True,
            )

        if config.model.resnet_block_type == "ddpm":
            ResnetBlock = functools.partial(
                blocks.ResnetBlockDDPM,
                dropout=config.model.dropout,
                init_scale=config.model.init_scale,
                skip_rescale=config.model.skip_rescale,
                temb_dim=config.model.num_features * 4,
            )

        elif config.model.resnet_block_type == "biggan":
            ResnetBlock = functools.partial(
                blocks.ResnetBlockBigGAN,
                dropout=config.model.dropout,
                fir=config.model.fir,
                fir_kernel=config.model.fir_kernel,
                init_scale=config.model.init_scale,
                skip_rescale=config.model.skip_rescale,
                temb_dim=config.model.num_features * 4,
            )

        else:
            raise ValueError(
                f"resblock type {config.model.resnet_block_type} unrecognized."
            )

        # Downsampling blocks

        channels = config.data.num_channels
        if config.model.progressive_input != "none":
            input_pyramid_ch = channels

        # Initial convolution
        modules.append(blocks.ddpm_conv3x3(channels, config.model.num_features))

        # Keep track of the channels at each layer
        hs_channels = [config.model.num_features]

        in_ch = config.model.num_features
        for i_level in range(self._num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(config.model.num_resnet_blocks):
                out_ch = config.model.num_features * channel_multipliers[i_level]
                modules.append(
                    ResnetBlock(
                        in_ch=in_ch,
                        out_ch=out_ch,
                        act=blocks.get_activation(config),
                    )
                )
                in_ch = out_ch

                if self._all_resolutions[i_level] in config.model.attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_channels.append(in_ch)

            if i_level != self._num_resolutions - 1:
                if config.model.resnet_block_type == "ddpm":
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(
                        ResnetBlock(
                            down=True,
                            in_ch=in_ch,
                            act=blocks.get_activation(config),
                        )
                    )

                if config.model.progressive_input == "input_skip":
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == "cat":
                        in_ch *= 2

                elif config.model.progressive_input == "residual":
                    modules.append(
                        pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch)
                    )
                    input_pyramid_ch = in_ch

                hs_channels.append(in_ch)

        # Middle layers
        in_ch = hs_channels[-1]
        modules.append(
            ResnetBlock(
                in_ch=in_ch,
                act=blocks.get_activation(config),
            )
        )
        modules.append(AttnBlock(channels=in_ch))
        modules.append(
            ResnetBlock(
                in_ch=in_ch,
                act=blocks.get_activation(config),
            )
        )

        # Upsampling layers
        pyramid_channels = 0
        for i_level in reversed(range(self._num_resolutions)):
            for i_block in range(config.model.num_resnet_blocks + 1):
                out_ch = config.model.num_features * channel_multipliers[i_level]
                modules.append(
                    ResnetBlock(
                        in_ch=in_ch + hs_channels.pop(),
                        out_ch=out_ch,
                        act=blocks.get_activation(config),
                    )
                )
                in_ch = out_ch

            if self._all_resolutions[i_level] in config.model.attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if config.model.progressive_output != "none":
                if i_level == self._num_resolutions - 1:
                    if config.model.progressive_output == "output_skip":
                        modules.append(blocks.get_activation(config))
                        modules.append(
                            torch.nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            blocks.ddpm_conv3x3(
                                in_ch, channels, init_scale=config.model.init_scale
                            )
                        )
                        pyramid_channels = channels
                    elif config.model.progressive_output == "residual":
                        modules.append(blocks.get_activation(config))
                        modules.append(
                            torch.nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(blocks.ddpm_conv3x3(in_ch, in_ch, bias=True))
                        pyramid_channels = in_ch
                    else:
                        raise ValueError(
                            f"{config.model.progressive_output} is not a valid name."
                        )
                else:
                    if config.model.progressive_output == "output_skip":
                        modules.append(blocks.get_activation(config))
                        modules.append(
                            torch.nn.GroupNorm(
                                num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch,
                                eps=1e-6,
                            )
                        )
                        modules.append(
                            blocks.ddpm_conv3x3(
                                in_ch,
                                channels,
                                bias=True,
                                init_scale=config.model.init_scale,
                            )
                        )
                        pyramid_channels = channels
                    elif config.model.progressive_output == "residual":
                        modules.append(
                            pyramid_upsample(in_ch=pyramid_channels, out_ch=in_ch)
                        )
                        pyramid_channels = in_ch
                    else:
                        raise ValueError(
                            f"{config.model.progressive_output} is not a valid name"
                        )

            if i_level != 0:
                if config.model.resnet_block_type == "ddpm":
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(
                        ResnetBlock(
                            in_ch=in_ch,
                            up=True,
                            act=blocks.get_activation(config),
                        )
                    )

        # Make sure we used up all of our channels
        assert not hs_channels

        if config.model.progressive_output != "output_skip":
            modules.append(blocks.get_activation(config))
            modules.append(
                torch.nn.GroupNorm(
                    num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6
                )
            )
            modules.append(
                blocks.ddpm_conv3x3(in_ch, channels, init_scale=config.model.init_scale)
            )
        self._all_modules = torch.nn.ModuleList(modules)

    def forward(self, x, time_cond):
        # timestep/noise_level embedding; only for continuous training
        modules = self._all_modules
        # print(modules)

        m_idx = 0
        if self._config.model.embedding_type == "fourier":
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas))
            m_idx += 1

        elif self._config.model.embedding_type == "positional":
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self._sigmas[time_cond.long()]
            temb = blocks.get_timestep_embedding(
                timesteps, self._config.model.num_features
            )

        else:
            raise ValueError(
                f"embedding type {self._config.model.embedding_type} unknown."
            )

        # Embed the timestep
        temb = modules[m_idx](temb)
        m_idx += 1

        act = modules[m_idx]
        m_idx += 1
        temb = modules[m_idx](act(temb))
        m_idx += 1

        # Downsampling block
        input_pyramid = None
        if self._config.model.progressive_input != "none":
            input_pyramid = x

        # Initial convolution
        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self._num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self._config.model.num_resnet_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self._config.model.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self._num_resolutions - 1:
                if self._config.model.resnet_block_type == "ddpm":
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1

                if self._config.model.progressive_input == "input_skip":
                    input_pyramid = self._pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self._config.model.progressive_input == "residual":
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self._config.model.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.0)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid

                hs.append(h)

        # Middle layers
        h = hs[-1]
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self._num_resolutions)):
            for i_block in range(self._config.model.num_resnet_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1

            if h.shape[-1] in self._config.model.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self._config.model.progressive_output != "none":
                if i_level == self._num_resolutions - 1:
                    if self._config.model.progressive_output == "output_skip":
                        act = modules[m_idx]
                        m_idx += 1
                        pyramid = act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    elif self._config.model.progressive_output == "residual":
                        act = modules[m_idx]
                        m_idx += 1
                        pyramid = act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(
                            f"{self._config.model.progressive_output} is not a valid name."
                        )
                else:
                    if self._config.model.progressive_output == "output_skip":
                        pyramid = self._pyramid_upsample(pyramid)
                        act = modules[m_idx]
                        m_idx += 1
                        pyramid_h = act(modules[m_idx](h))
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self._config.model.progressive_output == "residual":
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self._config.model.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.0)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(
                            f"{self._config.model.progressive_output} is not a valid name"
                        )

            if i_level != 0:
                if self._config.model.resnet_block_type == "ddpm":
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, temb)
                    m_idx += 1

        assert not hs

        if self._config.model.progressive_output == "output_skip":
            h = pyramid
        else:
            final_act = modules[m_idx]
            m_idx += 1
            h = final_act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules)
        if self._config.model.scale_by_sigma:
            used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
            h = h / used_sigmas

        return h
