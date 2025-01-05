import torch

from xdiffusion.autoencoders.base import VariationalAutoEncoder
from xdiffusion.autoencoders.layers import Encoder, Decoder
from xdiffusion.autoencoders.distributions import DiagonalGaussianDistribution
from xdiffusion.utils import DotConfig, instantiate_from_config


class AutoencoderKL(torch.nn.Module, VariationalAutoEncoder):
    def __init__(
        self,
        config: DotConfig,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
    ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**config.encoder_decoder_config._cfg)
        self.decoder = Decoder(**config.encoder_decoder_config._cfg)
        self.loss = instantiate_from_config(config.loss_config._cfg)

        assert config.encoder_decoder_config["double_z"]
        self.quant_conv = torch.nn.Conv2d(
            2 * config.encoder_decoder_config["z_channels"], 2 * config.embed_dim, 1
        )
        self.post_quant_conv = torch.nn.Conv2d(
            config.embed_dim, config.encoder_decoder_config["z_channels"], 1
        )
        self.embed_dim = config.embed_dim
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["model_state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def encode_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into latents."""
        encoder_posterior = self.encode(x)
        z = encoder_posterior.sample().detach()
        return z

    def decode_from_latents(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes latents into images."""
        return self.decode(z)

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        return x

    def training_step(self, batch, batch_idx, optimizer_idx, global_step):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            return aeloss, reconstructions, posterior

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                optimizer_idx,
                global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            return discloss, reconstructions, posterior

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )
        return self.log_dict

    def configure_optimizers(self, learning_rate):
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=learning_rate,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight
