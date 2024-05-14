"""Diffusion Model for Score-SDE.

Allows different configuration of score network, SDE, and
samplers. We have implemented the variance preserving SDE below,
with ancestral sampling, which corresponds to DDPM.
"""

import torch

from sde.vpsde import VPSDE
from utils import normalize_to_neg_one_to_one

from samplers.pc import PredictorCorrectorSampler
from samplers.ancestral import AncestralSamplingPredictor


class GaussianDiffusion_ScoreSDE(torch.nn.Module):
    def __init__(self, score_network_type, config):
        super().__init__()

        self._score_network = score_network_type(config)
        self._sde = VPSDE(
            beta_min=config.model.beta_min,
            beta_max=config.model.beta_max,
            N=config.model.num_scales,
        )
        self._loss_function = self._sde.get_sde_loss_fn(
            config.training.reduce_mean,
            config.training.continuous,
            config.training.likelihood_weighting,
        )
        self._sampler = PredictorCorrectorSampler(
            predictor=AncestralSamplingPredictor(
                sde=self._sde,
                score_model=self._score_network,
                probability_flow=config.sampling.probability_flow,
                continuous=config.training.continuous,
            ),
            corrector=None,
            n_steps=config.sampling.n_steps_each,
            denoise=True,
            continuous=config.training.continuous,
        )

    def loss_on_batch(self, data: torch.Tensor):
        # Center the data
        data = normalize_to_neg_one_to_one(data)
        return self._loss_function(self._score_network, data)

    def sample(self, image_size: int, num_channels: int, batch_size: int):
        # Use the device that the current model is on.
        # Assumes all of the parameters are on the same device
        device = next(self.parameters()).device

        samples = self._sampler.sample(
            self._sde,
            self._score_network,
            device,
            (batch_size, num_channels, image_size, image_size),
        )
        return samples
