"""Ancestral sampling.

Implements ancestral sampling, from DDPM. This corresponds to Algorithm 2
from the DDPM paper.
"""

import torch
from samplers.predictor import Predictor


class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_model, probability_flow=False, continuous=True):
        super().__init__()
        self.sde = sde
        self.rsde = sde.reverse(score_model, probability_flow)
        self.score_model = score_model
        self.probability_flow = probability_flow
        self.continuous = continuous
        assert (
            not probability_flow
        ), "Probability flow not supported by ancestral sampling"

    def update(self, x, t):
        """Update function for VPSDE"""
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = sde.score(x, t, self.score_model, self.continuous)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1.0 - beta)[
            :, None, None, None
        ]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean
