import torch
from tqdm import tqdm
from typing import Optional

from samplers.corrector import Corrector
from samplers.predictor import Predictor

from utils import unnormalize_to_zero_to_one


class PredictorCorrectorSampler:
    """Implements Algorithm 1 of the Score-SDE paper."""

    def __init__(
        self,
        predictor: Optional[Predictor],
        corrector: Optional[Corrector],
        n_steps=1,
        denoise=False,
        continuous=True,
        rtol=1e-5,
        atol=1e-5,
        method="RK45",
        epsilon=1e-3,
    ):
        self._denoise = denoise
        self._rtol = rtol
        self._atol = atol
        self._method = method
        self._epsilon = epsilon
        self._continuous = continuous
        self._n_steps = n_steps
        self._predictor = predictor
        self._corrector = corrector

    def sample(self, sde, score_model, device, sample_shape, z=None):
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(sample_shape).to(device)
            timesteps = torch.linspace(sde.T, self._epsilon, sde.N, device=device)

            for i in tqdm(range(sde.N), total=sde.N, leave=False):
                t = timesteps[i]
                batched_t = torch.ones(sample_shape[0], device=t.device) * t

                if self._predictor is not None:
                    x, x_mean = self._predictor.update(x, batched_t)

                if self._corrector is not None:
                    for _ in range(self._n_steps):
                        x, x_mean = self._corrector.update(x, batched_t)

            x = unnormalize_to_zero_to_one(x_mean if self._denoise else x)
            x = x.to("cpu")
            return x
