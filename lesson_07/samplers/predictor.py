"""Interface for Predictor component of PC samplers.

Defines the Predictor interface from Section 4.2 of Score-SDE. Based on the
original implementation at:
https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py#L126
"""

import abc


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    @abc.abstractmethod
    def update(self, x, t):
        """One update of the predictor.

        Args:
          x: Tensor batch representing the current state, shape (B, C, H, W)
          t: Tensor batch representing the current time step.

        Returns:
          x: Tensor batch of the next state.
          x_mean: Tensor batch of the next state without random noise. Useful for denoising.
        """
        raise RuntimeError(
            f"Derived class ({type(self).__name__}) failed to implement update."
        )
