"""Interface for Corrector component of PC samplers.

Defines the Corrector interface from Section 4.2 of Score-SDE. Based on the
original implementation at:
https://github.com/yang-song/score_sde_pytorch/blob/main/sampling.py#L151
"""

import abc


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    @abc.abstractmethod
    def update(self, x, t):
        """One update of the corrector.

        Args:
          x: Tensor batch representing the current state, of shape (B, C, H, W)
          t: Tensor batch representing the current time step.

        Returns:
          x: Tensor batch of the next state.
          x_mean: Tensor batch of the next state without random noise. Useful for denoising.
        """
        raise RuntimeError(
            f"Derived class ({type(self).__name__}) failed to implement update."
        )
