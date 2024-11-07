"""Utilities for importance sampling during trainig."""

from abc import abstractmethod
import numpy as np
import torch


class ScheduleSampler:
    """Base class for all timestep schedule samplers.

    This class defines the interface for creating a new timestep scheduler.
    The original DDPM paper used a uniform distribution of timesteps during
    training, each with equal weights. Improved DDPM introduced the notion of
    importance sampling into the the schedule creation during training. This
    distribution over timesteps is intended to reduce variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged. However, subclasses may override sample()
    to change how the resampled terms are reweighted, allowing for actual
    changes in the objective.
    """

    @abstractmethod
    def weights(self) -> np.ndarray:
        """Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive. The weights
        will be applied to the total loss at the end of each training step
        and before applying backpropagation.
        """

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        Args:
            ts: Tensor batch of int timesteps.
            losses: Tensor batch of float losses, one per timestep.
        """

    def sample(self, batch_size, device):
        """Importance-sample timesteps for a batch.

        Args:
            batch_size: The size of the batch to sample.
            device: The torch device to save to.
        Returns:
            A tuple (timesteps, weights):
                timesteps: Tensor batch of timestep indices.
                weights: Tensor batch of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    """Uniform sampler for timestes.

    Equivalent to generating timesteps using torch.randint().
    """

    def __init__(self, num_timesteps):
        super().__init__()
        self._weights = np.ones([num_timesteps])

    def weights(self):
        return self._weights

    def update_with_all_losses(self, ts, losses):
        return


class ImportanceSampler(ScheduleSampler):
    """Importance sampler over timesteps."""

    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([num_timesteps], dtype=np.int64)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()
