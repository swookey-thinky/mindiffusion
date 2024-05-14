"""Base defintition of SDE class.

This class is based on the authors original implementation at:
https://github.com/yang-song/score_sde_pytorch/blob/main/sde_lib.py#L7
"""

import abc
import torch


class SDE(abc.ABC):
    """SDE abstract class."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
          N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        raise RuntimeError(
            f"Derived class ({type(self).__name__}) failed to implement T"
        )

    @abc.abstractmethod
    def sde(self, x, t):
        """Calculates the drift coefficient $f(x,t)$ and the diffusion coefficient $g(t)$.

        Child classes return $f(x,t)$ and $g(t)$ from Eq. 5
        in the authors paper.

        Args:
            x: Tensor batch of data of shape (B, C, H, W).
            t: Tensor batch of continuous time, as torch.float32. In the
               range [0, self.T]

        Returns:
            Tuple of (drift, diffusion).
        """
        raise RuntimeError(
            f"Derived class ({type(self).__name__}) failed to implement sde"
        )

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x)$.

        Since the marginal distribution $p_t(x)$ in a diffusion model is
        constrained to be a Gaussian distribution, this function returns
        the mean and std deviation of the prior distribution at time t.

        Args:
            x: Tensor batch of data of shape (B, C, H, W).
            t: Tensor batch of continuous time, as torch.float32. In the
               range [0, self.T]

        Returns:
            Tensor batch of shape (B, C, H, W) describing the mean and std deviation
            at each pixel, of the prior distribution $p_t(x)$.
        """
        raise RuntimeError(
            f"Derived class ({type(self).__name__}) failed to implement marginal_prob"
        )

    @abc.abstractmethod
    def prior_sampling(self, shape):
        """Generate one sample from the prior distribution, $p_T(x)$.

        Args:
            x: Tensor batch of data of shape (B, C, H, W).
            t: Tensor batch of continuous time, as torch.float32

        Returns:
            Tensor batch of shape (B, C, H, W) sampling from the prior
            distribution $p_T(x)$.

        """
        raise RuntimeError(
            f"Derived class ({type(self).__name__}) failed to implement prior_sampling"
        )

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
          z: latent code

        Returns:
          log probability density
        """
        raise RuntimeError(
            f"Derived class ({type(self).__name__}) failed to implement prior_logp"
        )

    @abc.abstractmethod
    def score(self, x, t, score_model, continuous):
        """Calculates the score function of the SDE, $/grad{/log{p_t(x)}}$.

        Args:
            x: Tensor batch of data of shape (B, C, H, W).
            t: Tensor batch of continuous time, as torch.float32
            score_model: The score model to use with the SDE.
            continuous: True if this is a continuous time SDE.

        Returns:
            Tensor batch of shape (B, C, H, W) sampling from the prior
            distribution $p_T(x)$.

        """
        raise RuntimeError(
            f"Derived class ({type(self).__name__}) failed to implement score"
        )

    def discretize(self, x, t):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
          x: a torch tensor
          t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
          f, G
        """
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, score_model, probability_flow=False, continuous=True):
        """Create the reverse-time SDE/ODE.

        Returns the reverse time SDE for this class based on Eq. 6.

        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize
        score_fn = self.score

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                """Create the drift and diffusion functions for the reverse SDE/ODE.

                Args:
                    x: Tensor batch of data of shape (B, C, H, W).
                    t: Tensor batch of continuous time, as torch.float32. In the
                    range [0, self.T]

                Returns:
                    Tuple of (drift, diffusion).
                """
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t, score_model, continuous)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                """Create discretized iteration rules for the reverse diffusion sampler.

                Returns f and g from Eq. 6.

                Args:
                    x: a torch tensor
                    t: a torch float representing the time step (from 0 to `self.T`)

                Returns:
                    f, G
                """
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(
                    x, t, score_model, continuous
                ) * (0.5 if self.probability_flow else 1.0)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    def get_sde_loss_fn(
        self, reduce_mean=True, continuous=True, likelihood_weighting=True, eps=1e-5
    ):
        """Create a loss function for training with arbirary SDEs.

        Args:
            reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
            continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
                ad-hoc interpolation to take continuous time steps.
            likelihood_weighting: If `True`, weight the mixture of score matching losses
                according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
            eps: A `float` number. The smallest time step to sample from.

        Returns:
        A loss function.
        """
        reduce_op = (
            torch.mean
            if reduce_mean
            else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
        )

        def loss_fn(score_model, batch):
            """Compute the loss function.

            Args:
            model: A score model.
            batch: A mini-batch of training data.

            Returns:
             loss: A scalar that represents the average loss value across the mini-batch.
            """
            t = torch.rand(batch.shape[0], device=batch.device) * (self.T - eps) + eps
            z = torch.randn_like(batch)
            mean, std = self.marginal_prob(batch, t)
            perturbed_data = mean + std[:, None, None, None] * z
            score = self.score(perturbed_data, t, score_model, continuous)

            if not likelihood_weighting:
                losses = torch.square(score * std[:, None, None, None] + z)
                losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            else:
                g2 = self.sde(torch.zeros_like(batch), t)[1] ** 2
                losses = torch.square(score + z / std[:, None, None, None])
                losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

            loss = torch.mean(losses)
            return loss

        return loss_fn
