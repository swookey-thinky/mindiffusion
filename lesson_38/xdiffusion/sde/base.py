from abc import abstractmethod
import torch


class SDE:

    @property
    @abstractmethod
    def T(self):
        """End time of the SDE."""
        raise RuntimeError(
            f"Derived class ({type(self).__name__}) failed to implement T"
        )

    @property
    @abstractmethod
    def N(self):
        """The number of discretization steps."""
        raise RuntimeError(
            f"Derived class ({type(self).__name__}) failed to implement N"
        )

    @abstractmethod
    def sde(self, x, context):
        """Calculates the drift coefficient $f(x,t)$ and the diffusion coefficient $g(t)$.

        Child classes return $f(x,t)$ and $g(t)$ from Eq. 5 in the Score-SDE paper.

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

    @abstractmethod
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

    @abstractmethod
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

    def discretize(self, x, context):
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
        drift, diffusion = self.sde(x, context)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=x.device))
        return f, G

    def reverse(self, score_fn, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Uses Eq. 6 and 13 from Score-SDE to create the reverse time SDE.

        Args:
            score_fn: A time-dependent score-based model that takes x and t and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        # Build the class for reverse-time SDE.
        class RSDE(self.__class__):
            """A reverse time SDE, based on "Reverse-time diffusion equation models" [Anderson (1982)]

            Eq. 6 from Score-SDE.
            """

            def __init__(self, n, t):
                self._probability_flow = probability_flow
                self._n = n
                self._t = t

            @property
            def T(self):
                return self._t

            @property
            def N(self):
                return self._n

            def sde(self, x, context):
                """Create the drift and diffusion functions for the reverse SDE/ODE.

                Eq. 6 from Score-SDE. The probability flow version is Eq. 13 in Score-SDE.
                """
                drift, diffusion = sde_fn(x, context)
                score = score_fn(x, context)
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (
                    0.5 if self._probability_flow else 1.0
                )
                # Set the diffusion function to zero for ODEs.
                diffusion = 0.0 if self._probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, context):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, context)
                rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, context) * (
                    0.5 if self._probability_flow else 1.0
                )
                rev_G = torch.zeros_like(G) if self._probability_flow else G
                return rev_f, rev_G

        return RSDE(N, T)
