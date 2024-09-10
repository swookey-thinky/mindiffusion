from xdiffusion.sde import SDE


class RectifiedFlow(SDE):
    def __init__(self, N, T, **kwargs):
        super().__init__()
        self._N = N
        self._T = T

    def T(self) -> int:
        return self._T

    def N(self) -> int:
        return self._N

    def ode(self, intial_input, diffusion_model, reverse: bool = False):
        raise NotImplemented()

    def z0(self, x):
        raise NotImplemented()

    def sigma_t(self, t: float):
        return 0.0

    def noise_scale(self) -> float:
        return 1.0
