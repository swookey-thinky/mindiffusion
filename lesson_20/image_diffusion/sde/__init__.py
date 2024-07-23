from abc import abstractmethod


class SDE:

    @abstractmethod
    def T(self) -> int:
        pass

    @abstractmethod
    def N(self) -> int:
        pass

    @abstractmethod
    def ode(self, intial_input, diffusion_model, reverse: bool = False):
        pass

    @abstractmethod
    def z0(self, x):
        pass

    @abstractmethod
    def sigma_t(self, t: float):
        pass

    @abstractmethod
    def noise_scale(self) -> float:
        pass
