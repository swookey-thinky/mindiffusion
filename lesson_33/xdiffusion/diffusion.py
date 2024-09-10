from abc import abstractmethod
from enum import Enum
import torch
from typing import Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Self

from xdiffusion.samplers.base import ReverseProcessSampler
from xdiffusion.scheduler import NoiseScheduler
from xdiffusion.utils import DotConfig
from xdiffusion.sde import SDE


class PredictionType(Enum):
    EPSILON = "epsilon"
    V = "v"
    RECTIFIED_FLOW = "rectified_flow"


class DiffusionModel(torch.nn.Module):

    @abstractmethod
    def loss_on_batch(self, images: torch.Tensor, context: Dict) -> Dict:
        pass

    @abstractmethod
    def sample(
        self,
        context: Optional[Dict] = None,
        num_samples: int = 16,
        guidance_fn: Optional[Callable] = None,
        classifier_free_guidance: Optional[float] = None,
        sampler: Optional[ReverseProcessSampler] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        pass

    @abstractmethod
    def print_model_summary(self):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        pass

    @abstractmethod
    def configure_optimizers(self, learning_rate: float) -> List[torch.optim.Optimizer]:
        pass

    @abstractmethod
    def configure_learning_rate_schedule(
        self, optimizers: List[torch.optim.Optimizer]
    ) -> List[torch.optim.lr_scheduler._LRScheduler]:
        pass

    @abstractmethod
    def models(self) -> List[Self]:
        pass

    @abstractmethod
    def config(self) -> DotConfig:
        pass

    @abstractmethod
    def process_input(self, x: torch.Tensor, context: Dict) -> torch.Tensor:
        pass

    @abstractmethod
    def predict_score(
        self, x: torch.Tensor, context: Dict
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass

    @abstractmethod
    def is_learned_sigma(self) -> bool:
        pass

    @abstractmethod
    def noise_scheduler(self) -> NoiseScheduler:
        pass

    @abstractmethod
    def classifier_free_guidance(self) -> float:
        pass

    @abstractmethod
    def prediction_type(self) -> PredictionType:
        pass

    @abstractmethod
    def sde(self) -> Optional[SDE]:
        """Gets the SDE associated with this diffusion model, if it exists."""
        pass
