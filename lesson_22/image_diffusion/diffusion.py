from abc import abstractmethod
import torch
from typing import Callable, Dict, List, Optional, Tuple
from typing_extensions import Self

from image_diffusion.utils import DotConfig


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
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        pass

    @abstractmethod
    def print_model_summary(self):
        pass

    @abstractmethod
    def configure_optimizers(self, learning_rate: float) -> List[torch.optim.Optimizer]:
        pass

    @abstractmethod
    def models(self) -> List[Self]:
        pass

    @abstractmethod
    def config(self) -> DotConfig:
        pass
