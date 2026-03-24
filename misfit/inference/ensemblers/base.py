"""Abstract base class for all MISFIT prediction ensemblers."""
from abc import ABC, abstractmethod
from typing import Any, List

import torch


class AbstractEnsembler(ABC):
    """Abstract base class for aggregating a list of prediction tensors.

    An ensembler combines multiple predictions — from different TTA
    augmentations or (in future) different checkpoints — into a single
    output tensor.
    """

    def __init__(self) -> None:
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def combine(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate *predictions* into a single tensor.

        Args:
            predictions: List of tensors, each of the same shape.

        Returns:
            Aggregated tensor of the same shape as each input.
        """

    def __call__(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        return self.combine(predictions)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AbstractEnsembler) and self.name == other.name
