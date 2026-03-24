"""Mean ensembler for MISFIT inference."""
from typing import List

import torch

from misfit.inference.ensemblers.base import AbstractEnsembler
from misfit.inference.ensemblers.ensembler_registry import register_ensembler


@register_ensembler("mean")
class MeanEnsembler(AbstractEnsembler):
    """Element-wise mean over a list of prediction tensors."""

    def combine(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """Average *predictions* element-wise.

        Args:
            predictions: Non-empty list of tensors with identical shapes.

        Returns:
            Element-wise mean tensor.

        Raises:
            ValueError: If *predictions* is empty.
        """
        if not predictions:
            raise ValueError("MeanEnsembler requires at least one prediction.")
        return torch.stack(predictions, dim=0).mean(dim=0)
