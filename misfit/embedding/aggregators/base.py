"""Abstract base class for MISFIT crop aggregators."""
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class AbstractAggregator(nn.Module, ABC):
    """Aggregates a variable-length sequence of crop embeddings into one vector.

    Subclasses receive per-crop feature vectors (already global-average-pooled
    from the encoder bottleneck) and optionally their 3-D positions within the
    original volume, and return a single embedding representing the whole image.

    Args:
        embed_dim: Dimensionality ``C`` of each crop feature vector.
    """

    name: str  # Unique identifier used as registry key.

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    @abstractmethod
    def forward(
        self,
        crop_features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Aggregate crop features into a single volume embedding.

        Args:
            crop_features: ``(N, C)`` tensor of per-crop embeddings.
            positions: ``(N, 3)`` tensor of normalised 3-D crop coordinates
                in ``[0, 1]``, or ``None`` if position encoding is disabled.
            padding_mask: ``(N,)`` boolean tensor where ``True`` marks padded
                (invalid) crops that should be ignored, or ``None`` if all
                crops are valid.

        Returns:
            ``(C,)`` global volume embedding.
        """
