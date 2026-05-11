"""Mean-pool aggregator — zero-shot, no training required."""

import torch

from misfit.embedding.aggregators.aggregator_registry import register_aggregator
from misfit.embedding.aggregators.base import AbstractAggregator


@register_aggregator("mean_pool")
class MeanPoolAggregator(AbstractAggregator):
    """Average crop embeddings element-wise, ignoring padded crops.

    Requires no training — usable immediately from a pretrained encoder.
    Good baseline for retrieval, clustering, and outlier detection.
    """

    def forward(
        self,
        crop_features: torch.Tensor,
        positions: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the mean of all valid (non-padded) crop embeddings.

        Args:
            crop_features: ``(N, C)`` per-crop embeddings.
            positions: Ignored — mean pool is position-agnostic.
            padding_mask: ``(N,)`` boolean mask, ``True`` = padded.

        Returns:
            ``(C,)`` mean embedding.
        """
        if padding_mask is not None:
            valid = ~padding_mask  # (N,)
            return crop_features[valid].mean(dim=0)
        return crop_features.mean(dim=0)
