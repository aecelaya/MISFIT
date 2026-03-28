"""Attention-pool aggregator — lightweight, trainable."""
from typing import Optional

import torch
import torch.nn as nn

from misfit.embedding.aggregators.base import AbstractAggregator
from misfit.embedding.aggregators.aggregator_registry import register_aggregator


@register_aggregator("attention_pool")
class AttentionPoolAggregator(AbstractAggregator):
    """Single learned query attending over the crop sequence.

    A single learnable query vector ``q ∈ ℝᶜ`` computes scaled dot-product
    attention over the ``N`` crop keys.  Padded crops are masked out before
    softmax so they contribute nothing to the output.

    Optionally adds a linear projection of the normalised 3-D crop position
    to the keys before attention, allowing the aggregator to weight crops
    differently based on where they are in the volume (e.g., apical lung
    vs. basal lung in a chest CT).

    Args:
        embed_dim: Dimensionality ``C`` of each crop embedding.
        use_position_encoding: If ``True`` (default), learn a linear
            projection from 3-D normalised coordinates to ``ℝᶜ`` that is
            added to the keys before computing attention weights.
    """

    def __init__(self, embed_dim: int, use_position_encoding: bool = True) -> None:
        super().__init__(embed_dim)
        self.query = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.scale = embed_dim ** -0.5

        self.pos_proj: Optional[nn.Linear] = (
            nn.Linear(3, embed_dim, bias=False) if use_position_encoding else None
        )

    def forward(
        self,
        crop_features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attended global embedding.

        Args:
            crop_features: ``(N, C)`` per-crop embeddings.
            positions: ``(N, 3)`` normalised crop coordinates.  Required
                when ``use_position_encoding=True``.
            padding_mask: ``(N,)`` boolean mask, ``True`` = padded crop
                that should be ignored.

        Returns:
            ``(C,)`` attended global embedding.
        """
        keys = crop_features  # (N, C)

        if self.pos_proj is not None and positions is not None:
            keys = keys + self.pos_proj(positions)  # (N, C)

        # Scaled dot-product attention: query (1, C) × keys (C, N) → (1, N)
        attn_logits = (self.query @ keys.T) * self.scale  # (1, N)

        if padding_mask is not None:
            # Set logits for padded crops to -inf before softmax.
            attn_logits = attn_logits.masked_fill(
                padding_mask.unsqueeze(0), float("-inf")
            )

        attn_weights = torch.softmax(attn_logits, dim=-1)  # (1, N)
        return (attn_weights @ keys).squeeze(0)             # (C,)
