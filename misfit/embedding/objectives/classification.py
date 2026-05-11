"""Classification objective — cross-entropy loss with a linear head."""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler

from misfit.embedding.objectives.base import (
    CropFeaturesDataset,
    TrainingObjective,
)
from misfit.embedding.objectives.objective_registry import register_objective


@register_objective("classification")
class ClassificationObjective(TrainingObjective):
    """Supervised classification with cross-entropy loss.

    Expects integer or string labels in *label_col*.  String labels are
    mapped to contiguous integer indices sorted lexicographically.

    The output head is a single linear layer: ``embed_dim → num_classes``.
    Standard :class:`~torch.utils.data.RandomSampler` is used — no special
    batch structure is required for classification.
    """

    name: str = "classification"

    def validate_labels(
        self, labels_df: pd.DataFrame, label_col: str
    ) -> pd.DataFrame:
        """Drop rows with missing labels and build ``label_to_idx`` mapping.

        Sets ``self.label_to_idx`` and ``self.num_classes`` as side-effects.

        Args:
            labels_df: Raw label DataFrame with ``volume_id`` column.
            label_col: Column to use as the training target.

        Returns:
            Cleaned DataFrame with no null labels.

        Raises:
            ValueError: If no valid samples remain after cleaning.
        """
        df = labels_df.dropna(subset=[label_col]).copy()
        if df.empty:
            raise ValueError(
                f"No valid samples found in label column '{label_col}'."
            )

        unique_labels = sorted(df[label_col].astype(str).unique())
        self.label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        # Normalise the label column to string so the mapping applies cleanly.
        df[label_col] = df[label_col].astype(str)
        return df

    def build_dataset(
        self,
        labels_df: pd.DataFrame,
        label_col: str,
    ) -> CropFeaturesDataset:
        return CropFeaturesDataset(
            labels_df=labels_df,
            label_col=label_col,
            label_to_idx=self.label_to_idx,
        )

    def build_batch_sampler(
        self, dataset: CropFeaturesDataset, batch_size: int
    ) -> BatchSampler | None:
        """Return ``None`` — standard random batching is sufficient."""
        return None

    def build_head(self, embed_dim: int) -> nn.Module:
        """Linear projection from ``embed_dim`` to ``num_classes``."""
        return nn.Linear(embed_dim, self.num_classes)

    def compute_loss(
        self, head_output: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy loss.

        Args:
            head_output: ``(B, num_classes)`` logits.
            labels: ``(B,)`` integer class indices.

        Returns:
            Scalar cross-entropy loss.
        """
        return nn.functional.cross_entropy(head_output, labels)
