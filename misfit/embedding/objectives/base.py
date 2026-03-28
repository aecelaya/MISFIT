"""Abstract base class and shared dataset for MISFIT embedding objectives."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, Dataset


class CropFeaturesDataset(Dataset):
    """Dataset of pre-extracted per-crop feature files.

    Each ``.npz`` file corresponds to one volume and contains two arrays:

    - ``features``: ``(N_crops, C)`` float32 — per-crop embeddings produced
      by global-average-pooling the encoder bottleneck.
    - ``positions``: ``(N_crops, 3)`` float32 — normalised 3-D crop
      coordinates in ``[0, 1]``.

    Args:
        labels_df: DataFrame with at least ``features_path`` and one label
            column.  ``features_path`` must be the absolute path to the
            ``.npz`` file for each volume.
        label_col: Name of the label column to use.
        label_to_idx: Optional mapping from raw label value to integer index.
            If ``None``, raw values are used as-is (must already be integers).
    """

    def __init__(
        self,
        labels_df: pd.DataFrame,
        label_col: str,
        label_to_idx: Optional[dict] = None,
    ) -> None:
        self.label_to_idx = label_to_idx

        self.samples: List[Tuple[Path, Any]] = []
        for _, row in labels_df.iterrows():
            npz_path = Path(row["features_path"])
            if not npz_path.exists():
                continue
            label = row[label_col]
            if label_to_idx is not None:
                label = label_to_idx[label]
            self.samples.append((npz_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        npz_path, label = self.samples[idx]
        data = np.load(npz_path)
        features  = torch.from_numpy(data["features"]).float()   # (N, C)
        positions = torch.from_numpy(data["positions"]).float()  # (N, 3)
        return features, positions, label


def crop_collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length crop sequences with zero-padding.

    Returns:
        features:     ``(B, N_max, C)``
        positions:    ``(B, N_max, 3)``
        padding_mask: ``(B, N_max)`` — ``True`` where padded
        labels:       ``(B,)``
    """
    features_list, positions_list, labels = zip(*batch)
    max_n = max(f.shape[0] for f in features_list)
    C = features_list[0].shape[1]

    B = len(features_list)
    padded_features  = torch.zeros(B, max_n, C)
    padded_positions = torch.zeros(B, max_n, 3)
    padding_mask     = torch.ones(B, max_n, dtype=torch.bool)  # True = padded

    for i, (feats, pos) in enumerate(zip(features_list, positions_list)):
        n = feats.shape[0]
        padded_features[i, :n]  = feats
        padded_positions[i, :n] = pos
        padding_mask[i, :n]     = False

    return padded_features, padded_positions, padding_mask, torch.tensor(labels)


class TrainingObjective(ABC):
    """Abstract base class for embedding training objectives.

    Each concrete objective knows how to:
    - validate and clean the label DataFrame for its specific semantics,
    - build a :class:`CropFeaturesDataset`,
    - supply a custom :class:`~torch.utils.data.BatchSampler` (or ``None``),
    - build the output head that sits on top of the aggregator,
    - compute the training loss.
    """

    name: str

    @abstractmethod
    def validate_labels(
        self, labels_df: pd.DataFrame, label_col: str
    ) -> pd.DataFrame:
        """Validate and clean *labels_df* for this objective.

        Args:
            labels_df: Raw label DataFrame with a ``volume_id`` column.
            label_col: Column to use as the training target.

        Returns:
            Cleaned DataFrame ready for :meth:`build_dataset`.

        Raises:
            ValueError: If the labels are incompatible with this objective.
        """

    @abstractmethod
    def build_dataset(
        self, labels_df: pd.DataFrame, label_col: str
    ) -> CropFeaturesDataset:
        """Build the training dataset."""

    @abstractmethod
    def build_batch_sampler(
        self, dataset: CropFeaturesDataset, batch_size: int
    ) -> Optional[BatchSampler]:
        """Return a custom :class:`~torch.utils.data.BatchSampler` or ``None``."""

    @abstractmethod
    def build_head(self, embed_dim: int) -> nn.Module:
        """Build the output head that maps ``(B, C)`` embeddings to loss inputs."""

    @abstractmethod
    def compute_loss(
        self, head_output: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute the scalar training loss."""
