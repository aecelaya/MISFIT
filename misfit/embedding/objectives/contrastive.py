"""Contrastive objective — Supervised Contrastive loss with K=2 enforcement."""
import warnings
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, Sampler

from misfit.embedding.embedding_constants import ec
from misfit.embedding.objectives.base import (
    CropFeaturesDataset,
    TrainingObjective,
)
from misfit.embedding.objectives.objective_registry import register_objective


# ---------------------------------------------------------------------------
# GroupedBatchSampler
# ---------------------------------------------------------------------------

class GroupedBatchSampler(Sampler[List[int]]):
    """Yield batches with exactly K=2 samples per group.

    Each batch contains ``M`` groups, each contributing exactly 2 indices, so
    ``batch_size = M × 2``.  Groups are shuffled every epoch; within each
    group the pair is sampled without replacement.

    Args:
        group_indices: List of lists; ``group_indices[g]`` is the list of
            dataset indices belonging to group *g* (all have ≥ 2 members).
        batch_size: Must be even (M × 2).  If not, it is rounded down to the
            nearest even number ≥ 2.
        drop_last: If ``True``, drop the final incomplete batch.
    """

    def __init__(
        self,
        group_indices: List[List[int]],
        batch_size: int,
        drop_last: bool = True,
    ) -> None:
        super().__init__()
        if batch_size < 2:
            raise ValueError("batch_size must be ≥ 2 for contrastive training.")
        # Round down to nearest even number.
        self.pairs_per_batch = batch_size // 2
        self.group_indices = group_indices
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle group order each epoch.
        group_order = torch.randperm(len(self.group_indices)).tolist()
        batch: List[int] = []

        for g in group_order:
            members = self.group_indices[g]
            # Sample exactly 2 without replacement from this group.
            perm = torch.randperm(len(members))[:2].tolist()
            batch.extend(members[i] for i in perm)

            if len(batch) // 2 == self.pairs_per_batch:
                yield batch
                batch = []

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        n_batches = len(self.group_indices) // self.pairs_per_batch
        if not self.drop_last and len(self.group_indices) % self.pairs_per_batch:
            n_batches += 1
        return n_batches


# ---------------------------------------------------------------------------
# SupCon loss
# ---------------------------------------------------------------------------

class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive loss (Khosla et al., NeurIPS 2020).

    Within a batch, all samples sharing the same integer label are treated as
    mutual positives.  The loss encourages same-label embeddings to be closer
    than cross-label embeddings in the projection space.

    Args:
        temperature: Logit scale (default ``0.07``).
    """

    def __init__(self, temperature: float = ec.SUPCON_TEMPERATURE) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, projections: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute SupCon loss.

        Args:
            projections: ``(B, D)`` ℓ²-normalised projection vectors.
            labels: ``(B,)`` integer group indices.

        Returns:
            Scalar SupCon loss.
        """
        B = projections.shape[0]
        device = projections.device

        # Cosine similarity matrix: (B, B)
        sim = torch.mm(projections, projections.T) / self.temperature

        # Numerical stability: subtract row-max.
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # Positive mask: same label, different index.
        labels_col = labels.unsqueeze(1)  # (B, 1)
        pos_mask = (labels_col == labels_col.T) & ~torch.eye(B, dtype=torch.bool, device=device)

        # Exclude self from denominator.
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)

        exp_sim = torch.exp(sim) * self_mask  # zero diagonal
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + ec.SUPCON_EPS)

        # Mean over positives for each anchor, then mean over anchors.
        n_pos = pos_mask.sum(dim=1).float()
        # Anchors with no positive in this batch contribute 0 loss.
        valid = n_pos > 0
        if not valid.any():
            return projections.sum() * 0.0  # differentiable zero

        mean_log_prob_pos = (log_prob * pos_mask).sum(dim=1) / (n_pos + ec.SUPCON_EPS)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


# ---------------------------------------------------------------------------
# ContrastiveObjective
# ---------------------------------------------------------------------------

@register_objective("contrastive")
class ContrastiveObjective(TrainingObjective):
    """Supervised contrastive training for the aggregator.

    Groups are defined by a *group column* (e.g. patient ID, diagnosis).
    Each group must have **exactly ≥ K=2** samples.  Groups with fewer
    members are dropped with a ``UserWarning``; a ``ValueError`` is raised if
    fewer than 2 valid groups remain.

    The output head is a two-layer MLP projection head (``embed_dim →
    embed_dim → PROJECTION_DIM``) with ReLU activation and ℓ²
    normalisation.  The contrastive loss operates on the projected space.
    """

    name: str = "contrastive"

    def validate_labels(
        self, labels_df: pd.DataFrame, label_col: str
    ) -> pd.DataFrame:
        """Filter groups with < K=2 samples; assign integer group indices.

        Sets ``self.label_to_idx`` as a side-effect.

        Args:
            labels_df: Raw label DataFrame with ``volume_id`` column.
            label_col: Column identifying group membership (e.g., patient ID).

        Returns:
            Cleaned DataFrame where every group has ≥ 2 members.

        Raises:
            ValueError: If fewer than 2 valid groups remain after filtering.
        """
        df = labels_df.dropna(subset=[label_col]).copy()
        df[label_col] = df[label_col].astype(str)

        group_counts = df[label_col].value_counts()
        small_groups = group_counts[group_counts < ec.MIN_SAMPLES_PER_GROUP].index.tolist()

        if small_groups:
            warnings.warn(
                f"Dropping {len(small_groups)} group(s) with fewer than "
                f"{ec.MIN_SAMPLES_PER_GROUP} samples: {small_groups[:5]}"
                + (" ..." if len(small_groups) > 5 else ""),
                UserWarning,
                stacklevel=2,
            )
            df = df[~df[label_col].isin(small_groups)]

        valid_groups = sorted(df[label_col].unique())
        if len(valid_groups) < 2:
            raise ValueError(
                f"Contrastive training requires at least 2 valid groups with "
                f"≥ {ec.MIN_SAMPLES_PER_GROUP} samples each.  Only "
                f"{len(valid_groups)} remain after filtering."
            )

        self.label_to_idx = {lbl: idx for idx, lbl in enumerate(valid_groups)}
        return df.reset_index(drop=True)

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
    ) -> Optional[BatchSampler]:
        """Build a :class:`GroupedBatchSampler` ensuring K=2 per group.

        Returns ``None`` (falls back to random) if the dataset is too small
        to form a single valid batch.
        """
        # Build per-group index lists from the dataset.
        from collections import defaultdict
        group_to_indices: dict = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            group_to_indices[label].append(idx)

        group_indices = [idxs for idxs in group_to_indices.values() if len(idxs) >= 2]

        if not group_indices:
            warnings.warn(
                "No groups with ≥ 2 cached feature files found.  "
                "Falling back to random sampling.",
                UserWarning,
                stacklevel=2,
            )
            return None

        return GroupedBatchSampler(
            group_indices=group_indices,
            batch_size=batch_size,
            drop_last=True,
        )

    def build_head(self, embed_dim: int) -> nn.Module:
        """Two-layer MLP projection head with ℓ² normalisation."""
        return nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, ec.PROJECTION_DIM),
            _L2NormLayer(),
        )

    def compute_loss(
        self, head_output: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Supervised Contrastive loss on ℓ²-normalised projections.

        Args:
            head_output: ``(B, PROJECTION_DIM)`` ℓ²-normalised projections.
            labels: ``(B,)`` integer group indices.

        Returns:
            Scalar SupCon loss.
        """
        return self._loss_fn(head_output, labels)

    def __init__(self) -> None:
        self._loss_fn = SupervisedContrastiveLoss()


class _L2NormLayer(nn.Module):
    """Thin wrapper around :func:`torch.nn.functional.normalize`."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(x, dim=-1)
