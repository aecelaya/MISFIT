"""Base class for all MISFIT reconstruction losses."""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module, ABC):
    """Abstract base class for MISFIT MAE reconstruction losses.

    All reconstruction losses receive a decoder output, the original
    unmasked volume, and a binary mask indicating which voxels were masked
    during encoding. Subclasses must implement forward() and compute the
    loss only over masked voxels.

    Input convention (enforced by loss_utils.check_reconstruction_inputs):
        reconstruction: (B, C, D, H, W) — decoder output, continuous values.
        target:         (B, C, D, H, W) — original input volume.
        mask:           (B, 1, D, H, W) — binary, 1 = masked, 0 = visible.
    """

    @abstractmethod
    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reconstruction loss over masked voxels.

        Args:
            reconstruction: Decoder output of shape (B, C, D, H, W).
            target: Original unmasked input of shape (B, C, D, H, W).
            mask: Binary mask of shape (B, 1, D, H, W) where 1 indicates
                a masked (held-out) voxel and 0 indicates a visible voxel.

        Returns:
            Scalar loss tensor.
        """
