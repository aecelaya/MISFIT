"""Masked MSE reconstruction loss for MAE pretraining."""
import torch

from misfit.loss_functions.base import ReconstructionLoss
from misfit.loss_functions.loss_registry import register_loss
from misfit.loss_functions.loss_utils import check_reconstruction_inputs


@register_loss("masked_mse")
class MaskedMSELoss(ReconstructionLoss):
    """Mean squared error computed only over masked voxels.

    The loss is the mean of squared differences between the reconstruction
    and the original target, restricted to voxels where mask == 1. Visible
    voxels (mask == 0) do not contribute to the loss.

    This is the standard reconstruction objective from the original MAE paper
    (He et al., 2022).
    """

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked MSE loss.

        Args:
            reconstruction: Decoder output of shape (B, C, D, H, W).
            target: Original unmasked input of shape (B, C, D, H, W).
            mask: Binary mask of shape (B, 1, D, H, W); 1 = masked voxel.

        Returns:
            Scalar MSE loss over masked voxels.
        """
        check_reconstruction_inputs(reconstruction, target, mask)

        # Broadcast mask over channels: (B, 1, D, H, W) → (B, C, D, H, W).
        mask = mask.expand_as(reconstruction)
        diff_sq = (reconstruction - target) ** 2
        return (diff_sq * mask).sum() / mask.sum().clamp(min=1)
