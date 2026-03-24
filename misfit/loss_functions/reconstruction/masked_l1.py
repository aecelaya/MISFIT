"""Masked L1 reconstruction loss for MAE pretraining."""
import torch

from misfit.loss_functions.base import ReconstructionLoss
from misfit.loss_functions.loss_registry import register_loss
from misfit.loss_functions.loss_utils import check_reconstruction_inputs


@register_loss("masked_l1")
class MaskedL1Loss(ReconstructionLoss):
    """Mean absolute error computed only over masked voxels.

    The loss is the mean of absolute differences between the reconstruction
    and the original target, restricted to voxels where mask == 1. Visible
    voxels (mask == 0) do not contribute to the loss.

    L1 loss is more robust to intensity outliers than MSE, which is relevant
    in medical imaging where artifacts (CT metal, MRI bias field) can produce
    extreme voxel values. Prefer this over MaskedMSELoss when pretraining on
    heterogeneous datasets with known intensity artifacts.
    """

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked L1 loss.

        Args:
            reconstruction: Decoder output of shape (B, C, D, H, W).
            target: Original unmasked input of shape (B, C, D, H, W).
            mask: Binary mask of shape (B, 1, D, H, W); 1 = masked voxel.

        Returns:
            Scalar L1 loss over masked voxels.
        """
        check_reconstruction_inputs(reconstruction, target, mask)

        # Broadcast mask over channels: (B, 1, D, H, W) → (B, C, D, H, W).
        mask = mask.expand_as(reconstruction)
        abs_diff = (reconstruction - target).abs()
        return (abs_diff * mask).sum() / mask.sum().clamp(min=1)
