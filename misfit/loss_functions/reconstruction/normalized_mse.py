"""Normalized masked MSE reconstruction loss for MAE pretraining.

Implements the per-patch normalization variant from the original MAE paper
(He et al., 2022), adapted for 3D medical volumes. Before computing MSE,
the target voxel intensities are normalized within each masked patch (zero
mean, unit variance). This removes absolute intensity scale from the
reconstruction task, which is especially important for multi-modality
medical imaging where CT (Hounsfield units) and MRI (arbitrary units) have
incompatible scales.

Reference:
    He, K. et al. (2022). Masked Autoencoders Are Scalable Vision Learners.
    CVPR 2022. https://arxiv.org/abs/2111.06377
"""
import torch

from misfit.loss_functions.base import ReconstructionLoss
from misfit.loss_functions.loss_registry import register_loss
from misfit.loss_functions.loss_utils import check_reconstruction_inputs
from misfit.utils.normalization import normalize_patchwise


@register_loss("normalized_masked_mse")
class NormalizedMaskedMSELoss(ReconstructionLoss):
    """Per-patch normalized MSE computed only over masked voxels.

    For each 3D patch in the mask grid, the target voxel intensities are
    normalized to zero mean and unit variance before computing MSE. This
    forces the encoder to learn relative spatial structure rather than
    absolute intensity values, and is the recommended loss for pretraining
    on heterogeneous multi-modality datasets.

    The decoder's reconstruction head should produce unnormalized outputs
    (no final activation). During pretraining, the model learns to predict
    normalized patch values; at fine-tuning time the SSL head is discarded
    and only the encoder weights are transferred.

    Args:
        patch_size: Edge length (in voxels) of each 3D mask cube. Must match
            the mask_patch_size used in SwinMAE.generate_mask() so that
            normalization aligns with the masking grid. Defaults to 16.
        eps: Small constant added to per-patch std to prevent division by
            zero in uniform patches. Defaults to 1e-6.

    Raises:
        ValueError: If spatial dimensions of the input are not divisible by
            patch_size.
    """

    def __init__(self, patch_size: int = 16, eps: float = 1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.eps = eps

    def _normalize_patches(
        self,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize target intensities within each mask cube of the grid.

        Thin wrapper around
        :func:`misfit.utils.normalization.normalize_patchwise` — the same
        transform is applied by ``misfit_evaluate`` and ``misfit_inspect`` so
        metrics are computed in the space this loss actually optimizes.

        Args:
            target: Input volume of shape (B, C, D, H, W). Spatial
                dimensions must be divisible by ``patch_size``.

        Returns:
            Normalized volume of shape (B, C, D, H, W).

        Raises:
            ValueError: If any spatial dimension is not divisible by
                ``patch_size``.
        """
        return normalize_patchwise(target, self.patch_size, eps=self.eps)

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-patch normalized masked MSE loss.

        Args:
            reconstruction: Decoder output of shape (B, C, D, H, W).
                Should be unnormalized (no final activation in the decoder).
            target: Original unmasked input of shape (B, C, D, H, W).
            mask: Binary mask of shape (B, 1, D, H, W); 1 = masked voxel.

        Returns:
            Scalar normalized MSE loss over masked voxels.
        """
        check_reconstruction_inputs(reconstruction, target, mask)

        # Normalize target within each patch cube.
        target_normalized = self._normalize_patches(target)

        # Compute MSE between reconstruction and normalized target,
        # restricted to masked voxels.
        mask = mask.expand_as(reconstruction)
        diff_sq = (reconstruction - target_normalized) ** 2
        return (diff_sq * mask).sum() / mask.sum().clamp(min=1)
