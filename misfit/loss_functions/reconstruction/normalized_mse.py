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
        """Normalize target intensities within each patch in the grid.

        Reshapes the volume into non-overlapping patch cubes of size
        patch_size^3, computes the mean and std within each cube, and
        returns a normalized volume of the same shape.

        Args:
            target: Input volume of shape (B, C, D, H, W). Spatial
                dimensions must be divisible by patch_size.

        Returns:
            Normalized volume of shape (B, C, D, H, W).

        Raises:
            ValueError: If any spatial dimension is not divisible by
                patch_size.
        """
        B, C, D, H, W = target.shape
        p = self.patch_size

        for dim, size in zip(("D", "H", "W"), (D, H, W), strict=True):
            if size % p != 0:
                raise ValueError(
                    f"Spatial dimension {dim}={size} is not divisible by "
                    f"patch_size={p}. Ensure img_size and patch_size are "
                    f"consistent between SwinMAE and NormalizedMaskedMSELoss."
                )

        gd, gh, gw = D // p, H // p, W // p
        num_patches = gd * gh * gw

        # Reshape: (B, C, D, H, W) → (B, C, num_patches, p^3).
        # Step 1: split spatial dims into (grid, patch) pairs.
        t = target.reshape(B, C, gd, p, gh, p, gw, p)
        # Step 2: group all patch dimensions last.
        t = t.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(B, C, num_patches, p * p * p)

        # Compute per-patch statistics over the p^3 voxels.
        mean = t.mean(dim=-1, keepdim=True)   # (B, C, num_patches, 1)
        std = t.std(dim=-1, keepdim=True)      # (B, C, num_patches, 1)
        t_norm = (t - mean) / (std + self.eps)

        # Reshape back: (B, C, num_patches, p^3) → (B, C, D, H, W).
        t_norm = t_norm.reshape(B, C, gd, gh, gw, p, p, p)
        t_norm = t_norm.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(B, C, D, H, W)
        return t_norm

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
