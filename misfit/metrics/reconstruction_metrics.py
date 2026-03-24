"""Pure-numpy reconstruction quality metric functions for MISFIT.

All functions operate on single-volume numpy arrays of shape (D, H, W) and
return a scalar float.  The ``mask`` argument is a binary array of the same
shape where 1 indicates a voxel that was masked (hidden from the encoder)
during pretraining — metrics that evaluate on masked voxels only use it as
a boolean selector.

SSIM is computed on the full volume because structural similarity requires a
spatial neighbourhood; restricting it to scattered masked voxels would give
a meaningless result.
"""
import numpy as np
from skimage.metrics import structural_similarity


def compute_masked_mae(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Mean absolute error on masked voxels only.

    Args:
        reconstruction: Reconstructed volume, shape (D, H, W).
        target: Ground-truth volume, shape (D, H, W).
        mask: Binary mask, shape (D, H, W). 1 = masked patch.
        eps: Added to denominator to prevent division by zero.

    Returns:
        Scalar MAE over masked voxels.
    """
    masked_voxels = mask > 0
    n = masked_voxels.sum() + eps
    return float(np.abs(reconstruction[masked_voxels] - target[masked_voxels]).sum() / n)


def compute_masked_mse(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Mean squared error on masked voxels only.

    Args:
        reconstruction: Reconstructed volume, shape (D, H, W).
        target: Ground-truth volume, shape (D, H, W).
        mask: Binary mask, shape (D, H, W). 1 = masked patch.
        eps: Added to denominator to prevent division by zero.

    Returns:
        Scalar MSE over masked voxels.
    """
    masked_voxels = mask > 0
    n = masked_voxels.sum() + eps
    diff = reconstruction[masked_voxels] - target[masked_voxels]
    return float((diff ** 2).sum() / n)


def compute_masked_psnr(
    reconstruction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    data_range: float = 6.0,
    eps: float = 1e-8,
) -> float:
    """Peak signal-to-noise ratio on masked voxels only.

    PSNR = 10 * log10(data_range^2 / MSE).  Uses the masked MSE so it
    reflects reconstruction quality on the held-out patches.

    Args:
        reconstruction: Reconstructed volume, shape (D, H, W).
        target: Ground-truth volume, shape (D, H, W).
        mask: Binary mask, shape (D, H, W). 1 = masked patch.
        data_range: Peak-to-peak intensity range of normalised inputs.
            Defaults to 6.0 (≈ ±3σ after clip + z-score).
        eps: Added to MSE to prevent log(0).

    Returns:
        Scalar PSNR in dB.
    """
    mse = compute_masked_mse(reconstruction, target, mask, eps=eps)
    return float(10.0 * np.log10(data_range ** 2 / (mse + eps)))


def compute_ssim(
    reconstruction: np.ndarray,
    target: np.ndarray,
    data_range: float = 6.0,
) -> float:
    """Structural similarity index (SSIM) over the full 3D volume.

    Computed on the full volume (not just masked voxels) because SSIM
    requires a spatial neighbourhood for its local statistics window.

    Args:
        reconstruction: Reconstructed volume, shape (D, H, W).
        target: Ground-truth volume, shape (D, H, W).
        data_range: Peak-to-peak intensity range. Defaults to 6.0.

    Returns:
        Scalar SSIM in [-1, 1].
    """
    return float(
        structural_similarity(
            target,
            reconstruction,
            data_range=data_range,
        )
    )
