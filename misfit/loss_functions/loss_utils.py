"""Shared utilities for MISFIT reconstruction losses."""
import torch


def check_reconstruction_inputs(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Validate shapes and types for reconstruction loss inputs.

    Args:
        reconstruction: Decoder output, expected shape (B, C, D, H, W).
        target: Original input volume, expected shape (B, C, D, H, W).
        mask: Binary voxel mask, expected shape (B, 1, D, H, W).

    Raises:
        ValueError: If any shape or compatibility constraint is violated.
    """
    if reconstruction.ndim != 5:
        raise ValueError(
            f"reconstruction must be 5D (B, C, D, H, W), "
            f"got shape {tuple(reconstruction.shape)}."
        )
    if target.ndim != 5:
        raise ValueError(
            f"target must be 5D (B, C, D, H, W), "
            f"got shape {tuple(target.shape)}."
        )
    if mask.ndim != 5:
        raise ValueError(
            f"mask must be 5D (B, 1, D, H, W), "
            f"got shape {tuple(mask.shape)}."
        )
    if reconstruction.shape != target.shape:
        raise ValueError(
            f"reconstruction and target must have the same shape. "
            f"Got {tuple(reconstruction.shape)} vs {tuple(target.shape)}."
        )
    if mask.shape[1] != 1:
        raise ValueError(
            f"mask must have exactly 1 channel (B, 1, D, H, W), "
            f"got {mask.shape[1]} channels."
        )
    if mask.shape[0] != reconstruction.shape[0]:
        raise ValueError(
            f"mask batch size ({mask.shape[0]}) does not match "
            f"reconstruction batch size ({reconstruction.shape[0]})."
        )
    if mask.shape[2:] != reconstruction.shape[2:]:
        raise ValueError(
            f"mask spatial dimensions {tuple(mask.shape[2:])} do not match "
            f"reconstruction spatial dimensions {tuple(reconstruction.shape[2:])}."
        )
