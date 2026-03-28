"""Utility functions for MISFIT inference modules."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

import misfit.models  # noqa: F401 — trigger model registrations
from misfit.models.model_registry import get_model_from_registry


def get_default_device() -> str:
    """Return the default inference device (CUDA if available, else CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
) -> Dict:
    """Load a MISFIT checkpoint from disk.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint produced by
            :class:`~misfit.training.trainers.mae_trainer.MAETrainer`.
        device: Target device. Defaults to :func:`get_default_device`.

    Returns:
        Checkpoint dictionary with keys ``model``, ``epoch``, etc.
    """
    device = device or get_default_device()
    return torch.load(
        Path(checkpoint_path),
        map_location=device,
        weights_only=True,
    )


def build_model_from_checkpoint(
    checkpoint: Dict,
    model_config: Dict,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """Reconstruct a model from a model config dict and load checkpoint weights.

    Args:
        checkpoint: Checkpoint dictionary (output of :func:`load_checkpoint`).
        model_config: Model configuration dict (``config["model"]`` from
            ``config.json``).  Must contain ``name``, ``patch_size``,
            ``mask_patch_size``, and ``mask_ratio``.
        device: Target device. Defaults to :func:`get_default_device`.

    Returns:
        Model in eval mode with checkpoint weights loaded.
    """
    device = device or get_default_device()

    model = get_model_from_registry(
        model_config["name"],
        in_channels=1,
        img_size=tuple(model_config["patch_size"]),
        mask_patch_size=model_config["mask_patch_size"],
        mask_ratio=model_config["mask_ratio"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def load_and_normalise(
    nifti_path: Union[str, Path],
    p1: float,
    p99: float,
    fg_mean: float,
    fg_std: float,
    eps: float = 1e-8,
) -> Optional[np.ndarray]:
    """Load a NIfTI volume and apply clip + z-score normalisation.

    Args:
        nifti_path: Path to a ``.nii`` or ``.nii.gz`` file.
        p1: 1st-percentile foreground intensity (lower clip bound).
        p99: 99th-percentile foreground intensity (upper clip bound).
        fg_mean: Foreground mean after clipping.
        fg_std: Foreground standard deviation after clipping.
        eps: Minimum value for fg_std to avoid division by zero.

    Returns:
        Float32 numpy array of shape (D, H, W), or None if the file cannot
        be read.
    """
    try:
        img = nib.load(nifti_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
    except Exception:  # pylint: disable=broad-except
        return None

    if data.ndim == 4:
        data = data[..., 0]

    data = np.clip(data, p1, p99)
    data = (data - fg_mean) / max(fg_std, eps)
    return data


def pad_to_multiple(
    volume: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Zero-pad *volume* so every dimension is a multiple of *patch_size*.

    Args:
        volume: Input array of shape (D, H, W).
        patch_size: Target patch dimensions (pd, ph, pw).

    Returns:
        Tuple of ``(padded, original_shape)`` where *padded* has each
        dimension as an exact multiple of the corresponding patch dimension,
        and *original_shape* is the original ``(D, H, W)``.
    """
    original_shape = volume.shape
    pad_width = []
    for dim, p in zip(volume.shape, patch_size):
        remainder = dim % p
        pad = (p - remainder) % p  # 0 if already a multiple
        pad_width.append((0, pad))
    if any(p[1] > 0 for p in pad_width):
        volume = np.pad(volume, pad_width, mode="constant", constant_values=0)
    return volume, original_shape


def centre_crop_or_pad(
    volume: np.ndarray,
    target: Tuple[int, int, int],
) -> np.ndarray:
    """Pad (if needed) then centre-crop a volume to *target* shape.

    Args:
        volume: Input array of shape (D, H, W).
        target: Desired (D, H, W) output shape.

    Returns:
        Array of exactly shape *target*.
    """
    # Pad any axis that is smaller than the target.
    pad_width = []
    for dim, t in zip(volume.shape, target):
        deficit = max(0, t - dim)
        pad_before = deficit // 2
        pad_width.append((pad_before, deficit - pad_before))
    if any(p[0] + p[1] > 0 for p in pad_width):
        volume = np.pad(volume, pad_width, mode="constant", constant_values=0)

    # Centre crop.
    slices = []
    for dim, t in zip(volume.shape, target):
        start = (dim - t) // 2
        slices.append(slice(start, start + t))
    return volume[tuple(slices)]
