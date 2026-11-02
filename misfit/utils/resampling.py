"""Optional resampling to a target voxel spacing (``--target-spacing``).

By default MISFIT feeds volumes at their native voxel grid. With a target
spacing set, every code path that turns a NIfTI volume into model input
(training, evaluation, inspection, encoding, embedding) resamples the
normalized volume to that spacing first, so a patch covers the same physical
extent in every volume and matches a MIST model preprocessed at that spacing.

The target spacing lives in ``config.json`` under ``spatial_config``, the same
section and keys MIST uses, so MIST can read a MISFIT config directly.
Spacings and shapes use the array's own (x, y, z) axis order, which is how both
nibabel (MISFIT) and MIST's SimpleITK reader lay out volumes.
"""
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def get_target_spacing(config: dict) -> tuple[float, float, float] | None:
    """Return the target spacing saved in a ``config.json`` dict, if any.

    Configs written before ``--target-spacing`` existed have no
    ``spatial_config`` section; they trained at native resolution.

    Args:
        config: Full config dict loaded from ``config.json``.

    Returns:
        The (x, y, z) target spacing, or None for native resolution.
    """
    spacing = config.get("spatial_config", {}).get("target_spacing")
    return None if spacing is None else tuple(float(s) for s in spacing)


def get_row_spacing(row: pd.Series) -> tuple[float, float, float]:
    """Return a volume's (x, y, z) voxel spacing from its index row."""
    return (
        float(row["spacing_d"]),
        float(row["spacing_h"]),
        float(row["spacing_w"]),
    )


def resampled_shape(
    shape: Sequence[int],
    spacing: Sequence[float],
    target_spacing: Sequence[float],
) -> tuple[int, int, int]:
    """Shape of a volume after resampling it from spacing to target_spacing.

    Each axis keeps its physical extent: ``round(n * spacing / target)``,
    floored at one voxel.
    """
    return tuple(
        max(1, round(n * s / t))
        for n, s, t in zip(shape, spacing, target_spacing, strict=True)
    )


def resample_to_shape(volume: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Trilinearly resample a 3D volume to an exact shape.

    Args:
        volume: Float array of shape (X, Y, Z).
        shape: Output shape (X', Y', Z').

    Returns:
        Float32 array of shape ``shape``. Returned unchanged (as float32) when
        the shape already matches.
    """
    shape = tuple(int(n) for n in shape)
    if tuple(volume.shape) == shape:
        return volume.astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))
    resampled = F.interpolate(
        tensor[None, None], size=shape, mode="trilinear", align_corners=False
    )
    return resampled[0, 0].numpy()


def resample_to_spacing(
    volume: np.ndarray,
    spacing: Sequence[float],
    target_spacing: Sequence[float] | None,
) -> np.ndarray:
    """Resample a 3D volume from its voxel spacing to target_spacing.

    Args:
        volume: Float array of shape (X, Y, Z).
        spacing: The volume's (x, y, z) voxel spacing.
        target_spacing: Desired (x, y, z) spacing, or None to leave the volume
            at its native resolution.

    Returns:
        Float32 array resampled to ``target_spacing`` (or ``volume`` itself,
        as float32, when ``target_spacing`` is None).
    """
    if target_spacing is None:
        return volume.astype(np.float32, copy=False)
    return resample_to_shape(
        volume, resampled_shape(volume.shape, spacing, target_spacing)
    )
