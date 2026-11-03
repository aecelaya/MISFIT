"""Built-in intensity normalization policies."""
from collections.abc import Mapping

import numpy as np

from misfit.normalization.base import IntensityScale, NormalizationPolicy
from misfit.normalization.normalization_registry import register_normalization_policy


@register_normalization_policy("clip_and_z_score")
class ClipAndZScore(NormalizationPolicy):
    """Clip to the foreground's [p1, p99], then z-score with foreground stats.

    MISFIT's default. Clipping removes modality-specific outliers (CT metal,
    MRI bias, PET hotspots) without modality-specific rules, and z-scoring with
    foreground (not whole-volume) statistics keeps air and background from
    dominating, so CT and MRI share one intensity range. All four statistics
    are precomputed by misfit_index, so nothing is computed at load time.
    """

    def normalize(
        self, volume: np.ndarray, row: Mapping
    ) -> tuple[np.ndarray, IntensityScale]:
        """Clip and z-score *volume* with its index row's statistics."""
        scale = IntensityScale(
            shift=float(row["fg_mean"]),
            scale=max(float(row["fg_std"]), self.EPS),
        )
        clipped = np.clip(volume, float(row["p1"]), float(row["p99"]))
        return ((clipped - scale.shift) / scale.scale).astype(np.float32), scale


@register_normalization_policy("z_score_only")
class ZScoreOnly(NormalizationPolicy):
    """Z-score with the mean and std of the volume's nonzero bounding box.

    nnssl's normalization, used to pretrain nnFoundation: nnssl crops each
    volume to the bounding box of its nonzero voxels, then z-scores with that
    box's mean and std — no clipping, and no foreground mask (nnFoundation's
    plan has ``use_mask_for_norm: false``). Matching it lets models pretrained
    by nnssl see the input distribution they were trained on. The statistics
    are computed from the volume at load time, so no index columns are needed.
    """

    def normalize(
        self, volume: np.ndarray, row: Mapping
    ) -> tuple[np.ndarray, IntensityScale]:
        """Z-score *volume* with its nonzero bounding box's statistics."""
        region = volume[_nonzero_bbox(volume)]
        scale = IntensityScale(
            shift=float(region.mean(dtype=np.float64)),
            scale=max(float(region.std(dtype=np.float64)), self.EPS),
        )
        return ((volume - scale.shift) / scale.scale).astype(np.float32), scale


def _nonzero_bbox(volume: np.ndarray) -> tuple[slice, ...]:
    """Slices of the bounding box of *volume*'s nonzero voxels.

    nnssl fills holes in the nonzero mask before taking its bounding box;
    filled holes are interior, so the box is the same. An all-zero volume
    keeps its full extent.
    """
    nonzero = volume != 0
    if not nonzero.any():
        return tuple(slice(None) for _ in volume.shape)
    slices = []
    for axis in range(volume.ndim):
        other = tuple(a for a in range(volume.ndim) if a != axis)
        hits = np.flatnonzero(nonzero.any(axis=other))
        slices.append(slice(int(hits[0]), int(hits[-1]) + 1))
    return tuple(slices)
