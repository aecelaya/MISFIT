"""Per-volume metadata computation utilities for the MISFIT index.

Each function in this module is designed to be called in a parallel worker
process. They have no shared state and rely only on standard library and
nibabel/numpy, keeping the pickling overhead low.
"""
import json
import warnings
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import skimage.filters

# Percentile range used to clip the volume before Otsu thresholding. The 33rd
# percentile lower bound anchors the clip near the air/tissue boundary in CT
# and near background in MRI/PET, giving Otsu a clean bimodal histogram.
_FG_PERCENTILE_LOW  = 33.0
_FG_PERCENTILE_HIGH = 99.5


def get_volume_id(path: Path) -> str:
    """Derive a clean volume identifier from a NIfTI file path.

    Strips both .nii.gz and .nii extensions so that
    ``/data/patient_001.nii.gz`` becomes ``patient_001``.

    Args:
        path: Path to a NIfTI file.

    Returns:
        Volume identifier string.
    """
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def collect_nifti_paths(data_dir: str | Path) -> list[Path]:
    """Recursively collect all NIfTI files under a directory.

    Args:
        data_dir: Root directory to search.

    Returns:
        Sorted list of Path objects for all .nii and .nii.gz files found.
    """
    data_dir = Path(data_dir)
    paths = sorted(
        list(data_dir.rglob("*.nii.gz")) + list(data_dir.rglob("*.nii"))
    )
    return paths


def compute_volume_stats(nifti_path: str | Path) -> dict[str, Any]:
    """Compute per-volume metadata for the MISFIT metadata index.

    Loads the full NIfTI volume and computes all statistics needed for
    on-the-fly normalization during training: foreground bounding box,
    intensity percentiles, and foreground mean/std (post-clip).

    Foreground is detected by clipping the volume to the [33rd, 99.5th]
    percentile range and applying an Otsu threshold. The resulting binary
    mask is used solely to derive a tight bounding box; intensity statistics
    are then computed over all voxels within that bounding box. This mirrors
    the approach used in MIST and is robust across CT, MRI, and PET:
        - CT: Otsu separates tissue from air, giving a tight body bbox.
        - MRI/PET: when the whole image is tissue, the bbox spans the full
          volume and statistics are computed over all voxels.

    If foreground detection yields an empty mask (e.g. constant volumes),
    the full volume is used as a fallback.

    Intensity statistics (p1, p99, fg_mean, fg_std) are computed over
    bbox voxels with p1/p99 clipping applied before mean/std to suppress
    metal artifacts, PET hotspots, and MRI bias fields.

    These values are used by the data loader to apply z-score normalization
    on the fly without recomputing stats per epoch.

    If the file cannot be read or processed, the returned dict contains an
    ``"error"`` key with the exception message and no other stat fields.

    Args:
        nifti_path: Path to a .nii or .nii.gz file.

    Returns:
        Dict with the following keys on success:
            volume_id (str), path (str),
            shape_d/h/w (int), spacing_d/h/w (float),
            affine (str, JSON-encoded 4x4 list),
            fg_x/y/z_start/end (int),
            p1, p99, fg_mean, fg_std (float).
        Dict with keys ``"path"`` and ``"error"`` on failure.
    """
    path = Path(nifti_path)
    try:
        img = nib.load(str(path))

        # --- Header (no decompression needed) ---
        zooms = img.header.get_zooms()
        affine = img.affine.tolist()

        # Handle 4D volumes (e.g., fMRI, DWI): take first frame.
        volume = np.asarray(img.dataobj, dtype=np.float32)
        if volume.ndim == 4:
            warnings.warn(
                f"{path}: 4D volume detected (shape {volume.shape}), "
                "using first frame only. If this is a multi-echo or "
                "diffusion series, consider preprocessing to 3D first.",
                stacklevel=2,
            )
            volume = volume[..., 0]
        if volume.ndim != 3:
            return {
                "path": str(path),
                "error": f"Expected 3D volume, got shape {volume.shape}.",
            }

        d, h, w = volume.shape
        spacing_d = float(zooms[0]) if len(zooms) > 0 else 1.0
        spacing_h = float(zooms[1]) if len(zooms) > 1 else 1.0
        spacing_w = float(zooms[2]) if len(zooms) > 2 else 1.0

        # --- Foreground detection ---
        # Clip to suppress outliers, then apply Otsu threshold. The resulting
        # mask is used only to derive a bounding box — not for masking stats.
        lower, upper = np.percentile(
            volume, [_FG_PERCENTILE_LOW, _FG_PERCENTILE_HIGH]
        )
        clipped = np.clip(volume, lower, upper)
        try:
            threshold = skimage.filters.threshold_otsu(clipped)
            fg_mask = clipped > threshold
        except Exception:  # noqa: BLE001
            fg_mask = np.zeros(volume.shape, dtype=bool)

        if not fg_mask.any():
            warnings.warn(
                f"{path}: foreground detection returned an empty mask. "
                "Falling back to full-volume statistics. "
                "This may indicate a corrupted or near-empty scan.",
                stacklevel=2,
            )
            fg_mask = np.ones(volume.shape, dtype=bool)

        # --- Foreground bounding box ---
        nz = np.argwhere(fg_mask)
        x_start = int(nz[:, 0].min())
        x_end   = int(nz[:, 0].max())
        y_start = int(nz[:, 1].min())
        y_end   = int(nz[:, 1].max())
        z_start = int(nz[:, 2].min())
        z_end   = int(nz[:, 2].max())

        # --- Intensity statistics over bbox voxels ---
        bbox_voxels = volume[
            x_start:x_end + 1,
            y_start:y_end + 1,
            z_start:z_end + 1,
        ].ravel()
        p1  = float(np.percentile(bbox_voxels, 1))
        p99 = float(np.percentile(bbox_voxels, 99))

        # Clip to [p1, p99] before computing mean/std to suppress outliers.
        bbox_clipped = np.clip(bbox_voxels, p1, p99)
        fg_mean = float(bbox_clipped.mean())
        fg_std  = float(bbox_clipped.std())

        return {
            "volume_id": get_volume_id(path),
            "path":      str(path.resolve()),
            "shape_d":   d,
            "shape_h":   h,
            "shape_w":   w,
            "spacing_d": spacing_d,
            "spacing_h": spacing_h,
            "spacing_w": spacing_w,
            "affine":    json.dumps(affine),
            "fg_x_start": x_start,
            "fg_x_end":   x_end,
            "fg_y_start": y_start,
            "fg_y_end":   y_end,
            "fg_z_start": z_start,
            "fg_z_end":   z_end,
            "p1":      p1,
            "p99":     p99,
            "fg_mean": fg_mean,
            "fg_std":  fg_std,
        }

    except Exception as exc:  # noqa: BLE001
        return {"path": str(path), "error": str(exc)}
