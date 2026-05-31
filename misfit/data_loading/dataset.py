"""PyTorch Dataset for MISFIT MAE pretraining.

Streams NIfTI volumes on the fly from a Parquet metadata index built by
misfit_index. Normalization (clip + z-score) is applied using precomputed
per-volume statistics stored in the index — no recomputation at training time.
"""
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from misfit.data_loading.data_loading_constants import dc
from misfit.data_loading.transforms import build_train_transforms, build_val_transforms


class MISFITDataset(Dataset):
    """On-the-fly NIfTI streaming dataset for MISFIT MAE pretraining.

    Each call to __getitem__ loads a NIfTI file from disk, normalizes it
    using precomputed statistics from the metadata index, and returns a
    fixed-size 3D patch ready for the SwinMAE model.

    Normalization pipeline (per volume):
        1. Clip voxel values to [p1, p99] — removes modality outliers
           (CT metal, MRI bias, PET hotspots) without modality-specific
           thresholds.
        2. Z-score using foreground mean and std — centers and scales each
           volume independently, making CT and MRI comparable in the same
           training batch.
        3. (Optional) Crop to the precomputed foreground bounding box so
           that random patch sampling only draws from regions that contain
           tissue. Volumes smaller than patch_size after cropping are
           zero-padded back to patch_size by the transform pipeline.
           Especially useful for skull-stripped MRI and CT where a large
           fraction of the volume is air or scanner background.

    Args:
        index_path: Path to the Parquet metadata index produced by
            misfit_index. Must contain columns: path, p1, p99, fg_mean,
            fg_std, spacing_d, spacing_h, spacing_w.
        patch_size: Spatial dimensions of the output patch (D, H, W).
            Must be divisible by 32 (SwinUNETR-V2 requirement) and must
            match SwinMAE.img_size. Defaults to (96, 96, 96).
        augment: If True, apply random crop + flips + light intensity
            augmentation (training mode). If False, apply deterministic
            center crop only (validation mode). Defaults to True.
        split: If provided and the index contains a ``split`` column, only
            rows whose ``split`` value matches this string are used.
            Typical values: ``"train"``, ``"val"``, ``"test"``.
            Defaults to None (all rows).
        crop_to_fg: If True, crop each volume to its precomputed foreground
            bounding box (fg_x/y/z_start/end columns) before patch sampling.
            Requires those columns in the index. Defaults to True.
    """

    def __init__(
        self,
        index_path: str | Path,
        patch_size: tuple[int, int, int] = (96, 96, 96),
        augment: bool = True,
        split: str | None = None,
        crop_to_fg: bool = True,
    ):
        self.index_df = pd.read_parquet(index_path)
        if split is not None and "split" in self.index_df.columns:
            self.index_df = self.index_df[self.index_df["split"] == split]
        self.index_df = self.index_df.reset_index(drop=True)
        self.patch_size = patch_size
        self.crop_to_fg = crop_to_fg
        self.transforms = (
            build_train_transforms(patch_size)
            if augment
            else build_val_transforms(patch_size)
        )

    def __len__(self) -> int:
        return len(self.index_df)

    def _crop_to_fg_bbox(self, volume: np.ndarray, row: pd.Series) -> np.ndarray:
        """Crop volume to the precomputed foreground bounding box.

        Coordinates stored in the index are inclusive on both ends, so the
        slice is [start : end + 1].  If the cropped region is smaller than
        patch_size in any axis, the SpatialPad transform that runs first in
        both train and val pipelines will pad it back out with zeros.

        Args:
            volume: Normalized voxel array of shape (D, H, W).
            row: Row from the metadata index for this volume.

        Returns:
            Sub-array containing only the foreground region.
        """
        x0, x1 = int(row["fg_x_start"]), int(row["fg_x_end"]) + 1
        y0, y1 = int(row["fg_y_start"]), int(row["fg_y_end"]) + 1
        z0, z1 = int(row["fg_z_start"]), int(row["fg_z_end"]) + 1
        return volume[x0:x1, y0:y1, z0:z1]

    def _normalize(self, volume: np.ndarray, row: pd.Series) -> np.ndarray:
        """Apply clip + z-score normalization using precomputed index stats.

        Args:
            volume: Raw voxel array of shape (D, H, W).
            row: Row from the metadata index DataFrame for this volume.

        Returns:
            Normalized float32 array of the same shape.
        """
        volume = np.clip(volume, float(row["p1"]), float(row["p99"]))
        std = max(float(row["fg_std"]), dc.NORM_EPS)
        return (volume - float(row["fg_mean"])) / std

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load, normalize, and crop one volume.

        Args:
            idx: Index into the metadata index DataFrame.

        Returns:
            Float32 tensor of shape (1, D, H, W) where spatial dims equal
            patch_size. Returns a zero tensor of the same shape if the
            file cannot be loaded, so that a single corrupt volume does not
            crash a training run.
        """
        row = self.index_df.iloc[idx]

        try:
            img = nib.load(str(row["path"]))
            volume = np.asarray(img.dataobj, dtype=np.float32)
        except Exception:  # noqa: BLE001
            spacing = torch.tensor(
                [float(row["spacing_d"]), float(row["spacing_h"]), float(row["spacing_w"])],
                dtype=torch.float32,
            )
            return {
                "image": torch.zeros(1, *self.patch_size, dtype=torch.float32),
                "spacing": spacing,
            }

        # Handle 4D volumes (fMRI, DWI): take the first frame.
        if volume.ndim == 4:
            warnings.warn(
                f"{row['path']}: 4D volume encountered during training "
                f"(shape {volume.shape}), using first frame only.",
                stacklevel=2,
            )
            volume = volume[..., 0]

        # Clip + z-score using precomputed index statistics.
        volume = self._normalize(volume, row)

        # Restrict patch sampling to the foreground region so that random
        # crops always land on tissue rather than air or scanner background.
        if self.crop_to_fg:
            volume = self._crop_to_fg_bbox(volume, row)

        # Add channel dim (1, D, H, W) for MONAI transforms.
        volume_t = torch.from_numpy(volume.copy()).unsqueeze(0)

        spacing = torch.tensor(
            [float(row["spacing_d"]), float(row["spacing_h"]), float(row["spacing_w"])],
            dtype=torch.float32,
        )

        # Apply spatial crop and augmentation.
        return {"image": self.transforms(volume_t), "spacing": spacing}
