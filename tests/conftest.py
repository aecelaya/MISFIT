"""Shared pytest fixtures for MISFIT tests."""
import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

# ---------------------------------------------------------------------------
# NIfTI / volume fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nifti_image():
    """A small in-memory NIfTI1Image (32, 32, 32) float32."""
    data = np.random.randn(32, 32, 32).astype(np.float32)
    return nib.Nifti1Image(data, np.eye(4))


@pytest.fixture
def nifti_path(tmp_path, nifti_image):
    """Write a NIfTI to disk and return its Path."""
    p = tmp_path / "volume.nii.gz"
    nib.save(nifti_image, str(p))
    return p


@pytest.fixture
def nifti_4d_path(tmp_path):
    """Write a 4-D NIfTI (32, 32, 32, 2) to disk."""
    data = np.random.randn(32, 32, 32, 2).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / "volume_4d.nii.gz"
    nib.save(img, str(p))
    return p


# ---------------------------------------------------------------------------
# Parquet index fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def parquet_index(tmp_path, nifti_path):
    """Write a minimal parquet index pointing to nifti_path."""
    df = pd.DataFrame([{
        "volume_id": "volume",
        "path": str(nifti_path),
        "shape_d": 32, "shape_h": 32, "shape_w": 32,
        "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
        "affine": json.dumps(np.eye(4).tolist()),
        "fg_x_start": 0, "fg_x_end": 31,
        "fg_y_start": 0, "fg_y_end": 31,
        "fg_z_start": 0, "fg_z_end": 31,
        "p1": -2.0, "p99": 2.0,
        "fg_mean": 0.0, "fg_std": 1.0,
    }])
    p = tmp_path / "index.parquet"
    df.to_parquet(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def small_batch():
    """A tiny (B=2, C=1, D=16, H=16, W=16) float32 tensor."""
    return torch.randn(2, 1, 16, 16, 16)


@pytest.fixture
def recon_target_mask():
    """Reconstruction, target, and mask tensors for loss tests."""
    B, C, D, H, W = 2, 1, 8, 8, 8
    reconstruction = torch.randn(B, C, D, H, W)
    target = torch.randn(B, C, D, H, W)
    mask = torch.zeros(B, 1, D, H, W)
    mask[:, :, :4, :, :] = 1.0  # mask first half of depth
    return reconstruction, target, mask
