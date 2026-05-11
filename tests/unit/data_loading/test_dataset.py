"""Tests for misfit.data_loading.dataset."""
import json

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from misfit.data_loading.dataset import MISFITDataset


def _write_index(tmp_path, nifti_path, p1=-2.0, p99=2.0, fg_mean=0.0, fg_std=1.0):
    df = pd.DataFrame([{
        "volume_id": "vol",
        "path": str(nifti_path),
        "shape_d": 32, "shape_h": 32, "shape_w": 32,
        "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
        "affine": json.dumps(np.eye(4).tolist()),
        "fg_x_start": 0, "fg_x_end": 31,
        "fg_y_start": 0, "fg_y_end": 31,
        "fg_z_start": 0, "fg_z_end": 31,
        "p1": p1, "p99": p99,
        "fg_mean": fg_mean, "fg_std": fg_std,
    }])
    p = tmp_path / "index.parquet"
    df.to_parquet(p, index=False)
    return p


def _make_nifti(tmp_path, name="volume.nii.gz", shape=(32, 32, 32)):
    data = np.random.randn(*shape).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / name
    nib.save(img, str(p))
    return p


def test_dataset_len(tmp_path):
    nifti_path = _make_nifti(tmp_path)
    index_path = _write_index(tmp_path, nifti_path)
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    assert len(ds) == 1


def test_dataset_getitem_shape_val(tmp_path):
    nifti_path = _make_nifti(tmp_path)
    index_path = _write_index(tmp_path, nifti_path)
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    sample = ds[0]
    assert isinstance(sample, dict)
    assert sample["image"].shape == (1, 32, 32, 32)
    assert sample["image"].dtype == torch.float32
    assert sample["spacing"].shape == (3,)
    assert sample["spacing"].dtype == torch.float32


def test_dataset_getitem_shape_train(tmp_path):
    nifti_path = _make_nifti(tmp_path)
    index_path = _write_index(tmp_path, nifti_path)
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=True)
    sample = ds[0]
    assert sample["image"].shape == (1, 32, 32, 32)


def test_dataset_normalize_zero_std(tmp_path):
    """When fg_std is 0, normalization should use eps instead of dividing by 0."""
    nifti_path = _make_nifti(tmp_path)
    index_path = _write_index(tmp_path, nifti_path, fg_std=0.0)
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    sample = ds[0]
    assert not torch.isnan(sample["image"]).any()


def test_dataset_4d_nifti(tmp_path):
    """4D NIfTI should take first frame, not crash."""
    nifti_path = _make_nifti(tmp_path, name="vol4d.nii.gz", shape=(32, 32, 32, 2))
    index_path = _write_index(tmp_path, nifti_path)
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    sample = ds[0]
    assert sample["image"].shape == (1, 32, 32, 32)


def test_dataset_missing_file_returns_zeros(tmp_path):
    """Missing nifti should return zeros image with spacing still populated."""
    index_path = _write_index(tmp_path, tmp_path / "missing.nii.gz")
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    sample = ds[0]
    assert torch.all(sample["image"] == 0.0)
    assert sample["spacing"].shape == (3,)


def test_dataset_pads_small_volume(tmp_path):
    """Volume smaller than patch_size should be padded."""
    nifti_path = _make_nifti(tmp_path, shape=(16, 16, 16))
    index_path = _write_index(tmp_path, nifti_path)
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    sample = ds[0]
    assert sample["image"].shape == (1, 32, 32, 32)
