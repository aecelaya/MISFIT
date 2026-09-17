"""Tests for misfit.data_loading.dataset."""
import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from misfit.data_loading.dataset import MISFITDataset


def _write_index(
    tmp_path, nifti_path,
    p1=-2.0, p99=2.0, fg_mean=0.0, fg_std=1.0,
    fg_x_start=0, fg_x_end=31,
    fg_y_start=0, fg_y_end=31,
    fg_z_start=0, fg_z_end=31,
):
    df = pd.DataFrame([{
        "volume_id": "vol",
        "path": str(nifti_path),
        "shape_d": 32, "shape_h": 32, "shape_w": 32,
        "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
        "affine": json.dumps(np.eye(4).tolist()),
        "fg_x_start": fg_x_start, "fg_x_end": fg_x_end,
        "fg_y_start": fg_y_start, "fg_y_end": fg_y_end,
        "fg_z_start": fg_z_start, "fg_z_end": fg_z_end,
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


def _write_multi_index(tmp_path, paths):
    """Like _write_index but with one row per path, in order."""
    rows = [{
        "volume_id": f"vol{i}",
        "path": str(p),
        "shape_d": 32, "shape_h": 32, "shape_w": 32,
        "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
        "affine": json.dumps(np.eye(4).tolist()),
        "fg_x_start": 0, "fg_x_end": 31,
        "fg_y_start": 0, "fg_y_end": 31,
        "fg_z_start": 0, "fg_z_end": 31,
        "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0,
    } for i, p in enumerate(paths)]
    out = tmp_path / "index.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    return out


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
    with pytest.warns(UserWarning, match="failed to load"):
        sample = ds[0]
    assert torch.all(sample["image"] == 0.0)
    assert sample["spacing"].shape == (3,)


# ---------------------------------------------------------------------------
# Load-failure circuit breaker
# ---------------------------------------------------------------------------
#
# Every path in the index was already confirmed loadable by misfit_index, so
# a load failure at train time means something changed since (deleted/moved
# file, transient filesystem error). An isolated one is tolerated (warn,
# substitute a zero volume); max_load_failures *consecutive* failures raises
# instead, so a systemic problem fails the run loudly rather than silently
# training on an escalating run of zero-filled "volumes".

def test_dataset_load_failure_warns_with_progress_toward_the_limit(tmp_path):
    index_path = _write_multi_index(
        tmp_path, [tmp_path / "missing0.nii.gz", tmp_path / "missing1.nii.gz"]
    )
    ds = MISFITDataset(
        index_path, patch_size=(32, 32, 32), augment=False, max_load_failures=3
    )
    with pytest.warns(UserWarning, match=r"1/3 consecutive failures"):
        ds[0]
    with pytest.warns(UserWarning, match=r"2/3 consecutive failures"):
        ds[1]


def test_dataset_load_failure_counter_resets_after_a_success(tmp_path):
    """A good load in between failures resets the streak -- isolated bad
    files spread across a long run never trip the breaker."""
    good_path = _make_nifti(tmp_path)
    index_path = _write_multi_index(
        tmp_path, [tmp_path / "missing0.nii.gz", good_path, tmp_path / "missing1.nii.gz"]
    )
    ds = MISFITDataset(
        index_path, patch_size=(32, 32, 32), augment=False, max_load_failures=2
    )
    with pytest.warns(UserWarning, match=r"1/2 consecutive failures"):
        ds[0]  # streak = 1
    ds[1]  # successful load -> streak resets to 0
    with pytest.warns(UserWarning, match=r"1/2 consecutive failures"):
        ds[2]  # streak = 1 again, not 2 -- would have raised if it hadn't reset


def test_dataset_raises_after_max_consecutive_load_failures(tmp_path):
    """max_load_failures consecutive failures aborts instead of continuing."""
    index_path = _write_multi_index(
        tmp_path,
        [tmp_path / "missing0.nii.gz", tmp_path / "missing1.nii.gz",
         tmp_path / "missing2.nii.gz"],
    )
    ds = MISFITDataset(
        index_path, patch_size=(32, 32, 32), augment=False, max_load_failures=3
    )
    with pytest.warns(UserWarning):
        ds[0]
    with pytest.warns(UserWarning):
        ds[1]
    with pytest.raises(RuntimeError, match="3 volume loads failed back to back"):
        ds[2]


def test_dataset_max_load_failures_defaults_to_three(tmp_path):
    index_path = _write_index(tmp_path, tmp_path / "missing.nii.gz")
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    assert ds.max_load_failures == 3


def test_dataset_pads_small_volume(tmp_path):
    """Volume smaller than patch_size should be padded."""
    nifti_path = _make_nifti(tmp_path, shape=(16, 16, 16))
    index_path = _write_index(tmp_path, nifti_path)
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    sample = ds[0]
    assert sample["image"].shape == (1, 32, 32, 32)


# ---------------------------------------------------------------------------
# crop_to_fg
# ---------------------------------------------------------------------------

def test_crop_to_fg_true_is_default(tmp_path):
    """crop_to_fg defaults to True."""
    nifti_path = _make_nifti(tmp_path)
    index_path = _write_index(tmp_path, nifti_path)
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    assert ds.crop_to_fg is True


def test_crop_to_fg_bbox_crops_to_correct_shape(tmp_path):
    """_crop_to_fg_bbox returns the sub-region defined by the index bbox."""
    nifti_path = _make_nifti(tmp_path, shape=(32, 32, 32))
    index_path = _write_index(
        tmp_path, nifti_path,
        fg_x_start=4, fg_x_end=11,   # 8 voxels
        fg_y_start=8, fg_y_end=19,   # 12 voxels
        fg_z_start=2, fg_z_end=17,   # 16 voxels
    )
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    row = ds.index_df.iloc[0]
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    cropped = ds._crop_to_fg_bbox(vol, row)
    assert cropped.shape == (8, 12, 16)


def test_crop_to_fg_small_bbox_padded_to_patch_size(tmp_path):
    """Fg bbox smaller than patch_size is padded back up by SpatialPad."""
    nifti_path = _make_nifti(tmp_path, shape=(32, 32, 32))
    index_path = _write_index(
        tmp_path, nifti_path,
        fg_x_start=10, fg_x_end=21,  # 12 voxels — less than patch_size=32
        fg_y_start=10, fg_y_end=21,
        fg_z_start=10, fg_z_end=21,
    )
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), augment=False)
    sample = ds[0]
    assert sample["image"].shape == (1, 32, 32, 32)


def test_crop_to_fg_false_skips_bbox_crop(tmp_path):
    """crop_to_fg=False passes the full volume to the transform pipeline."""
    nifti_path = _make_nifti(tmp_path, shape=(32, 32, 32))
    index_path = _write_index(
        tmp_path, nifti_path,
        fg_x_start=12, fg_x_end=19,  # tight 8-voxel bbox
        fg_y_start=12, fg_y_end=19,
        fg_z_start=12, fg_z_end=19,
    )
    ds = MISFITDataset(
        index_path, patch_size=(32, 32, 32), augment=False, crop_to_fg=False
    )
    sample = ds[0]
    assert sample["image"].shape == (1, 32, 32, 32)
