"""Tests for misfit.embedding.objectives.base.CropFeaturesDataset and crop_collate_fn."""
import numpy as np
import pandas as pd
import pytest
import torch

from misfit.embedding.objectives.base import CropFeaturesDataset, crop_collate_fn


def _make_npz(path, n_crops=5, C=16, d=3):
    np.savez(
        path,
        feature_map=np.random.randn(n_crops, C, d, d, d).astype(np.float32),
        positions=np.random.rand(n_crops, 3).astype(np.float32),
    )


def _make_labels_df(tmp_path, volume_ids, label_col="label", labels=None):
    if labels is None:
        labels = list(range(len(volume_ids)))
    return pd.DataFrame({
        "volume_id": volume_ids,
        "features_path": [str(tmp_path / f"{vid}.npz") for vid in volume_ids],
        label_col: labels,
    })


def test_crop_features_dataset_len(tmp_path):
    _make_npz(tmp_path / "vol1.npz")
    _make_npz(tmp_path / "vol2.npz")
    df = _make_labels_df(tmp_path, ["vol1", "vol2"])
    ds = CropFeaturesDataset(df, "label")
    assert len(ds) == 2


def test_crop_features_dataset_skips_missing(tmp_path):
    _make_npz(tmp_path / "vol1.npz")
    df = _make_labels_df(tmp_path, ["vol1", "missing_vol"])
    ds = CropFeaturesDataset(df, "label")
    assert len(ds) == 1


def test_crop_features_dataset_getitem_shapes(tmp_path):
    _make_npz(tmp_path / "vol1.npz", n_crops=5, C=16)
    df = _make_labels_df(tmp_path, ["vol1"])
    ds = CropFeaturesDataset(df, "label")
    features, positions, label = ds[0]
    assert features.shape == (5, 16)
    assert positions.shape == (5, 3)
    assert isinstance(label, int)


def test_crop_features_dataset_label_to_idx(tmp_path):
    _make_npz(tmp_path / "vol1.npz")
    _make_npz(tmp_path / "vol2.npz")
    df = pd.DataFrame({
        "volume_id": ["vol1", "vol2"],
        "features_path": [str(tmp_path / "vol1.npz"), str(tmp_path / "vol2.npz")],
        "label": ["cat", "dog"],
    })
    label_to_idx = {"cat": 0, "dog": 1}
    ds = CropFeaturesDataset(df, "label", label_to_idx=label_to_idx)
    _, _, label0 = ds[0]
    _, _, label1 = ds[1]
    assert label0 == 0
    assert label1 == 1


def test_crop_collate_fn_output_shapes(tmp_path):
    _make_npz(tmp_path / "vol1.npz", n_crops=4, C=8)
    _make_npz(tmp_path / "vol2.npz", n_crops=6, C=8)
    df = _make_labels_df(tmp_path, ["vol1", "vol2"])
    ds = CropFeaturesDataset(df, "label")
    batch = [ds[0], ds[1]]
    features, positions, padding_mask, labels = crop_collate_fn(batch)
    # Max crops = 6
    assert features.shape == (2, 6, 8)
    assert positions.shape == (2, 6, 3)
    assert padding_mask.shape == (2, 6)
    assert labels.shape == (2,)


def test_crop_collate_fn_padding_mask(tmp_path):
    _make_npz(tmp_path / "vol1.npz", n_crops=3, C=8)
    _make_npz(tmp_path / "vol2.npz", n_crops=5, C=8)
    df = _make_labels_df(tmp_path, ["vol1", "vol2"])
    ds = CropFeaturesDataset(df, "label")
    batch = [ds[0], ds[1]]
    _, _, padding_mask, _ = crop_collate_fn(batch)
    # vol1 has 3 crops, max is 5, so last 2 are padded
    assert padding_mask[0, 3:].all()  # padded
    assert not padding_mask[0, :3].any()  # not padded
    assert not padding_mask[1].any()  # vol2: all valid
