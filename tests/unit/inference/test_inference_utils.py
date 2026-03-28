"""Tests for misfit.inference.inference_utils."""
import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest
import torch

import misfit.models  # noqa
from misfit.inference.inference_utils import (
    build_model_from_checkpoint,
    centre_crop_or_pad,
    get_default_device,
    load_and_normalise,
    load_checkpoint,
    pad_to_multiple,
)
from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE


MODEL_CONFIG = {
    "name": "swinmae-small",
    "patch_size": [32, 32, 32],
    "mask_patch_size": 16,
    "mask_ratio": 0.75,
}


def _make_checkpoint(tmp_path, patch_size=(32, 32, 32)):
    model = SwinMAE(in_channels=1, feature_size=12, img_size=patch_size,
                    mask_patch_size=16, mask_ratio=0.75)
    p = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, p)
    return p


def _tiny_model_builder(_name, **kwargs):
    """Registry stub that always returns a feature_size=12 SwinMAE."""
    return SwinMAE(feature_size=12, **kwargs)


def test_get_default_device_returns_string():
    d = get_default_device()
    assert d in ("cuda", "cpu")


def test_load_checkpoint(tmp_path):
    ckpt_path = _make_checkpoint(tmp_path)
    ckpt = load_checkpoint(ckpt_path, device="cpu")
    assert "model" in ckpt


def test_load_checkpoint_default_device(tmp_path):
    ckpt_path = _make_checkpoint(tmp_path)
    ckpt = load_checkpoint(ckpt_path)
    assert "model" in ckpt


def test_build_model_from_checkpoint(tmp_path):
    ckpt_path = _make_checkpoint(tmp_path)
    ckpt = load_checkpoint(ckpt_path, device="cpu")
    with patch("misfit.inference.inference_utils.get_model_from_registry",
               side_effect=_tiny_model_builder):
        model = build_model_from_checkpoint(ckpt, MODEL_CONFIG, device="cpu")
    assert isinstance(model, SwinMAE)


def test_build_model_default_device(tmp_path):
    ckpt_path = _make_checkpoint(tmp_path)
    ckpt = load_checkpoint(ckpt_path, device="cpu")
    with patch("misfit.inference.inference_utils.get_model_from_registry",
               side_effect=_tiny_model_builder):
        model = build_model_from_checkpoint(ckpt, MODEL_CONFIG)
    assert model is not None


def test_load_and_normalise_success(tmp_path):
    data = np.random.randn(16, 16, 16).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / "vol.nii.gz"
    nib.save(img, str(p))

    result = load_and_normalise(p, p1=-2, p99=2, fg_mean=0.0, fg_std=1.0)
    assert result is not None
    assert result.shape == (16, 16, 16)


def test_load_and_normalise_missing_file(tmp_path):
    result = load_and_normalise(tmp_path / "missing.nii.gz", -1, 1, 0, 1)
    assert result is None


def test_load_and_normalise_4d(tmp_path):
    data = np.random.randn(16, 16, 16, 2).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / "vol4d.nii.gz"
    nib.save(img, str(p))
    result = load_and_normalise(p, -2, 2, 0.0, 1.0)
    assert result.shape == (16, 16, 16)


@pytest.mark.parametrize("vol_shape,target", [
    ((16, 16, 16), (8, 8, 8)),   # crop
    ((8, 8, 8), (16, 16, 16)),   # pad
    ((16, 16, 16), (16, 16, 16)), # exact
])
def test_centre_crop_or_pad_output_shape(vol_shape, target):
    vol = np.random.randn(*vol_shape).astype(np.float32)
    result = centre_crop_or_pad(vol, target)
    assert result.shape == target


# ---------------------------------------------------------------------------
# pad_to_multiple
# ---------------------------------------------------------------------------

def test_pad_to_multiple_already_multiple():
    vol = np.zeros((32, 32, 32))
    padded, orig = pad_to_multiple(vol, (32, 32, 32))
    assert padded.shape == (32, 32, 32)
    assert orig == (32, 32, 32)


def test_pad_to_multiple_pads_each_dimension():
    vol = np.zeros((33, 65, 16))
    padded, orig = pad_to_multiple(vol, (32, 32, 32))
    assert padded.shape == (64, 96, 32)
    assert orig == (33, 65, 16)


def test_pad_to_multiple_padded_region_is_zero():
    vol = np.ones((33, 33, 33))
    padded, _ = pad_to_multiple(vol, (32, 32, 32))
    assert padded.shape == (64, 64, 64)
    assert padded[33:, :, :].sum() == 0
    assert padded[:, 33:, :].sum() == 0
    assert padded[:, :, 33:].sum() == 0
