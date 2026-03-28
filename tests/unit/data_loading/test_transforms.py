"""Tests for misfit.data_loading.transforms."""
import torch
import pytest
from monai.transforms import Compose

from misfit.data_loading.transforms import build_train_transforms, build_val_transforms


def _make_volume(size=(32, 32, 32)):
    return torch.randn(1, *size)


def test_build_train_transforms_returns_compose():
    t = build_train_transforms((32, 32, 32))
    assert isinstance(t, Compose)


def test_build_val_transforms_returns_compose():
    t = build_val_transforms((32, 32, 32))
    assert isinstance(t, Compose)


def test_train_transforms_output_shape():
    t = build_train_transforms((32, 32, 32))
    out = t(_make_volume((32, 32, 32)))
    assert out.shape == (1, 32, 32, 32)


def test_val_transforms_output_shape():
    t = build_val_transforms((32, 32, 32))
    out = t(_make_volume((32, 32, 32)))
    assert out.shape == (1, 32, 32, 32)


def test_val_transforms_pads_small_volume():
    t = build_val_transforms((32, 32, 32))
    out = t(_make_volume((16, 16, 16)))
    assert out.shape == (1, 32, 32, 32)


def test_train_transforms_pads_small_volume():
    t = build_train_transforms((32, 32, 32))
    out = t(_make_volume((16, 16, 16)))
    assert out.shape == (1, 32, 32, 32)


def test_val_transforms_output_dtype():
    t = build_val_transforms((32, 32, 32))
    out = t(_make_volume())
    assert out.dtype == torch.float32


def test_train_transforms_output_dtype():
    t = build_train_transforms((32, 32, 32))
    out = t(_make_volume())
    assert out.dtype == torch.float32


def test_custom_patch_size():
    t = build_val_transforms((64, 64, 64))
    out = t(_make_volume((64, 64, 64)))
    assert out.shape == (1, 64, 64, 64)
