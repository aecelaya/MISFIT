"""Tests for misfit.data_loading.transforms."""
import pytest
import torch
from monai.transforms import Compose

from misfit.data_loading.transforms import build_train_transforms, build_val_transforms


def _make_volume(size=(32, 32, 32)):
    return torch.randn(1, *size)


@pytest.mark.parametrize("builder", [build_train_transforms, build_val_transforms],
                         ids=["train", "val"])
def test_transforms_returns_compose(builder):
    t = builder((32, 32, 32))
    assert isinstance(t, Compose)


@pytest.mark.parametrize("builder", [build_train_transforms, build_val_transforms],
                         ids=["train", "val"])
def test_transforms_output_shape(builder):
    t = builder((32, 32, 32))
    out = t(_make_volume((32, 32, 32)))
    assert out.shape == (1, 32, 32, 32)


@pytest.mark.parametrize("builder", [build_train_transforms, build_val_transforms],
                         ids=["train", "val"])
def test_transforms_pads_small_volume(builder):
    t = builder((32, 32, 32))
    out = t(_make_volume((16, 16, 16)))
    assert out.shape == (1, 32, 32, 32)


@pytest.mark.parametrize("builder", [build_train_transforms, build_val_transforms],
                         ids=["train", "val"])
def test_transforms_output_dtype(builder):
    t = builder((32, 32, 32))
    out = t(_make_volume())
    assert out.dtype == torch.float32


def test_custom_patch_size():
    t = build_val_transforms((64, 64, 64))
    out = t(_make_volume((64, 64, 64)))
    assert out.shape == (1, 64, 64, 64)
