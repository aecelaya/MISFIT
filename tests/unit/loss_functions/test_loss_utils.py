"""Tests for misfit.loss_functions.loss_utils."""
import torch
import pytest

from misfit.loss_functions.loss_utils import check_reconstruction_inputs


def _make(B=2, C=1, D=8, H=8, W=8):
    recon = torch.randn(B, C, D, H, W)
    target = torch.randn(B, C, D, H, W)
    mask = torch.zeros(B, 1, D, H, W)
    return recon, target, mask


def test_check_valid_inputs():
    r, t, m = _make()
    check_reconstruction_inputs(r, t, m)  # should not raise


def test_check_reconstruction_wrong_ndim():
    r = torch.randn(2, 1, 8, 8)  # 4D not 5D
    t = torch.randn(2, 1, 8, 8, 8)
    m = torch.zeros(2, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="5D"):
        check_reconstruction_inputs(r, t, m)


def test_check_target_wrong_ndim():
    r = torch.randn(2, 1, 8, 8, 8)
    t = torch.randn(2, 1, 8, 8)  # 4D
    m = torch.zeros(2, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="5D"):
        check_reconstruction_inputs(r, t, m)


def test_check_mask_wrong_ndim():
    r = torch.randn(2, 1, 8, 8, 8)
    t = torch.randn(2, 1, 8, 8, 8)
    m = torch.zeros(2, 1, 8, 8)  # 4D
    with pytest.raises(ValueError, match="5D"):
        check_reconstruction_inputs(r, t, m)


def test_check_shape_mismatch():
    r = torch.randn(2, 1, 8, 8, 8)
    t = torch.randn(2, 1, 8, 8, 4)  # different W
    m = torch.zeros(2, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="same shape"):
        check_reconstruction_inputs(r, t, m)


def test_check_mask_channels_not_one():
    r = torch.randn(2, 1, 8, 8, 8)
    t = torch.randn(2, 1, 8, 8, 8)
    m = torch.zeros(2, 2, 8, 8, 8)  # 2 channels
    with pytest.raises(ValueError, match="1 channel"):
        check_reconstruction_inputs(r, t, m)


def test_check_mask_batch_mismatch():
    r = torch.randn(2, 1, 8, 8, 8)
    t = torch.randn(2, 1, 8, 8, 8)
    m = torch.zeros(3, 1, 8, 8, 8)  # batch 3 vs 2
    with pytest.raises(ValueError, match="batch size"):
        check_reconstruction_inputs(r, t, m)


def test_check_mask_spatial_mismatch():
    r = torch.randn(2, 1, 8, 8, 8)
    t = torch.randn(2, 1, 8, 8, 8)
    m = torch.zeros(2, 1, 4, 8, 8)  # spatial mismatch
    with pytest.raises(ValueError, match="spatial dimensions"):
        check_reconstruction_inputs(r, t, m)
