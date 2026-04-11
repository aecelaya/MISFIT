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


@pytest.mark.parametrize("recon,target,mask,match", [
    pytest.param(
        torch.randn(2, 1, 8, 8),    torch.randn(2, 1, 8, 8, 8), torch.zeros(2, 1, 8, 8, 8),
        "5D", id="recon_4d",
    ),
    pytest.param(
        torch.randn(2, 1, 8, 8, 8), torch.randn(2, 1, 8, 8),    torch.zeros(2, 1, 8, 8, 8),
        "5D", id="target_4d",
    ),
    pytest.param(
        torch.randn(2, 1, 8, 8, 8), torch.randn(2, 1, 8, 8, 8), torch.zeros(2, 1, 8, 8),
        "5D", id="mask_4d",
    ),
    pytest.param(
        torch.randn(2, 1, 8, 8, 8), torch.randn(2, 1, 8, 8, 4), torch.zeros(2, 1, 8, 8, 8),
        "same shape", id="shape_mismatch",
    ),
    pytest.param(
        torch.randn(2, 1, 8, 8, 8), torch.randn(2, 1, 8, 8, 8), torch.zeros(2, 2, 8, 8, 8),
        "1 channel", id="mask_channels",
    ),
    pytest.param(
        torch.randn(2, 1, 8, 8, 8), torch.randn(2, 1, 8, 8, 8), torch.zeros(3, 1, 8, 8, 8),
        "batch size", id="batch_mismatch",
    ),
    pytest.param(
        torch.randn(2, 1, 8, 8, 8), torch.randn(2, 1, 8, 8, 8), torch.zeros(2, 1, 4, 8, 8),
        "spatial dimensions", id="spatial_mismatch",
    ),
])
def test_check_reconstruction_invalid_inputs(recon, target, mask, match):
    with pytest.raises(ValueError, match=match):
        check_reconstruction_inputs(recon, target, mask)
