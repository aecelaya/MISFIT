"""Tests for misfit.loss_functions.reconstruction.masked_l1."""
import torch
import pytest

from misfit.loss_functions.reconstruction.masked_l1 import MaskedL1Loss


@pytest.fixture
def loss_fn():
    return MaskedL1Loss()


def _tensors(B=2, C=1, D=8, H=8, W=8):
    recon = torch.randn(B, C, D, H, W)
    target = torch.randn(B, C, D, H, W)
    mask = torch.zeros(B, 1, D, H, W)
    mask[:, :, :4] = 1.0
    return recon, target, mask


def test_masked_l1_scalar(loss_fn):
    r, t, m = _tensors()
    loss = loss_fn(r, t, m)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_masked_l1_zero_when_perfect(loss_fn):
    r, t, m = _tensors()
    r = t.clone()
    loss = loss_fn(r, t, m)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_masked_l1_empty_mask(loss_fn):
    r, t, _ = _tensors()
    mask = torch.zeros(2, 1, 8, 8, 8)
    loss = loss_fn(r, t, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_masked_l1_validates_inputs(loss_fn):
    with pytest.raises(ValueError):
        loss_fn(torch.randn(2, 8, 8), torch.randn(2, 1, 8, 8, 8), torch.zeros(2, 1, 8, 8, 8))


def test_masked_l1_registered():
    import misfit.loss_functions  # noqa
    from misfit.loss_functions.loss_registry import get_loss
    cls = get_loss("masked_l1")
    assert cls is MaskedL1Loss
