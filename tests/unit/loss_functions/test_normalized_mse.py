"""Tests for misfit.loss_functions.reconstruction.normalized_mse."""
import pytest
import torch

from misfit.loss_functions.reconstruction.normalized_mse import NormalizedMaskedMSELoss


@pytest.fixture
def loss_fn():
    return NormalizedMaskedMSELoss(patch_size=4)


def _tensors(B=2, C=1, D=8, H=8, W=8, patch_size=4):
    recon = torch.randn(B, C, D, H, W)
    target = torch.randn(B, C, D, H, W)
    mask = torch.zeros(B, 1, D, H, W)
    mask[:, :, :patch_size] = 1.0
    return recon, target, mask


def test_normalized_mse_scalar(loss_fn):
    r, t, m = _tensors()
    loss = loss_fn(r, t, m)
    assert loss.ndim == 0
    assert loss.item() >= 0


def test_normalized_mse_validates_inputs(loss_fn):
    with pytest.raises(ValueError):
        loss_fn(torch.randn(2, 1, 4, 4), torch.randn(2, 1, 8, 8, 8), torch.zeros(2, 1, 8, 8, 8))


def test_normalized_mse_patch_size_mismatch_raises():
    """Patch size that doesn't divide spatial dims should raise ValueError."""
    loss = NormalizedMaskedMSELoss(patch_size=3)
    r = torch.randn(2, 1, 8, 8, 8)
    t = torch.randn(2, 1, 8, 8, 8)
    m = torch.zeros(2, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="divisible"):
        loss(r, t, m)


def test_normalized_mse_empty_mask(loss_fn):
    r, t, _ = _tensors()
    mask = torch.zeros(2, 1, 8, 8, 8)
    loss = loss_fn(r, t, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_normalize_patches_shape(loss_fn):
    target = torch.randn(2, 1, 8, 8, 8)
    normed = loss_fn._normalize_patches(target)
    assert normed.shape == target.shape


def test_normalized_mse_registered():
    import misfit.loss_functions  # noqa
    from misfit.loss_functions.loss_registry import get_loss
    cls = get_loss("normalized_masked_mse")
    assert cls is NormalizedMaskedMSELoss


def test_normalize_patches_unchanged_after_shared_helper_refactor():
    """Training invariant: _normalize_patches must still produce the exact
    pre-refactor output (the loss is on the training path)."""
    def _pre_refactor(target, p, eps=1e-6):
        B, C, D, H, W = target.shape
        gd, gh, gw = D // p, H // p, W // p
        t = target.reshape(B, C, gd, p, gh, p, gw, p)
        t = t.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(B, C, gd * gh * gw, p ** 3)
        t = (t - t.mean(-1, keepdim=True)) / (t.std(-1, keepdim=True) + eps)
        t = t.reshape(B, C, gd, gh, gw, p, p, p)
        return t.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(B, C, D, H, W)

    torch.manual_seed(0)
    target = torch.randn(2, 1, 16, 16, 16)
    loss = NormalizedMaskedMSELoss(patch_size=8)
    assert torch.equal(loss._normalize_patches(target), _pre_refactor(target, 8))
