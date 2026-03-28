"""Tests for misfit.metrics.reconstruction_metrics."""
import numpy as np
import pytest

from misfit.metrics.reconstruction_metrics import (
    compute_masked_mae,
    compute_masked_mse,
    compute_masked_psnr,
    compute_ssim,
)


def _arrays(d=16, h=16, w=16):
    rng = np.random.default_rng(0)
    recon = rng.standard_normal((d, h, w)).astype(np.float32)
    target = rng.standard_normal((d, h, w)).astype(np.float32)
    mask = np.zeros((d, h, w), dtype=np.float32)
    mask[:d//2] = 1.0
    return recon, target, mask


def test_compute_masked_mae_non_negative():
    r, t, m = _arrays()
    assert compute_masked_mae(r, t, m) >= 0


def test_compute_masked_mae_zero_when_perfect():
    r, t, m = _arrays()
    assert compute_masked_mae(r, r, m) == pytest.approx(0.0, abs=1e-5)


def test_compute_masked_mae_empty_mask():
    r, t, _ = _arrays()
    m = np.zeros_like(r)
    val = compute_masked_mae(r, t, m)
    assert val == pytest.approx(0.0, abs=1e-3)  # ~0 / eps


def test_compute_masked_mse_non_negative():
    r, t, m = _arrays()
    assert compute_masked_mse(r, t, m) >= 0


def test_compute_masked_mse_zero_when_perfect():
    r, t, m = _arrays()
    assert compute_masked_mse(r, r, m) == pytest.approx(0.0, abs=1e-5)


def test_compute_masked_psnr_finite():
    r, t, m = _arrays()
    val = compute_masked_psnr(r, t, m, data_range=6.0)
    assert np.isfinite(val)
    assert val > 0


def test_compute_masked_psnr_perfect_is_high():
    r, t, m = _arrays()
    val = compute_masked_psnr(r, r, m, data_range=6.0)
    assert val > 50.0  # nearly infinite


def test_compute_ssim_in_range():
    r, t, m = _arrays()
    val = compute_ssim(r, t, data_range=6.0)
    assert -1.0 <= val <= 1.0


def test_compute_ssim_identical_volumes():
    r, _, _ = _arrays()
    val = compute_ssim(r, r, data_range=6.0)
    assert val == pytest.approx(1.0, abs=1e-5)


def test_metrics_registry_masked_mae_callable():
    from misfit.metrics.metrics_registry import get_metric
    metric = get_metric("masked_mae")
    r, t, m = _arrays()
    val = metric(r, t, m)
    assert isinstance(val, float)


def test_metrics_registry_ssim_callable():
    from misfit.metrics.metrics_registry import get_metric
    metric = get_metric("ssim")
    r, t, m = _arrays()
    val = metric(r, t, m)
    assert isinstance(val, float)


def test_metrics_registry_masked_psnr_with_data_range():
    from misfit.metrics.metrics_registry import get_metric
    metric = get_metric("masked_psnr")
    r, t, m = _arrays()
    val = metric(r, t, m, data_range=6.0)
    assert isinstance(val, float)
