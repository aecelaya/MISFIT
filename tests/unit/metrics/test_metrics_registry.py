"""Tests for misfit.metrics.metrics_registry."""
import numpy as np
import pytest

from misfit.metrics.metrics_registry import (
    METRIC_REGISTRY,
    ReconstructionMetric,
    get_metric,
    list_registered_metrics,
    register_metric,
)


def test_list_registered_metrics_sorted():
    metrics = list_registered_metrics()
    assert metrics == sorted(metrics)


def test_list_registered_metrics_contains_builtins():
    metrics = list_registered_metrics()
    assert "masked_mae" in metrics
    assert "masked_mse" in metrics
    assert "masked_psnr" in metrics
    assert "ssim" in metrics


def test_get_metric_success():
    metric = get_metric("masked_mae")
    assert isinstance(metric, ReconstructionMetric)


def test_get_metric_unknown_raises():
    with pytest.raises(ValueError, match="not registered"):
        get_metric("totally_unknown_metric_xyz")


def test_metric_has_best_worst():
    metric = get_metric("masked_mae")
    assert hasattr(metric, "best")
    assert hasattr(metric, "worst")
    assert hasattr(metric, "name")


def test_init_subclass_enforces_name():
    """Missing 'name' class attribute should raise TypeError."""
    with pytest.raises(TypeError, match="name"):
        class BadMetric(ReconstructionMetric):
            best = 0.0
            worst = float("inf")
            def __call__(self, r, t, m, **kw): return 0.0


def test_init_subclass_enforces_best():
    with pytest.raises(TypeError, match="best"):
        class BadMetric(ReconstructionMetric):
            name = "bad_test_metric_best"
            worst = float("inf")
            def __call__(self, r, t, m, **kw): return 0.0


def test_init_subclass_enforces_worst():
    with pytest.raises(TypeError, match="worst"):
        class BadMetric(ReconstructionMetric):
            name = "bad_test_metric_worst"
            best = 0.0
            def __call__(self, r, t, m, **kw): return 0.0


def test_masked_mse_metric_callable():
    """MaskedMSE.__call__ (line 119) is exercised via the registry instance."""
    metric = get_metric("masked_mse")
    rng = np.random.default_rng(0)
    recon = rng.random((8, 8, 8)).astype(np.float32)
    target = rng.random((8, 8, 8)).astype(np.float32)
    mask = (rng.random((8, 8, 8)) > 0.5).astype(np.float32)
    result = metric(recon, target, mask)
    assert isinstance(result, float)
    assert result >= 0.0


def test_register_metric_decorator():
    @register_metric
    class _TestMetricReg(ReconstructionMetric):
        name = "_test_metric_reg_abc"
        best = 0.0
        worst = float("inf")
        def __call__(self, r, t, m, **kw): return 1.0

    assert "_test_metric_reg_abc" in METRIC_REGISTRY
    m = get_metric("_test_metric_reg_abc")
    assert m(None, None, None) == 1.0
