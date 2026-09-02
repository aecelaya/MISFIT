"""Tests for misfit.utils.hardware — BF16 capability detection and AMP resolution."""
import contextlib
import warnings

import pytest
import torch

from misfit.utils.hardware import autocast_context, bf16_supported, resolve_amp

# ---------------------------------------------------------------------------
# bf16_supported
# ---------------------------------------------------------------------------

def test_bf16_supported_false_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bf16_supported() is False


def test_bf16_supported_true_on_ampere(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    assert bf16_supported() is True


def test_bf16_supported_true_on_hopper(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    assert bf16_supported() is True


def test_bf16_supported_false_on_volta(monkeypatch):
    """SM 7.0 (V100) reports BF16 via emulation only — bf16_supported must reject it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 0))
    assert bf16_supported() is False


# ---------------------------------------------------------------------------
# resolve_amp
# ---------------------------------------------------------------------------

def test_resolve_amp_returns_false_when_not_requested(monkeypatch):
    monkeypatch.setattr("misfit.utils.hardware.bf16_supported", lambda: True)
    assert resolve_amp(False) is False


def test_resolve_amp_true_when_requested_and_supported(monkeypatch):
    monkeypatch.setattr("misfit.utils.hardware.bf16_supported", lambda: True)
    assert resolve_amp(True) is True


def test_resolve_amp_downgrades_to_fp32_and_warns(monkeypatch):
    monkeypatch.setattr("misfit.utils.hardware.bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="falling back to FP32"):
        assert resolve_amp(True) is False


def test_resolve_amp_names_the_gpu_in_the_warning(monkeypatch):
    monkeypatch.setattr("misfit.utils.hardware.bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Tesla V100-SXM2-16GB")
    with pytest.warns(UserWarning, match="Tesla V100"):
        assert resolve_amp(True) is False


def test_resolve_amp_downgrade_silent_when_warn_false(monkeypatch):
    monkeypatch.setattr("misfit.utils.hardware.bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        assert resolve_amp(True, warn=False) is False


# ---------------------------------------------------------------------------
# autocast_context
# ---------------------------------------------------------------------------

def test_autocast_context_disabled_is_nullcontext():
    ctx = autocast_context(False)
    assert isinstance(ctx, contextlib.nullcontext)


def test_autocast_context_enabled_is_autocast():
    ctx = autocast_context(True)
    assert isinstance(ctx, torch.autocast)
