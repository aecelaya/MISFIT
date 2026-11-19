"""Tests for misfit.utils.hardware — BF16 capability detection and AMP resolution."""
import contextlib
import types
import warnings

import pytest
import torch

from misfit.utils import hardware
from misfit.utils.hardware import (
    autocast_context,
    bf16_supported,
    get_accelerator_type,
    resolve_amp,
)

# ---------------------------------------------------------------------------
# get_accelerator_type
# ---------------------------------------------------------------------------

def test_get_accelerator_type_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_accelerator_type() == "cpu"


def test_get_accelerator_type_cuda_when_hip_unset(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert get_accelerator_type() == "cuda"


def test_get_accelerator_type_rocm_when_hip_set(monkeypatch):
    """torch.version.hip is a version string on ROCm builds, None otherwise."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.41134", raising=False)
    assert get_accelerator_type() == "rocm"


# ---------------------------------------------------------------------------
# bf16_supported — CUDA
# ---------------------------------------------------------------------------

def test_bf16_supported_false_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert bf16_supported() is False


def test_bf16_supported_true_on_ampere(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    assert bf16_supported() is True


def test_bf16_supported_true_on_hopper(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    assert bf16_supported() is True


def test_bf16_supported_false_on_volta(monkeypatch):
    """SM 7.0 (V100) reports BF16 via emulation only — bf16_supported must reject it."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 0))
    assert bf16_supported() is False


# ---------------------------------------------------------------------------
# bf16_supported — AMD ROCm
# ---------------------------------------------------------------------------

class _FakeDeviceProperties:
    """Minimal stand-in for torch.cuda.get_device_properties()'s return."""

    def __init__(self, gcn_arch_name: str) -> None:
        self.gcnArchName = gcn_arch_name


def test_bf16_supported_true_on_cdna(monkeypatch):
    """CDNA (MFMA matrix hardware), e.g. MI210 — the ':sramecc+:xnack-' suffix is stripped."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.41134", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda idx=0: _FakeDeviceProperties("gfx90a:sramecc+:xnack-"),
    )
    assert bf16_supported() is True


def test_bf16_supported_true_on_rdna3(monkeypatch):
    """RDNA3+ (WMMA matrix hardware), e.g. RX 7900."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2.41134", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda idx=0: _FakeDeviceProperties("gfx1100"),
    )
    assert bf16_supported() is True


def test_bf16_supported_false_on_rdna2(monkeypatch):
    """RDNA1/2 (no matrix hardware), e.g. RX 6800 / gfx1030.

    Regression guard: torch.cuda.is_bf16_supported() reports True on RDNA2 via
    software emulation on plain shader ALUs — confirmed (in MIST) to regress
    training speed vs. FP32 on a real gfx1030 card — and get_device_capability()
    reports (10, 3), so neither the is_bf16_supported() shortcut nor the CUDA
    'major >= 8' check may be used. The gcnArchName allow-list must decide.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.3.42134", raising=False)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda idx=0: _FakeDeviceProperties("gfx1030"),
    )
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))
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


def test_resolve_amp_names_the_amd_gpu_in_the_warning(monkeypatch):
    """An RDNA1/2 AMD card downgrades to FP32 and names itself in the warning."""
    monkeypatch.setattr("misfit.utils.hardware.bf16_supported", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "AMD Radeon RX 6800")
    with pytest.warns(UserWarning, match="AMD Radeon RX 6800"):
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = autocast_context(True)
    assert isinstance(ctx, torch.autocast)


def test_autocast_context_enabled_uses_cuda_device_type_on_rocm(monkeypatch):
    """ROCm reuses the "cuda" autocast device type, same as real CUDA."""
    monkeypatch.setattr("misfit.utils.hardware.get_accelerator_type", lambda: "rocm")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = autocast_context(True)
    assert ctx.device == "cuda"


def test_autocast_context_enabled_uses_cpu_device_type_on_cpu(monkeypatch):
    """The defensive CPU branch: "cpu" autocast device type, not "cuda".

    bf16_supported() is always False on CPU, so resolve_amp() never lets
    enabled=True reach here on real CPU-only hardware — this covers the branch
    directly in case a caller bypasses that.
    """
    monkeypatch.setattr("misfit.utils.hardware.get_accelerator_type", lambda: "cpu")
    ctx = autocast_context(True)
    assert ctx.device == "cpu"


# ---------------------------------------------------------------------------
# resolve_compile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "accelerator", "os_name", "expected"),
    [
        (True, "cuda", "posix", True),
        (True, "rocm", "posix", True),
        (True, "cpu", "posix", False),
        (True, "cuda", "nt", False),
        (False, "cuda", "posix", False),
    ],
)
def test_resolve_compile(monkeypatch, requested, accelerator, os_name, expected):
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: accelerator)
    monkeypatch.setattr(hardware.os, "name", os_name)
    assert hardware.resolve_compile(requested) is expected


# ---------------------------------------------------------------------------
# tune_gpu_math
# ---------------------------------------------------------------------------


def test_tune_gpu_math_is_skipped_on_cpu(monkeypatch):
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: "cpu")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    hardware.tune_gpu_math()
    assert torch.backends.cudnn.benchmark is False


def _restore_math_settings(monkeypatch):
    precision = torch.get_float32_matmul_precision()
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(hardware, "get_accelerator_type", lambda: "cuda")
    return precision


def test_tune_gpu_math_turns_on_the_fast_settings(monkeypatch):
    precision = _restore_math_settings(monkeypatch)
    conv = types.SimpleNamespace(fp32_precision="fp32")
    monkeypatch.setattr(torch.backends.cudnn, "conv", conv, raising=False)
    try:
        hardware.tune_gpu_math()
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(precision)
    assert conv.fp32_precision == "tf32"
    assert torch.backends.cudnn.benchmark is True


def test_tune_gpu_math_on_older_pytorch(monkeypatch):
    """Before PyTorch 2.5 there is no cudnn.conv, only allow_tf32."""
    precision = _restore_math_settings(monkeypatch)
    fake_cudnn = types.SimpleNamespace(allow_tf32=False, benchmark=False)
    monkeypatch.setattr(hardware.torch.backends, "cudnn", fake_cudnn)
    try:
        hardware.tune_gpu_math()
    finally:
        torch.set_float32_matmul_precision(precision)
    assert fake_cudnn.allow_tf32 is True
    assert fake_cudnn.benchmark is True

