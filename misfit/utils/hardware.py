"""Hardware capability helpers for MISFIT automatic mixed precision (AMP).

MISFIT's AMP uses BF16 autocast, which is only hardware-accelerated on NVIDIA
Ampere or newer GPUs (A100, H100, RTX 30xx+) and on AMD CDNA (MI100/200/300
series) or RDNA3+ GPUs (RX 7000 series and newer, WMMA-capable). On pre-Ampere
NVIDIA cards (T4, V100, RTX 20xx) and on AMD RDNA1/2 GPUs (RX 5000/6000 series)
BF16 autocast runs without matrix/tensor-core support and is slower than plain
FP32 — and on CPU it is unavailable entirely. These helpers resolve a requested
AMP setting against the current hardware so training, evaluation, and inference
transparently fall back to FP32 instead of silently running a slow or
unsupported path.

PyTorch's ROCm build reuses the ``torch.cuda`` namespace and the ``"cuda"``
device string as a compatibility shim, so ``torch.cuda.is_available()``, the
per-rank ``cuda:<rank>`` device, the NCCL backend (which routes to RCCL), and
``torch.autocast("cuda", ...)`` all already work on AMD GPUs without any change
— the one thing that does not is native BF16 detection, which is what this
module handles.

This mirrors ``mist.utils.hardware`` so MISFIT and MIST resolve precision
identically.
"""

import contextlib
import warnings
from typing import Literal

import torch

AcceleratorType = Literal["cuda", "rocm", "cpu"]

# gcnArchName prefixes (before ROCm's trailing ":sramecc+:xnack-" feature
# suffix, hence the .split(":")[0] normalization below) of AMD GPUs with
# hardware matrix-acceleration for BF16: CDNA (MFMA -- MI100/MI200/MI300/
# MI350 series) and RDNA3+ (WMMA -- RX 7000 series and newer, plus the
# Strix/Strix Halo APUs). RDNA1/2 (gfx10xx, e.g. the RX 5000/6000 series and
# Radeon PRO W6000 series) has no matrix hardware at all and executes BF16 on
# plain shader ALUs -- confirmed empirically (in MIST) on a Radeon RX
# 6800-class (gfx1030) card, where torch.cuda.is_bf16_supported() reports True
# but enabling AMP measurably regressed training speed vs. FP32. This list is
# intentionally an allow-list, not a denylist: an unrecognized gcnArchName
# (a future architecture, or one we haven't confirmed) falls back to False
# and a warning via resolve_amp, the same conservative-by-default posture
# bf16_supported() already takes for pre-Ampere NVIDIA GPUs.
_ROCM_BF16_ACCELERATED_ARCHES = frozenset(
    {
        "gfx908",  # MI100 (CDNA1)
        "gfx90a",  # MI210 / MI250 / MI250X (CDNA2)
        "gfx940",
        "gfx941",
        "gfx942",  # MI300 series (CDNA3)
        "gfx950",  # MI350 series (CDNA4)
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",  # RDNA3 (RX 7000 series)
        "gfx1150",
        "gfx1151",  # RDNA3.5 (Strix/Strix Halo APUs)
        "gfx1200",
        "gfx1201",  # RDNA4 (RX 9000 series)
    }
)


def get_accelerator_type() -> AcceleratorType:
    """Return the accelerator MISFIT is currently running on.

    PyTorch's ROCm build reuses the ``torch.cuda`` namespace as a compatibility
    shim, so ``torch.cuda.is_available()`` is ``True`` on AMD GPUs too.
    ``torch.version.hip`` is the documented way to tell the two apart: it is a
    version string on ROCm builds and ``None`` on CUDA / CPU-only builds.
    """
    if not torch.cuda.is_available():
        return "cpu"
    if getattr(torch.version, "hip", None) is not None:
        return "rocm"
    return "cuda"


def bf16_supported() -> bool:
    """Return True if the current accelerator *natively* supports BF16.

    On CUDA, checks the compute capability (Ampere / SM 8.0 or newer) rather
    than ``torch.cuda.is_bf16_supported()``, which by default returns True on
    pre-Ampere GPUs (T4, V100) via slow software emulation — which would defeat
    the FP32 fallback this module exists to provide.

    On ROCm, ``torch.cuda.get_device_capability()`` returns a gfx-derived
    pseudo-capability that is ``>= 8`` for essentially every AMD GPU (CDNA
    reports ``(9, 0)``, RDNA reports ``(10, x)`` / ``(11, 0)``), so the CUDA
    check cannot be reused. ``torch.cuda.is_bf16_supported()`` is no help
    either — it reports True on RDNA1/2 GPUs (e.g. the RX 6800, gfx1030),
    which have no BF16 matrix hardware at all and run it on plain shader ALUs
    with no speed benefit. So instead this checks the GPU's ``gcnArchName``
    against ``_ROCM_BF16_ACCELERATED_ARCHES``, an allow-list of architectures
    known to have hardware matrix acceleration for BF16 (CDNA's MFMA,
    RDNA3+'s WMMA). An unrecognized architecture is treated as unsupported,
    the same conservative default used for pre-Ampere NVIDIA GPUs.
    """
    accelerator = get_accelerator_type()
    if accelerator == "cpu":
        return False
    if accelerator == "rocm":
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        return arch in _ROCM_BF16_ACCELERATED_ARCHES
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def resolve_amp(requested: bool, *, warn: bool = True) -> bool:
    """Resolve a requested AMP setting against the current hardware.

    Returns ``True`` only when AMP was requested *and* BF16 is hardware
    supported. When a request is downgraded — because there is no GPU, the
    NVIDIA GPU is pre-Ampere, or the AMD GPU has no BF16 matrix hardware
    (RDNA1/2) — a ``UserWarning`` is emitted so callers know FP32 will be used
    instead.

    Args:
        requested: The AMP setting requested via config (``training.amp``).
        warn: Whether to warn when the request is downgraded to FP32.

    Returns:
        The effective AMP setting for the current hardware.
    """
    if not requested:
        return False
    if bf16_supported():
        return True
    if warn:
        device = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU (no CUDA device)"
        )
        warnings.warn(
            f"AMP was requested but {device} has no hardware BF16 support; "
            "falling back to FP32. Set training.amp = false to silence this.",
            stacklevel=2,
        )
    return False


def autocast_context(enabled: bool):
    """Return a BF16 autocast context for the current accelerator, or null.

    ``resolve_amp`` / ``bf16_supported`` already guarantee ``enabled`` is only
    ``True`` when the current accelerator has native BF16 support, so this just
    needs to point ``torch.autocast`` at the right device type: ``"cuda"``
    covers both real CUDA and ROCm (the same compatibility shim used elsewhere
    in this module), and CPU never reaches the ``True`` branch since
    ``bf16_supported()`` is always ``False`` there.
    """
    if enabled:
        device_type = "cpu" if get_accelerator_type() == "cpu" else "cuda"
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return contextlib.nullcontext()
