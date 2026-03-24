"""Registry for reconstruction metrics used in MISFIT evaluation.

Mirrors MIST's metrics_registry pattern: a ``ReconstructionMetric`` base
class enforces a consistent interface, a module-level ``METRIC_REGISTRY``
maps names to callable instances, and ``@register_metric`` decorates
concrete classes to self-register on import.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np

from misfit.metrics import reconstruction_metrics as _m
from misfit.metrics.metrics_constants import ReconstructionMetricsConstants as _c


class ReconstructionMetric(ABC):
    """Base class for all MISFIT reconstruction quality metrics.

    Subclasses must define:
        name (str): Unique identifier used as registry key and CSV column.
        best (float): Ideal value (e.g. 0.0 for MAE, 1.0 for SSIM).
        worst (float): Worst-case fallback value.
    """

    name: str
    best: float
    worst: float

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for attr in ("name", "best", "worst"):
            if not any(
                attr in base.__dict__
                for base in cls.__mro__
                if base not in (ReconstructionMetric, object)
            ):
                raise TypeError(
                    f"{cls.__name__} must define class attribute '{attr}'"
                )

    @abstractmethod
    def __call__(
        self,
        reconstruction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
        **kwargs,
    ) -> float:
        """Compute the metric for a single volume.

        Args:
            reconstruction: Model output, shape (D, H, W).
            target: Ground-truth input, shape (D, H, W).
            mask: Binary mask, shape (D, H, W). 1 = masked patch.
            **kwargs: Metric-specific keyword arguments.

        Returns:
            Scalar metric value.
        """


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, ReconstructionMetric] = {}


def register_metric(cls):
    """Class decorator — instantiates and registers the metric."""
    instance = cls()
    METRIC_REGISTRY[instance.name] = instance
    return cls


def get_metric(name: str) -> ReconstructionMetric:
    """Retrieve a metric by name.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Metric '{name}' is not registered. "
            f"Available: {list_registered_metrics()}"
        )
    return METRIC_REGISTRY[name]


def list_registered_metrics() -> List[str]:
    """Return a sorted list of all registered metric names."""
    return sorted(METRIC_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Registered metrics
# ---------------------------------------------------------------------------

@register_metric
class MaskedMAE(ReconstructionMetric):
    """Mean absolute error on masked voxels."""
    name = "masked_mae"
    best = 0.0
    worst = float("inf")

    def __call__(self, reconstruction, target, mask, **kwargs):
        return _m.compute_masked_mae(reconstruction, target, mask)


@register_metric
class MaskedMSE(ReconstructionMetric):
    """Mean squared error on masked voxels."""
    name = "masked_mse"
    best = 0.0
    worst = float("inf")

    def __call__(self, reconstruction, target, mask, **kwargs):
        return _m.compute_masked_mse(reconstruction, target, mask)


@register_metric
class MaskedPSNR(ReconstructionMetric):
    """Peak signal-to-noise ratio on masked voxels (dB, higher is better)."""
    name = "masked_psnr"
    best = float("inf")
    worst = 0.0

    def __call__(self, reconstruction, target, mask, **kwargs):
        return _m.compute_masked_psnr(
            reconstruction, target, mask,
            data_range=kwargs.get("data_range", _c.SSIM_DATA_RANGE),
        )


@register_metric
class SSIM(ReconstructionMetric):
    """Structural similarity index over the full 3D volume."""
    name = "ssim"
    best = 1.0
    worst = -1.0

    def __call__(self, reconstruction, target, mask, **kwargs):
        return _m.compute_ssim(
            reconstruction, target,
            data_range=kwargs.get("data_range", _c.SSIM_DATA_RANGE),
        )
