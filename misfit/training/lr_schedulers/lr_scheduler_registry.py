"""Learning rate scheduler registry for MISFIT training.

All schedulers support an optional linear warmup phase prepended via
SequentialLR. Warmup is especially important for SwinUNETR-V2 — the
transformer's attention layers are sensitive to large gradients early in
training. A 5–20 epoch warmup (starting at 1% of base LR) stabilises
training significantly.
"""
from typing import Callable, Dict, List

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ConstantLR,
    LinearLR,
    PolynomialLR,
    SequentialLR,
)

from misfit.training.lr_schedulers.lr_scheduler_constants import lc

LR_SCHEDULER_REGISTRY: Dict[str, Callable] = {}


def register_lr_scheduler(name: str) -> Callable:
    """Decorator to register an LR scheduler builder function."""
    def decorator(fn: Callable) -> Callable:
        LR_SCHEDULER_REGISTRY[name.lower()] = fn
        return fn
    return decorator


def get_lr_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 0,
) -> optim.lr_scheduler.LRScheduler:
    """Build an LR scheduler, optionally prepending a linear warmup phase.

    When warmup_epochs > 0, the main scheduler is told it will run for
    (epochs - warmup_epochs) steps so that it decays to the intended
    minimum LR by the end of training.

    Args:
        name: Registered scheduler name (case-insensitive).
        optimizer: The optimizer whose LR will be scheduled.
        epochs: Total training epochs (warmup + main schedule).
        warmup_epochs: Number of linear warmup epochs. The LR rises from
            1% of base LR to base LR over this period. Defaults to 0.

    Returns:
        Configured LR scheduler. If warmup_epochs > 0, returns a
        SequentialLR that chains warmup → main schedule.

    Raises:
        ValueError: If the name is not registered.
    """
    key = name.lower()
    if key not in LR_SCHEDULER_REGISTRY:
        raise ValueError(
            f"LR scheduler '{name}' is not registered. "
            f"Available: {sorted(LR_SCHEDULER_REGISTRY.keys())}"
        )

    main_epochs = max(epochs - warmup_epochs, 1)
    main_scheduler = LR_SCHEDULER_REGISTRY[key](optimizer, main_epochs)

    if warmup_epochs > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=lc.WARMUP_START_FACTOR,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_epochs],
        )

    return main_scheduler


def list_lr_schedulers() -> List[str]:
    """Return a sorted list of registered scheduler names."""
    return sorted(LR_SCHEDULER_REGISTRY.keys())


@register_lr_scheduler("cosine")
def _cosine(optimizer, epochs):
    return CosineAnnealingLR(optimizer, T_max=epochs)


@register_lr_scheduler("polynomial")
def _polynomial(optimizer, epochs):
    return PolynomialLR(optimizer, total_iters=epochs, power=lc.POLYNOMIAL_DECAY)


@register_lr_scheduler("constant")
def _constant(optimizer, epochs):  # epochs unused
    return ConstantLR(optimizer, factor=lc.CONSTANT_FACTOR)
