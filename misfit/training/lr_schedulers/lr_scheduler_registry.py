"""Learning rate scheduler registry for MISFIT training.

Schedulers are stepped once per optimizer step, so a schedule's shape does
not depend on dataset size. All schedulers support an optional linear warmup
phase prepended via SequentialLR. Warmup is especially important for
SwinUNETR-V2 — the transformer's attention layers are sensitive to large
gradients early in training — and starts at 1% of the base LR.
"""
from collections.abc import Callable

import torch.optim as optim
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearLR,
    PolynomialLR,
    SequentialLR,
)

from misfit.training.lr_schedulers.lr_scheduler_constants import lc

LR_SCHEDULER_REGISTRY: dict[str, Callable] = {}


def register_lr_scheduler(name: str) -> Callable:
    """Decorator to register an LR scheduler builder function."""
    def decorator(fn: Callable) -> Callable:
        LR_SCHEDULER_REGISTRY[name.lower()] = fn
        return fn
    return decorator


def get_lr_scheduler(
    name: str,
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 0,
) -> optim.lr_scheduler.LRScheduler:
    """Build an LR scheduler, optionally prepending a linear warmup phase.

    Call ``scheduler.step()`` once per optimizer step. When warmup_steps > 0,
    the main scheduler is told it will run for (total_steps - warmup_steps)
    steps so that it decays to the intended minimum LR by the end of
    training.

    Args:
        name: Registered scheduler name (case-insensitive).
        optimizer: The optimizer whose LR will be scheduled.
        total_steps: Total optimizer steps (warmup + main schedule).
        warmup_steps: Number of linear warmup steps. The LR rises from
            1% of base LR to base LR over this period. Defaults to 0.

    Returns:
        Configured LR scheduler. If warmup_steps > 0, returns a
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

    main_steps = max(total_steps - warmup_steps, 1)
    main_scheduler = LR_SCHEDULER_REGISTRY[key](optimizer, main_steps)

    if warmup_steps > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=lc.WARMUP_START_FACTOR,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[warmup_steps],
        )

    return main_scheduler


def list_lr_schedulers() -> list[str]:
    """Return a sorted list of registered scheduler names."""
    return sorted(LR_SCHEDULER_REGISTRY.keys())


@register_lr_scheduler("cosine")
def _cosine(optimizer, steps):
    return CosineAnnealingLR(optimizer, T_max=steps)


@register_lr_scheduler("polynomial")
def _polynomial(optimizer, steps):
    return PolynomialLR(optimizer, total_iters=steps, power=lc.POLYNOMIAL_DECAY)


@register_lr_scheduler("constant")
def _constant(optimizer, steps):  # steps unused
    return ConstantLR(optimizer, factor=lc.CONSTANT_FACTOR)
