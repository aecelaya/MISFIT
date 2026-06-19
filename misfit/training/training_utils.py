"""Shared training utilities for MISFIT."""
import random

import numpy as np
import torch


class RunningMean:
    """Incremental mean tracker for loss values within an epoch.

    Avoids accumulating a list of loss values, which would grow unboundedly
    for large datasets. Reset between epochs.

    Example::

        meter = RunningMean()
        for batch in loader:
            loss = compute_loss(batch)
            meter.update(loss.item())
        print(meter.value)  # mean over all batches
    """

    def __init__(self) -> None:
        self._sum = 0.0
        self._count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Accumulate one or more observations.

        Args:
            value: Loss value (already averaged over the batch, or a sum).
            n: Number of observations represented by value. Defaults to 1.
        """
        self._sum += value * n
        self._count += n

    @property
    def value(self) -> float:
        """Current mean. Returns 0.0 if no observations have been added."""
        return self._sum / self._count if self._count > 0 else 0.0

    def reset(self) -> None:
        """Reset the accumulator for the next epoch."""
        self._sum = 0.0
        self._count = 0


def build_accumulation_plan(
    num_batches: int, accum_steps: int
) -> list[tuple[int, bool]]:
    """Per-batch ``(window_size, is_window_end)`` for gradient accumulation.

    Guarantees two properties that a naive ``micro_count % accum_steps == 0``
    schedule gets wrong on the trailing partial window:

    1. The **final batch always closes its window** (``is_window_end=True``).
       Under DDP the closing micro-step runs with gradient synchronisation, so
       the optimizer never steps on un-all-reduced gradients (which would let
       ranks silently diverge).
    2. Each window's loss is normalised by its **actual** size, so the trailing
       window of ``num_batches % accum_steps`` micro-steps is scaled correctly
       rather than by the full ``accum_steps``.

    Args:
        num_batches: Number of batches the epoch will iterate.
        accum_steps: Micro-batches accumulated per optimizer step (>= 1).

    Returns:
        A list of length ``num_batches``; entry ``i`` is the
        ``(window_size, is_window_end)`` for the i-th batch.

    Raises:
        ValueError: If ``accum_steps < 1``.
    """
    if accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {accum_steps}.")

    full = num_batches - (num_batches % accum_steps)
    plan: list[tuple[int, bool]] = []
    for i in range(num_batches):
        window_size = accum_steps if i < full else num_batches - full
        is_window_end = ((i + 1) % accum_steps == 0) or (i == num_batches - 1)
        plan.append((window_size, is_window_end))
    return plan


def set_seed(seed: int, rank: int = 0) -> None:
    """Set all random seeds with a per-rank offset for DDP reproducibility.

    Each rank receives a unique seed (seed + rank) so that data shuffling
    and dropout are independent across processes, while remaining fully
    reproducible given the same base seed.

    Args:
        seed: Base random seed.
        rank: Process rank. Defaults to 0 (single-GPU / main process).
    """
    final_seed = seed + rank
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    torch.cuda.manual_seed_all(final_seed)
