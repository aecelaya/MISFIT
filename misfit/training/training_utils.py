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
