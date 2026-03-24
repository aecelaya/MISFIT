"""Default constants for learning rate schedulers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class LRSchedulerConstants:
    POLYNOMIAL_DECAY: float = 0.9
    WARMUP_START_FACTOR: float = 0.01  # LR starts at 1% of base LR
    CONSTANT_FACTOR: float = 1.0


lc = LRSchedulerConstants()
