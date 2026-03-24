"""Default constants for optimizers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizerConstants:
    SGD_MOMENTUM: float = 0.9
    SGD_NESTEROV: bool = True


oc = OptimizerConstants()
