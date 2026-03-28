"""Training constants for MISFIT."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConstants:
    """Frozen container for training hyperparameter constants."""

    # Optimizer epsilon — float16 needs a larger eps to avoid NaN in AMP.
    # bfloat16 and full-precision runs use standard 1e-8 (same dynamic range
    # as float32, so gradient underflow is not a concern).
    AMP_FP16_EPS: float = 1e-4
    NO_AMP_EPS: float = 1e-8

    # Gradient clipping max norm. Prevents exploding gradients, especially
    # important early in training when the Swin ViT encoder is freshly
    # initialised.
    GRAD_CLIP_VALUE: float = 1.0

    # Seed base. Each rank adds its rank index to get a unique but
    # reproducible RNG stream per process.
    DEFAULT_SEED: int = 42


tc = TrainerConstants()
