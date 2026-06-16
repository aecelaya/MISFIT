"""Training constants for MISFIT."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConstants:
    """Frozen container for training hyperparameter constants."""

    NO_AMP_EPS: float = 1e-8

    # Gradient clipping max norm. Prevents exploding gradients, especially
    # important early in training when the Swin ViT encoder is freshly
    # initialised.
    GRAD_CLIP_VALUE: float = 1.0

    # Seed base. Each rank adds its rank index to get a unique but
    # reproducible RNG stream per process.
    DEFAULT_SEED: int = 42


tc = TrainerConstants()
