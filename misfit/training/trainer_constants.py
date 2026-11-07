"""Training constants for MISFIT."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainerConstants:
    """Frozen container for training hyperparameter constants."""

    NO_AMP_EPS: float = 1e-8

    # Gradient clipping max norm. Prevents exploding gradients, especially
    # important early in training when the Swin ViT encoder is freshly
    # initialized.
    GRAD_CLIP_VALUE: float = 1.0

    # Warn when a validation loss is more than this many times the lowest so far.
    # A judgment call: well above noise, since validation repeats the same masks
    # on the same volumes, but it only warns, so a false alarm costs little.
    VAL_DRIFT_WARNING_RATIO: float = 1.5

    # Seed base. Each rank adds its rank index to get a unique but
    # reproducible RNG stream per process.
    DEFAULT_SEED: int = 42


tc = TrainerConstants()
