"""Augmentation hyperparameters for MISFIT data loading.

Augmentation is kept deliberately light for MAE pretraining. The random
masking inside SwinMAE is the primary self-supervised signal; heavy spatial
or intensity augmentation on top of it adds noise without clear benefit.
Compare to contrastive methods (SimCLR, DINO) which rely on aggressive
augmentation to create positive pairs — MAE does not.

All probabilities and ranges are stored as a frozen dataclass so they can
be imported as a single constant without instantiation.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoadingConstants:
    """Frozen container for all augmentation hyperparameters."""

    # --- Spatial augmentation ---
    # Random flips are cheap and modality-agnostic. Left-right symmetry
    # holds approximately for most anatomical structures.
    FLIP_PROB: float = 0.5

    # --- Intensity augmentation ---
    # Light Gaussian noise simulates scanner noise variation across sites.
    NOISE_PROB: float = 0.15
    NOISE_STD_MIN: float = 0.0
    NOISE_STD_MAX: float = 0.1

    # Multiplicative brightness shift (±25%) simulates gain differences
    # across MRI sequences and CT reconstruction kernels.
    BRIGHTNESS_PROB: float = 0.15
    BRIGHTNESS_FACTOR: float = 0.25

    # --- Normalization ---
    NORM_EPS: float = 1e-8


# Module-level singleton — import and use directly.
dc = DataLoadingConstants()
