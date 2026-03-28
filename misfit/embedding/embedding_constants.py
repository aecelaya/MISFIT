"""Constants for MISFIT embedding module."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class EmbeddingConstants:
    """Constants for aggregator training and inference."""

    # Minimum number of samples a group must have to be included in
    # contrastive training.  Groups with fewer samples are dropped with a
    # warning.  K=2 means every anchor always has exactly one positive.
    MIN_SAMPLES_PER_GROUP: int = 2

    # Temperature for the Supervised Contrastive loss.
    SUPCON_TEMPERATURE: float = 0.07

    # Output dimension of the contrastive projection head.
    PROJECTION_DIM: int = 128

    # Small epsilon to prevent log(0) in SupCon loss.
    SUPCON_EPS: float = 1e-8


ec = EmbeddingConstants()
