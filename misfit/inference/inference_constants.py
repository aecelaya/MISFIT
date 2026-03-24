"""Constants for MISFIT inference modules."""
import dataclasses
from typing import FrozenSet


@dataclasses.dataclass(frozen=True)
class InferenceConstants:
    """Constants for MISFIT inference modules."""

    # Batch size used when running sliding-window inference.
    SLIDING_WINDOW_BATCH_SIZE: int = 1

    # Valid patch blend modes for sliding-window inference.
    SLIDING_WINDOW_PATCH_BLEND_MODES: FrozenSet[str] = dataclasses.field(
        default_factory=lambda: frozenset({"gaussian", "constant"})
    )

    # Default fractional overlap between sliding-window patches.
    DEFAULT_PATCH_OVERLAP: float = 0.5

    # Default blend mode for sliding-window inference.
    DEFAULT_BLEND_MODE: str = "gaussian"


ic = InferenceConstants()
