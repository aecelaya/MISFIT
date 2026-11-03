"""Base class for intensity normalization policies."""
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np


class IntensityScale(NamedTuple):
    """The affine map a policy applied: ``normalized = (raw - shift) / scale``.

    ``misfit_inspect`` inverts it (``raw = normalized * scale + shift``) to
    write reconstructions in the original intensity range.
    """

    shift: float
    scale: float


class NormalizationPolicy(ABC):
    """Maps a raw volume to the intensity range the model is trained on.

    Every code path that turns a NIfTI volume into model input (training,
    evaluation, inspection, encoding, embedding) applies the run's policy, so
    they all see identically normalized data.
    """

    # Minimum scale, to avoid dividing by zero on constant volumes.
    EPS: float = 1e-8

    @abstractmethod
    def normalize(
        self, volume: np.ndarray, row: Mapping
    ) -> tuple[np.ndarray, IntensityScale]:
        """Normalize one raw volume.

        Args:
            volume: Raw voxel array (X, Y, Z).
            row: The volume's row from the misfit_index Parquet index, for
                policies that use its precomputed statistics.

        Returns:
            Tuple of the normalized float32 volume (same shape) and the
            :class:`IntensityScale` applied.
        """
