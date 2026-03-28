"""Embedder — tiles a volume into crops, encodes each, aggregates to one vector."""
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from misfit.embedding.aggregators.base import AbstractAggregator


class Embedder(nn.Module):
    """Produce a single global embedding for a whole 3-D volume.

    Pipeline
    --------
    1. **Tile** — divide the padded volume into non-overlapping
       ``patch_size³`` crops.
    2. **Encode** — run each crop through ``encoder_fn`` (e.g. the MAE encoder),
       which returns a 5-D feature map ``(1, C, D', H', W')``.
    3. **Pool** — global-average-pool the spatial dimensions to get ``(C,)``
       per crop.
    4. **Aggregate** — pass the full ``(N_crops, C)`` sequence + normalised
       3-D positions to the :class:`~misfit.embedding.aggregators.base.AbstractAggregator`.

    Args:
        encoder_fn: Callable ``(1, 1, D, H, W) → (1, C, D', H', W')`` that
            maps a single-crop batch to a bottleneck feature map.  Typically
            ``lambda x: model.encoder(x)[-1]``.
        aggregator: Trained (or zero-shot) aggregator module.
        patch_size: Edge length of each cubic crop in voxels (default ``96``).
        device: Device on which to run forward passes.
    """

    def __init__(
        self,
        encoder_fn: Callable[[torch.Tensor], torch.Tensor],
        aggregator: AbstractAggregator,
        patch_size: int = 96,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.encoder_fn = encoder_fn
        self.aggregator = aggregator
        self.patch_size = patch_size
        self.device = device or torch.device("cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def embed(self, volume: torch.Tensor) -> torch.Tensor:
        """Compute a global embedding for *volume*.

        Args:
            volume: ``(1, D, H, W)`` float32 tensor (single channel, already
                normalised to z-score).

        Returns:
            ``(C,)`` global embedding vector.
        """
        features, positions = self._extract_crop_features(volume)
        return self.aggregator(
            crop_features=features,
            positions=positions,
            padding_mask=None,  # no padding — all crops are valid
        )

    @torch.no_grad()
    def extract_crop_features(
        self, volume: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return per-crop features and positions as numpy arrays.

        Suitable for caching to ``.npz`` files used by
        :class:`~misfit.embedding.objectives.base.CropFeaturesDataset`.

        Args:
            volume: ``(1, D, H, W)`` float32 tensor.

        Returns:
            features:  ``(N_crops, C)`` float32 ndarray.
            positions: ``(N_crops, 3)`` float32 ndarray.
        """
        features, positions = self._extract_crop_features(volume)
        return features.cpu().numpy(), positions.cpu().numpy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """Pad *volume* so each spatial dimension is divisible by patch_size.

        Args:
            volume: ``(1, D, H, W)``

        Returns:
            Padded ``(1, D', H', W')`` where D', H', W' are multiples of
            ``patch_size``.
        """
        _, D, H, W = volume.shape
        p = self.patch_size

        pad_d = (p - D % p) % p
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p

        if pad_d or pad_h or pad_w:
            # torch.nn.functional.pad expects (left, right, top, bottom, front, back)
            volume = torch.nn.functional.pad(
                volume, (0, pad_w, 0, pad_h, 0, pad_d)
            )
        return volume

    def _tile_crops(
        self, volume: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split *volume* into non-overlapping ``patch_size³`` crops.

        Args:
            volume: ``(1, D', H', W')`` — already padded.

        Returns:
            crops:    ``(N_crops, 1, P, P, P)``
            centres:  ``(N_crops, 3)`` normalised 3-D crop centre coordinates.
        """
        _, D, H, W = volume.shape
        p = self.patch_size

        nd, nh, nw = D // p, H // p, W // p
        crops_list = []
        centres_list = []

        for id_ in range(nd):
            for ih in range(nh):
                for iw in range(nw):
                    d0, h0, w0 = id_ * p, ih * p, iw * p
                    crop = volume[:, d0:d0 + p, h0:h0 + p, w0:w0 + p]
                    crops_list.append(crop.unsqueeze(0))  # (1, 1, P, P, P)

                    # Normalised centre coordinate in [0, 1].
                    cd = (d0 + p / 2) / D
                    ch = (h0 + p / 2) / H
                    cw = (w0 + p / 2) / W
                    centres_list.append([cd, ch, cw])

        crops = torch.cat(crops_list, dim=0)  # (N, 1, P, P, P)
        centres = torch.tensor(centres_list, dtype=torch.float32)  # (N, 3)
        return crops, centres

    def _extract_crop_features(
        self, volume: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tile, encode, and GAP-pool all crops.

        Args:
            volume: ``(1, D, H, W)`` float32 tensor.

        Returns:
            features:  ``(N_crops, C)`` on ``self.device``.
            positions: ``(N_crops, 3)`` on CPU.
        """
        volume = volume.to(self.device)
        padded = self._pad_volume(volume)
        crops, positions = self._tile_crops(padded)  # (N, 1, P, P, P), (N, 3)

        features_list = []
        for crop in crops:
            # crop after iteration: (1, P, P, P) → unsqueeze → (1, 1, P, P, P)
            feat_map = self.encoder_fn(crop.unsqueeze(0).to(self.device))
            # Global average pool over spatial dims: (1, C)
            feat = feat_map.mean(dim=(2, 3, 4))
            features_list.append(feat)

        features = torch.cat(features_list, dim=0)  # (N, C)
        return features, positions.to(self.device)
