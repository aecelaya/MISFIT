"""Embedder — tiles a volume into crops, encodes each, aggregates to one vector."""
from collections.abc import Callable

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
       3-D positions to the
       :class:`~misfit.embedding.aggregators.base.AbstractAggregator`.

    Args:
        encoder_fn: Callable ``(1, 1, D, H, W) → (1, C, D', H', W')`` that
            maps a single-crop batch to a bottleneck feature map.  Typically
            ``lambda x: model.encoder(x)[-1]``.
        aggregator: Trained (or zero-shot) aggregator module.
        patch_size: Crop size in voxels. Either a single int (cubic crop) or a
            ``(D, H, W)`` sequence for anisotropic crops (default ``96``).
        device: Device on which to run forward passes.
    """

    def __init__(
        self,
        encoder_fn: Callable[[torch.Tensor], torch.Tensor],
        aggregator: AbstractAggregator,
        patch_size: int | tuple[int, int, int] = 96,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.encoder_fn = encoder_fn
        self.aggregator = aggregator
        self.patch_size = (
            (patch_size, patch_size, patch_size)
            if isinstance(patch_size, int)
            else tuple(patch_size)
        )
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
    ) -> tuple[np.ndarray, np.ndarray]:
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
        pd_, ph_, pw_ = self.patch_size

        pad_d = (pd_ - D % pd_) % pd_
        pad_h = (ph_ - H % ph_) % ph_
        pad_w = (pw_ - W % pw_) % pw_

        if pad_d or pad_h or pad_w:
            # torch.nn.functional.pad expects (left, right, top, bottom, front, back)
            volume = torch.nn.functional.pad(
                volume, (0, pad_w, 0, pad_h, 0, pad_d)
            )
        return volume

    def _tile_crops(
        self, volume: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split *volume* into non-overlapping ``patch_size³`` crops.

        Args:
            volume: ``(1, D', H', W')`` — already padded.

        Returns:
            crops:    ``(N_crops, 1, Pd, Ph, Pw)``
            centres:  ``(N_crops, 3)`` normalised 3-D crop centre coordinates.
        """
        _, D, H, W = volume.shape
        pd_, ph_, pw_ = self.patch_size

        nd, nh, nw = D // pd_, H // ph_, W // pw_
        crops_list = []
        centres_list = []

        for id_ in range(nd):
            for ih in range(nh):
                for iw in range(nw):
                    d0, h0, w0 = id_ * pd_, ih * ph_, iw * pw_
                    crop = volume[:, d0:d0 + pd_, h0:h0 + ph_, w0:w0 + pw_]
                    crops_list.append(crop.unsqueeze(0))  # (1, 1, Pd, Ph, Pw)

                    # Normalized centre coordinate in [0, 1].
                    cd = (d0 + pd_ / 2) / D
                    ch = (h0 + ph_ / 2) / H
                    cw = (w0 + pw_ / 2) / W
                    centres_list.append([cd, ch, cw])

        crops = torch.cat(crops_list, dim=0)  # (N, 1, P, P, P)
        centres = torch.tensor(centres_list, dtype=torch.float32)  # (N, 3)
        return crops, centres

    def _extract_crop_features(
        self, volume: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tile, encode, and GAP-pool all crops.

        Args:
            volume: ``(1, D, H, W)`` float32 tensor.

        Returns:
            features:  ``(N_crops, C)`` on ``self.device``.
            positions: ``(N_crops, 3)`` on ``self.device``.
        """
        volume = volume.to(self.device)
        padded = self._pad_volume(volume)
        crops, positions = self._tile_crops(padded)  # (N, 1, P, P, P), (N, 3)

        features_list = []
        for crop in crops:
            # crop after iteration: (1, P, P, P) → unsqueeze → (1, 1, P, P, P)
            feat_map = self.encoder_fn(crop.unsqueeze(0))
            # Global average pool over spatial dims: (1, C)
            feat = feat_map.mean(dim=(2, 3, 4))
            features_list.append(feat)

        features = torch.cat(features_list, dim=0)  # (N, C)
        return features, positions.to(self.device)
