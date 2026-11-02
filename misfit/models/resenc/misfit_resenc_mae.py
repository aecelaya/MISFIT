"""Residual Encoder UNet (ResEnc) with a 3D Masked Autoencoder (MAE) objective.

Architecture overview:

  1. Masking  — randomly replace mask_ratio of 3D patch blocks (size
               mask_patch_size^3) with a learnable scalar mask token.
  2. Network  — MIST's full ResEnc UNet (residual encoder, plain UNet decoder
               with skip connections) processes the masked image.
  3. Head     — the decoder's full-resolution 1x1x1 layer outputs the
               reconstructed intensities.
  4. Loss     — computed by the training loop over masked voxels only.

This follows the MAE recipe nnssl uses for ResEnc pretraining: masking is
dense (applied to the image, since a CNN has no tokens to drop) and the whole
UNet, skip connections included, reconstructs the image. The skips cannot leak
the target because masked regions are blank at every resolution.

Topology:
    MIST's ResEnc derives its stages, strides, and kernel sizes from the patch
    size and target spacing. With ``--target-spacing`` set, MISFIT plans from
    that spacing, exactly as MIST would for data preprocessed to it. Without
    one, volumes stay at their native (mixed) resolutions and the topology is
    planned isotropically (equal spacing on every axis), so it depends only on
    img_size. Only the ratios between axes matter for planning. Fine-tuning a
    MIST resenc whose planned patch size or spacing differs still transfers
    every shared stage exactly (MIST warns about the mismatch rather than
    failing).

Transfer learning compatibility:
    The encoder is MIST's ResidualEncoder held as ``self.encoder``, so
    get_encoder_state_dict() returns MIST's own ``encoder.*`` keys unchanged.
    The returned state dict can be saved with torch.save and passed directly to
    ``mist_train --pretrained-weights``.
"""

from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn
from mist.models.resenc.mist_resenc import ResEncUNet

from misfit.models.base_model import MISFITModel
from misfit.models.masking import apply_mask, generate_patch_mask

# Spacing used to plan the ResEnc topology when no target spacing is set
# (native resolution): equal on every axis, i.e. an isotropic topology.
ISOTROPIC_PLANNING_SPACING = (1.0, 1.0, 1.0)


class ResEncMAE(MISFITModel):
    """ResEnc UNet with a 3D Masked Autoencoder pretraining objective.

    Spatial input requirements:
        - All img_size dimensions must be divisible by 32 (ResEnc downsamples
          by up to 2^5 = 32).
        - All img_size dimensions must be divisible by mask_patch_size.

    Args:
        in_channels: Number of input image channels. Defaults to 1.
        img_size: Spatial dimensions of the input volume (D, H, W). All
            dimensions must be divisible by 32. Defaults to (96, 96, 96).
        mask_patch_size: Edge length (in voxels) of each masked 3D cube.
            Must evenly divide all dimensions of img_size. Defaults to 16.
        mask_ratio: Fraction of patches to randomly mask during training.
            Defaults to 0.75 (standard MAE setting).
        target_spacing: (x, y, z) spacing the data is resampled to
            (``--target-spacing``), used to plan the topology as MIST does.
            None (native resolution) plans an isotropic topology. Defaults to
            None.

    Raises:
        ValueError: If any img_size dimension is not divisible by 32 or by
            mask_patch_size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        img_size: tuple[int, int, int] = (96, 96, 96),
        mask_patch_size: int = 16,
        mask_ratio: float = 0.75,
        target_spacing: tuple[float, float, float] | None = None,
        **kwargs: Any,
    ):
        super().__init__()

        # Validate spatial dimensions.
        for dim in img_size:
            if dim % 32 != 0:
                raise ValueError(
                    f"All img_size dimensions must be divisible by 32 "
                    f"(ResEnc total downsampling factor). "
                    f"Got img_size={img_size}."
                )
            if dim % mask_patch_size != 0:
                raise ValueError(
                    f"All img_size dimensions must be divisible by "
                    f"mask_patch_size={mask_patch_size}. "
                    f"Got img_size={img_size}."
                )

        self.in_channels = in_channels
        self.img_size = img_size
        self.mask_patch_size = mask_patch_size
        self.mask_ratio = mask_ratio

        # Build MIST's ResEnc UNet with one output channel per input channel,
        # so its full-resolution segmentation layer becomes the reconstruction
        # head. Deep supervision is off; MAE reconstructs at full resolution.
        _mist_resenc = ResEncUNet(
            in_channels=in_channels,
            out_channels=in_channels,
            patch_size=img_size,
            target_spacing=(
                ISOTROPIC_PLANNING_SPACING if target_spacing is None
                else tuple(target_spacing)
            ),
            use_deep_supervision=False,
        )
        self.encoder = _mist_resenc.encoder
        self.decoder = _mist_resenc.decoder
        self.head = self.decoder.seg_layers[-1]
        # The lower-resolution segmentation layers only serve deep supervision.
        # Drop them: they would never receive gradients, which DDP rejects.
        del self.decoder.seg_layers

        # Learnable mask token: scalar per channel, broadcast over spatial dims.
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels, 1, 1, 1))

    def generate_mask(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> torch.Tensor:
        """Generate a random patch-level binary mask.

        Args:
            x: Input tensor of shape (B, C, D, H, W). Only batch size,
                spatial shape, and device are used.
            mask_ratio: Fraction of patches to mask. If None, uses the value
                set at construction (``self.mask_ratio``). Defaults to None.

        Returns:
            Float mask tensor of shape (B, 1, D, H, W), where 1.0 = masked
            and 0.0 = visible.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        return generate_patch_mask(x, self.mask_patch_size, mask_ratio)

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W). All spatial dimensions
                must match img_size.
            mask_ratio: Override the instance mask_ratio for this call. If
                None, uses the value set at construction. Defaults to None.

        Returns:
            Dict with:
                ``"reconstruction"``: Reconstructed volume (B, C, D, H, W).
                ``"mask"``: Binary mask (B, 1, D, H, W); 1 = masked voxels.
        """
        mask = self.generate_mask(x, mask_ratio=mask_ratio)
        masked_x = apply_mask(x, mask, self.mask_token)

        # The encoder returns every stage's output (full → lowest resolution);
        # the decoder returns its stage outputs ordered the same way.
        decoder_outputs = self.decoder(self.encoder(masked_x))
        reconstruction = self.head(decoder_outputs[0])

        return {"reconstruction": reconstruction, "mask": mask}

    def get_encoder_state_dict(self) -> OrderedDict:
        """Return encoder-only weights for downstream transfer learning.

        Keys are ``encoder.<name>``, exactly as in MIST's ``ResEncUNet``, so
        the state dict can be saved with ``torch.save`` and passed to
        ``mist_train --pretrained-weights`` without key manipulation.

        Returns:
            OrderedDict mapping ``encoder.<name>`` → tensor.
        """
        return OrderedDict(
            (k, v) for k, v in self.state_dict().items() if k.startswith("encoder.")
        )
