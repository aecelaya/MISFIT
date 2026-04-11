"""SwinUNETR-V2 encoder with a 3D Masked Autoencoder (MAE) pretraining head.

Architecture overview:

  1. Masking  — randomly replace mask_ratio of 3D patch blocks (size
               mask_patch_size^3) with a learnable scalar mask token.
  2. Encoder  — SwinUNETR-V2 swinViT backbone processes the masked image and
               returns 5 hidden states at spatial resolutions D/2 … D/32.
  3. Decoder  — lightweight convolutional decoder upsamples from the D/32
               bottleneck back to the original resolution in 5 steps.
  4. Loss     — MSE between reconstruction and original, computed only over
               masked voxels (returned alongside reconstruction for the
               training loop to handle).

This masking strategy follows SimMIM (Xie et al., 2022): masking is applied
at the image level before the encoder rather than by dropping tokens, which
is required for Swin because windowed attention breaks with irregular token
counts.

Transfer learning compatibility:
    get_encoder_state_dict() remaps keys from 'encoder.<name>' to
    'model.swinViT.<name>', matching the key format used by MIST's
    MistSwinUNETR checkpoints.  The returned state dict can be saved with
    torch.save and passed directly to ``mist_train --pretrained-weights``.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR

from misfit.models.base_model import MISFITModel


class MAEDecoder(nn.Module):
    """Lightweight convolutional decoder for 3D MAE reconstruction.

    Progressively upsamples the encoder bottleneck back to the original image
    resolution using a series of ConvTranspose3d + Conv3d + InstanceNorm +
    LeakyReLU blocks. Each block doubles the spatial resolution.

    Args:
        in_channels: Number of channels in the bottleneck feature map.
        out_channels: Number of output channels (must match the input image
            channel count).
        num_upsample: Number of 2x upsampling steps. For SwinUNETR-V2 (32x
            total downsampling) this should be 5. Defaults to 5.
        base_channels: Minimum channel count at any intermediate stage.
            Prevents the channel schedule from collapsing to 0. Defaults to 32.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_upsample: int = 5,
        base_channels: int = 32,
    ):
        super().__init__()

        # Build a halving channel schedule floored at base_channels.
        channels = [
            max(in_channels // (2 ** i), base_channels)
            for i in range(num_upsample + 1)
        ]

        self.blocks = nn.ModuleList()
        for i in range(num_upsample):
            self.blocks.append(nn.Sequential(
                nn.ConvTranspose3d(
                    channels[i], channels[i + 1], kernel_size=2, stride=2
                ),
                nn.Conv3d(
                    channels[i + 1], channels[i + 1],
                    kernel_size=3, padding=1, bias=False,
                ),
                nn.InstanceNorm3d(channels[i + 1]),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            ))

        # Final 1x1 projection to the target channel count.
        self.head = nn.Conv3d(channels[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample bottleneck features to the original image resolution.

        Args:
            x: Bottleneck tensor of shape (B, in_channels, D', H', W').

        Returns:
            Reconstructed tensor of shape (B, out_channels, D, H, W).
        """
        for block in self.blocks:
            x = block(x)
        return self.head(x)


class SwinMAE(MISFITModel):
    """SwinUNETR-V2 encoder with a 3D Masked Autoencoder pretraining head.

    Randomly masks 3D patches of the input volume, encodes the masked image
    with the SwinUNETR-V2 swinViT backbone, and reconstructs the original
    intensities from the bottleneck features using a lightweight convolutional
    decoder.

    Spatial input requirements:
        - All img_size dimensions must be divisible by 32 (SwinUNETR
          downsamples by patch_size=2 × 2^4 stages = 32).
        - All img_size dimensions must be divisible by mask_patch_size.

    Args:
        in_channels: Number of input image channels. Defaults to 1.
        feature_size: Swin ViT base feature dimension. Controls capacity:
            24 (small), 48 (base), 96 (large). Defaults to 48.
        img_size: Spatial dimensions of the input volume (D, H, W). All
            dimensions must be divisible by 32. Defaults to (96, 96, 96).
        mask_patch_size: Edge length (in voxels) of each masked 3D cube.
            Must evenly divide all dimensions of img_size. Defaults to 16.
        mask_ratio: Fraction of patches to randomly mask during training.
            Defaults to 0.75 (standard MAE setting).

    Raises:
        ValueError: If any img_size dimension is not divisible by 32 or by
            mask_patch_size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        feature_size: int = 48,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        mask_patch_size: int = 16,
        mask_ratio: float = 0.75,
        **kwargs: Any,
    ):
        super().__init__()

        # Validate spatial dimensions.
        for dim in img_size:
            if dim % 32 != 0:
                raise ValueError(
                    f"All img_size dimensions must be divisible by 32 "
                    f"(SwinUNETR-V2 total downsampling factor). "
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

        # Build SwinUNETR-V2 and extract the swinViT backbone. The UNet
        # decoder is discarded; out_channels=2 is a placeholder.
        # MONAI >= 1.5 removed img_size from SwinUNETR; older versions require
        # it. Try without first, fall back to passing it for compatibility.
        try:
            _swinunetr = SwinUNETR(
                in_channels=in_channels,
                out_channels=2,
                feature_size=feature_size,
                use_v2=True,
                spatial_dims=3,
            )
        except TypeError:
            _swinunetr = SwinUNETR(
                img_size=img_size,
                in_channels=in_channels,
                out_channels=2,
                feature_size=feature_size,
                use_v2=True,
                spatial_dims=3,
            )
        self.encoder = _swinunetr.swinViT

        # Learnable mask token: scalar per channel, broadcast over spatial dims.
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels, 1, 1, 1))

        # Probe the encoder to determine bottleneck channel count. This is
        # safer than hardcoding 8*feature_size across MONAI versions.
        with torch.no_grad():
            _dummy = torch.zeros(1, in_channels, *img_size)
            _hidden = self.encoder(_dummy)
            _bottleneck_channels = _hidden[4].shape[1]

        # SwinUNETR-V2 downsamples by 32x total → 5 upsample steps restore
        # the original resolution.
        self.decoder = MAEDecoder(
            in_channels=_bottleneck_channels,
            out_channels=in_channels,
            num_upsample=5,
        )

    def generate_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Generate a random patch-level binary mask.

        Divides the volume into a grid of non-overlapping cubes of size
        mask_patch_size^3 and independently masks each cube with probability
        mask_ratio. Masking is random and independent per sample in the batch.

        Args:
            x: Input tensor of shape (B, C, D, H, W). Only batch size,
                spatial shape, and device are used.

        Returns:
            Float mask tensor of shape (B, 1, D, H, W), where 1.0 = masked
            and 0.0 = visible.
        """
        B, _, D, H, W = x.shape
        p = self.mask_patch_size
        gd, gh, gw = D // p, H // p, W // p
        num_patches = gd * gh * gw
        num_masked = int(self.mask_ratio * num_patches)

        # Rank patches by random noise; take the top num_masked as masked.
        noise = torch.rand(B, num_patches, device=x.device)
        ids_sorted = torch.argsort(noise, dim=1)
        mask_flat = torch.zeros(B, num_patches, device=x.device)
        mask_flat.scatter_(1, ids_sorted[:, :num_masked], 1.0)

        # Reshape from flat patch index → patch grid → voxel resolution.
        mask = mask_flat.view(B, 1, gd, gh, gw)
        mask = mask.repeat_interleave(p, dim=2)
        mask = mask.repeat_interleave(p, dim=3)
        mask = mask.repeat_interleave(p, dim=4)
        return mask  # (B, 1, D, H, W)

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W). All spatial dimensions
                must match img_size.
            mask_ratio: Override the instance mask_ratio for this call.
                Useful for curriculum training schedules. If None, uses the
                value set at construction. Defaults to None.

        Returns:
            Dict with:
                ``"reconstruction"``: Reconstructed volume (B, C, D, H, W).
                ``"mask"``: Binary mask (B, 1, D, H, W); 1 = masked voxels.
                    Use this to compute the loss over masked regions only:
                    ``loss = F.mse_loss(recon[mask], x[mask])``.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        # 1. Generate mask and replace masked voxels with the mask token.
        mask = self.generate_mask(x)
        masked_x = x * (1.0 - mask) + self.mask_token * mask

        # 2. Encode. swinViT returns a list of 5 hidden states at spatial
        #    resolutions D/2, D/4, D/8, D/16, D/32.
        hidden_states = self.encoder(masked_x)
        bottleneck = hidden_states[4]  # (B, C', D/32, H/32, W/32)

        # 3. Decode from bottleneck back to original resolution.
        reconstruction = self.decoder(bottleneck)

        return {"reconstruction": reconstruction, "mask": mask}

    def get_encoder_state_dict(self) -> OrderedDict:
        """Return encoder-only weights for downstream transfer learning.

        Remaps keys from ``encoder.<name>`` to ``model.swinViT.<name>`` so
        the returned state dict is directly compatible with MIST's
        ``MistSwinUNETR`` checkpoint format.  The remapped state dict can be
        saved with ``torch.save`` and passed to ``mist_train
        --pretrained-weights`` without any additional key manipulation.

        Returns:
            OrderedDict mapping ``model.swinViT.<name>`` → tensor.
        """
        src_prefix = "encoder."
        dst_prefix = "model.swinViT."
        return OrderedDict(
            {dst_prefix + k[len(src_prefix):]: v
             for k, v in self.state_dict().items()
             if k.startswith(src_prefix)}
        )
