"""Image-level (SimMIM-style) masking shared by the MAE models.

Masked 3D cubes are replaced in the input image itself, before the encoder,
rather than by dropping tokens. This works for any backbone: windowed-attention
transformers (Swin), whose window partitioning breaks with irregular token
counts, and CNNs, which have no tokens to drop.
"""

import torch


def generate_patch_mask(
    x: torch.Tensor,
    mask_patch_size: int,
    mask_ratio: float,
) -> torch.Tensor:
    """Generate a random patch-level binary mask.

    Divides the volume into a grid of non-overlapping cubes of size
    mask_patch_size^3 and masks exactly int(mask_ratio * num_patches) of them,
    chosen independently per sample in the batch.

    Args:
        x: Input tensor of shape (B, C, D, H, W). Only batch size, spatial
            shape, and device are used. Spatial dims must be divisible by
            mask_patch_size.
        mask_patch_size: Edge length (in voxels) of each masked cube.
        mask_ratio: Fraction of cubes to mask.

    Returns:
        Float mask tensor of shape (B, 1, D, H, W), where 1.0 = masked and
        0.0 = visible.
    """
    B, _, D, H, W = x.shape
    p = mask_patch_size
    gd, gh, gw = D // p, H // p, W // p
    num_patches = gd * gh * gw
    num_masked = int(mask_ratio * num_patches)

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


def apply_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    mask_token: torch.Tensor,
) -> torch.Tensor:
    """Replace masked voxels with the mask token.

    Args:
        x: Input tensor of shape (B, C, D, H, W).
        mask: Binary mask of shape (B, 1, D, H, W); 1 = masked.
        mask_token: Learnable fill value, broadcastable to x (typically
            shape (1, C, 1, 1, 1)).

    Returns:
        Masked input with the same shape as x.
    """
    return x * (1.0 - mask) + mask_token * mask
