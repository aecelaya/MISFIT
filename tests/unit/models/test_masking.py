"""Tests for misfit.models.masking (shared image-level MAE masking)."""
import pytest
import torch

from misfit.models.masking import apply_mask, generate_patch_mask
from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE


def test_generate_patch_mask_shape_and_values():
    x = torch.randn(2, 3, 32, 32, 16)
    mask = generate_patch_mask(x, mask_patch_size=16, mask_ratio=0.5)
    assert mask.shape == (2, 1, 32, 32, 16)
    assert ((mask == 0.0) | (mask == 1.0)).all()


@pytest.mark.parametrize("mask_ratio,expected", [(0.0, 0), (0.5, 2), (0.75, 3), (1.0, 4)])
def test_generate_patch_mask_masks_exact_patch_count(mask_ratio, expected):
    # 32×32×16 with 16³ cubes → 2×2×1 = 4 patches per sample.
    x = torch.randn(3, 1, 32, 32, 16)
    mask = generate_patch_mask(x, mask_patch_size=16, mask_ratio=mask_ratio)
    patches = mask[:, 0, ::16, ::16, ::16]
    assert (patches.flatten(1).sum(dim=1) == expected).all()


def test_generate_patch_mask_cubes_are_uniform():
    """Every voxel within a cube shares the cube's masked/visible state."""
    x = torch.randn(1, 1, 32, 32, 32)
    mask = generate_patch_mask(x, mask_patch_size=16, mask_ratio=0.5)
    cubes = mask.view(1, 1, 2, 16, 2, 16, 2, 16)
    assert (cubes.amin(dim=(3, 5, 7)) == cubes.amax(dim=(3, 5, 7))).all()


def test_generate_patch_mask_is_independent_per_sample():
    torch.manual_seed(0)
    x = torch.randn(8, 1, 64, 64, 64)
    mask = generate_patch_mask(x, mask_patch_size=16, mask_ratio=0.5)
    assert not all(torch.equal(mask[0], mask[i]) for i in range(1, 8))


def test_apply_mask_replaces_only_masked_voxels():
    x = torch.randn(2, 2, 16, 16, 16)
    mask = torch.zeros(2, 1, 16, 16, 16)
    mask[:, :, :8] = 1.0
    token = torch.tensor([3.0, -3.0]).view(1, 2, 1, 1, 1)
    out = apply_mask(x, mask, token)
    assert torch.equal(out[:, :, 8:], x[:, :, 8:])
    assert (out[:, 0, :8] == 3.0).all() and (out[:, 1, :8] == -3.0).all()


def test_swinmae_generate_mask_delegates_to_shared_masking():
    """SwinMAE's seeded masks are identical to the shared function's."""
    model = SwinMAE(feature_size=12, img_size=(32, 32, 32), mask_patch_size=16,
                    mask_ratio=0.75)
    x = torch.randn(2, 1, 32, 32, 32)
    torch.manual_seed(123)
    from_model = model.generate_mask(x)
    torch.manual_seed(123)
    from_shared = generate_patch_mask(x, mask_patch_size=16, mask_ratio=0.75)
    assert torch.equal(from_model, from_shared)
