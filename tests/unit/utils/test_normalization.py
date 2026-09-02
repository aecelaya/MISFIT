"""Tests for misfit.utils.normalization."""
import pytest
import torch

from misfit.utils.normalization import denormalize_patchwise, normalize_patchwise


def _reference_normalize(target: torch.Tensor, p: int, eps: float = 1e-6) -> torch.Tensor:
    """The pre-refactor per-cube normalization, inlined, for equality checks."""
    B, C, D, H, W = target.shape
    gd, gh, gw = D // p, H // p, W // p
    n = gd * gh * gw
    t = target.reshape(B, C, gd, p, gh, p, gw, p)
    t = t.permute(0, 1, 2, 4, 6, 3, 5, 7).reshape(B, C, n, p * p * p)
    mean = t.mean(dim=-1, keepdim=True)
    std = t.std(dim=-1, keepdim=True)
    t = (t - mean) / (std + eps)
    t = t.reshape(B, C, gd, gh, gw, p, p, p)
    return t.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(B, C, D, H, W)


# ---------------------------------------------------------------------------
# normalize_patchwise
# ---------------------------------------------------------------------------

def test_matches_reference_implementation():
    torch.manual_seed(0)
    x = torch.randn(2, 1, 32, 32, 32)
    assert torch.equal(normalize_patchwise(x, 16), _reference_normalize(x, 16))


def test_accepts_3d_4d_5d_consistently():
    torch.manual_seed(1)
    x5 = torch.randn(1, 1, 32, 32, 32)
    n5 = normalize_patchwise(x5, 16)
    n4 = normalize_patchwise(x5[0], 16)
    n3 = normalize_patchwise(x5[0, 0], 16)
    assert n5.shape == x5.shape
    assert torch.equal(n4, n5[0])
    assert torch.equal(n3, n5[0, 0])


def test_each_cube_is_zero_mean_unit_std():
    torch.manual_seed(2)
    x = torch.randn(1, 1, 32, 32, 32) * 5 + 3
    out = normalize_patchwise(x, 16)
    for d in (0, 16):
        for h in (0, 16):
            for w in (0, 16):
                cube = out[0, 0, d:d + 16, h:h + 16, w:w + 16]
                assert abs(float(cube.mean())) < 1e-4
                assert abs(float(cube.std()) - 1.0) < 1e-3


def test_rejects_indivisible_dims():
    with pytest.raises(ValueError, match="divisible"):
        normalize_patchwise(torch.randn(1, 1, 24, 32, 32), 16)


def test_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="3D, 4D, or 5D"):
        normalize_patchwise(torch.randn(16, 16), 16)


# ---------------------------------------------------------------------------
# denormalize_patchwise
# ---------------------------------------------------------------------------

def test_round_trip_is_identity():
    torch.manual_seed(3)
    x = torch.randn(1, 1, 32, 32, 32) * 4 - 1
    rt = denormalize_patchwise(normalize_patchwise(x, 16), x, 16)
    assert torch.allclose(rt, x, atol=1e-5)


def test_uses_reference_per_cube_stats():
    # Constant cubes with distinct means; a unit-ones input should map each cube
    # back to (mean + std) = mean + 0 = mean.
    ref = torch.zeros(1, 1, 32, 16, 16)
    ref[0, 0, :16] = 10.0
    ref[0, 0, 16:] = -4.0
    out = denormalize_patchwise(torch.ones_like(ref), ref, 16)
    assert torch.allclose(out[0, 0, :16], torch.full((16, 16, 16), 10.0), atol=1e-3)
    assert torch.allclose(out[0, 0, 16:], torch.full((16, 16, 16), -4.0), atol=1e-3)


def test_denorm_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        denormalize_patchwise(
            torch.ones(1, 1, 32, 32, 32), torch.ones(1, 1, 16, 32, 32), 16
        )
