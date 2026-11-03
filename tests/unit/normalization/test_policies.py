"""Tests for misfit.normalization (policies and registry)."""
import numpy as np
import pytest
from scipy.ndimage import binary_fill_holes

from misfit.normalization.base import IntensityScale, NormalizationPolicy
from misfit.normalization.normalization_registry import (
    DEFAULT_NORMALIZATION_POLICY,
    get_normalization_policy,
    get_normalization_policy_name,
    list_normalization_policies,
    register_normalization_policy,
)

ROW = {"p1": -1.0, "p99": 3.0, "fg_mean": 0.5, "fg_std": 2.0}


def _skull_stripped_volume(seed=0):
    """MRI-like: a nonzero 'brain' (with an interior hole) in zero padding."""
    rng = np.random.default_rng(seed)
    volume = np.zeros((20, 24, 16), np.float32)
    volume[3:15, 5:20, 2:12] = rng.normal(300, 80, (12, 15, 10))
    volume[8, 10, 6] = 0.0  # an interior zero: inside the box regardless
    return volume


def _nnssl_z_score(volume):
    """nnssl's steps, literally: crop to the nonzero mask's box, z-score it."""
    mask = binary_fill_holes(volume != 0)
    coords = np.argwhere(mask)
    lo, hi = coords.min(0), coords.max(0) + 1
    region = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return region.mean(dtype=np.float64), max(region.std(dtype=np.float64), 1e-8)


def test_registry_lists_both_policies_with_clip_as_default():
    assert list_normalization_policies() == ["clip_and_z_score", "z_score_only"]
    assert DEFAULT_NORMALIZATION_POLICY == "clip_and_z_score"
    assert isinstance(get_normalization_policy("Z_SCORE_ONLY"), NormalizationPolicy)


def test_registry_rejects_unknown_and_duplicate_names():
    with pytest.raises(ValueError, match="not registered"):
        get_normalization_policy("percentile")
    with pytest.raises(ValueError, match="already registered"):
        register_normalization_policy("z_score_only")(NormalizationPolicy)


def test_policy_name_from_config_defaults_for_older_configs():
    assert get_normalization_policy_name({"data": {"normalization_policy": "z_score_only"}}) == (
        "z_score_only"
    )
    assert get_normalization_policy_name({"data": {"index": "i.parquet"}}) == "clip_and_z_score"
    assert get_normalization_policy_name({}) == "clip_and_z_score"


def test_clip_and_z_score_matches_the_index_statistics():
    volume = np.linspace(-5, 5, 1000, dtype=np.float32).reshape(10, 10, 10)
    out, scale = get_normalization_policy("clip_and_z_score").normalize(volume, ROW)
    expected = (np.clip(volume, -1.0, 3.0) - 0.5) / 2.0
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    assert out.dtype == np.float32
    assert scale == IntensityScale(shift=0.5, scale=2.0)


def test_clip_and_z_score_floors_a_zero_std():
    _, scale = get_normalization_policy("clip_and_z_score").normalize(
        np.zeros((2, 2, 2), np.float32), {**ROW, "fg_std": 0.0}
    )
    assert scale.scale == NormalizationPolicy.EPS


def test_z_score_only_matches_nnssl():
    volume = _skull_stripped_volume()
    out, scale = get_normalization_policy("z_score_only").normalize(volume, ROW)
    mean, std = _nnssl_z_score(volume)
    assert scale.shift == pytest.approx(mean) and scale.scale == pytest.approx(std)
    np.testing.assert_allclose(out, (volume - mean) / std, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out * scale.scale + scale.shift, volume, atol=1e-3)


def test_z_score_only_uses_the_whole_volume_when_nothing_is_zero():
    volume = np.random.default_rng(1).normal(-500, 400, (8, 8, 8)).astype(np.float32)  # CT-like
    out, scale = get_normalization_policy("z_score_only").normalize(volume, {})
    assert scale.shift == pytest.approx(volume.mean(dtype=np.float64))
    assert abs(out.mean()) < 1e-5 and abs(out.std() - 1.0) < 1e-4


def test_z_score_only_keeps_an_all_zero_volume_finite():
    out, scale = get_normalization_policy("z_score_only").normalize(
        np.zeros((4, 4, 4), np.float32), {}
    )
    assert scale == IntensityScale(shift=0.0, scale=NormalizationPolicy.EPS)
    assert np.all(out == 0)
