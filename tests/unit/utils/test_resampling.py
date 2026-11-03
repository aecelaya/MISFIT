"""Tests for misfit.utils.resampling (optional --target-spacing resampling)."""
import numpy as np
import pandas as pd
import pytest

from misfit.utils.resampling import (
    get_row_spacing,
    get_target_spacing,
    resample_to_shape,
    resample_to_spacing,
    resampled_shape,
)


def test_get_target_spacing_reads_spatial_config():
    config = {"spatial_config": {"patch_size": [96] * 3, "target_spacing": [1, 1, 2.5]}}
    assert get_target_spacing(config) == (1.0, 1.0, 2.5)


@pytest.mark.parametrize("config", [
    {"spatial_config": {"patch_size": [96] * 3, "target_spacing": None}},
    {"model": {"patch_size": [96] * 3}},  # legacy config: no spatial_config
])
def test_get_target_spacing_none_means_native(config):
    assert get_target_spacing(config) is None


def test_get_row_spacing_uses_index_columns_in_xyz_order():
    row = pd.Series({"spacing_d": 0.8, "spacing_h": 0.9, "spacing_w": 5.0})
    assert get_row_spacing(row) == (0.8, 0.9, 5.0)


def test_resampled_shape_preserves_physical_extent():
    # 100 × 0.5 mm = 50 mm → 50 voxels at 1 mm; 20 × 5 mm → 100 at 1 mm.
    assert resampled_shape((100, 64, 20), (0.5, 1.0, 5.0), (1, 1, 1)) == (50, 64, 100)


def test_resampled_shape_floors_at_one_voxel():
    assert resampled_shape((2, 2, 2), (1, 1, 1), (10, 10, 10)) == (1, 1, 1)


def test_resample_to_shape_output_shape_and_dtype():
    out = resample_to_shape(np.random.rand(10, 12, 8), (20, 6, 8))
    assert out.shape == (20, 6, 8)
    assert out.dtype == np.float32


def test_resample_to_shape_same_shape_is_unchanged():
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    assert np.array_equal(resample_to_shape(vol, (8, 8, 8)), vol)


def test_resample_to_shape_preserves_constant_and_mean():
    const = np.full((10, 10, 4), 3.0, dtype=np.float32)
    assert np.allclose(resample_to_shape(const, (20, 20, 16)), 3.0)

    smooth = np.linspace(0, 1, 16 * 16 * 16, dtype=np.float32).reshape(16, 16, 16)
    assert resample_to_shape(smooth, (32, 32, 32)).mean() == pytest.approx(
        smooth.mean(), abs=1e-3
    )


def test_resample_to_spacing_none_is_native():
    vol = np.random.rand(5, 6, 7)
    out = resample_to_spacing(vol, (0.5, 0.5, 3.0), None)
    assert out.shape == vol.shape
    assert out.dtype == np.float32


def test_resample_to_spacing_resamples_to_target():
    vol = np.random.rand(64, 64, 10).astype(np.float32)
    out = resample_to_spacing(vol, (0.5, 0.5, 4.0), (1.0, 1.0, 1.0))
    assert out.shape == (32, 32, 40)


def test_resample_keeps_xyz_axis_alignment():
    """A structure at a known physical location stays there after resampling."""
    vol = np.zeros((40, 40, 8), dtype=np.float32)
    vol[30:40, :, :] = 1.0  # last 25% of the x axis
    out = resample_to_spacing(vol, (0.5, 0.5, 2.5), (1.0, 1.0, 1.0))
    assert out.shape == (20, 20, 20)
    assert out[:14].max() < 0.05 and out[16:].min() > 0.95
