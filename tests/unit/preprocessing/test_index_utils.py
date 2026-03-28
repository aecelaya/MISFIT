"""Tests for misfit.preprocessing.index_utils."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from misfit.preprocessing.index_utils import (
    collect_nifti_paths,
    compute_volume_stats,
    get_volume_id,
)


# ---------------------------------------------------------------------------
# get_volume_id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("patient_001.nii.gz", "patient_001"),
    ("patient_001.nii", "patient_001"),
    ("patient_001.mha", "patient_001"),
    ("sub-01_T1w.nii.gz", "sub-01_T1w"),
])
def test_get_volume_id(filename, expected):
    assert get_volume_id(Path(filename)) == expected


# ---------------------------------------------------------------------------
# collect_nifti_paths
# ---------------------------------------------------------------------------

def test_collect_nifti_paths_finds_both_extensions(tmp_path):
    (tmp_path / "a.nii.gz").write_bytes(b"")
    (tmp_path / "b.nii").write_bytes(b"")
    (tmp_path / "c.txt").write_bytes(b"")
    paths = collect_nifti_paths(tmp_path)
    names = {p.name for p in paths}
    assert "a.nii.gz" in names
    assert "b.nii" in names
    assert "c.txt" not in names


def test_collect_nifti_paths_recursive(tmp_path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "deep.nii.gz").write_bytes(b"")
    paths = collect_nifti_paths(tmp_path)
    assert any(p.name == "deep.nii.gz" for p in paths)


def test_collect_nifti_paths_empty_dir(tmp_path):
    assert collect_nifti_paths(tmp_path) == []


def test_collect_nifti_paths_string_input(tmp_path):
    (tmp_path / "a.nii.gz").write_bytes(b"")
    paths = collect_nifti_paths(str(tmp_path))
    assert len(paths) == 1


# ---------------------------------------------------------------------------
# compute_volume_stats
# ---------------------------------------------------------------------------

def test_compute_volume_stats_success(tmp_path):
    data = np.random.randn(16, 16, 16).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / "vol.nii.gz"
    nib.save(img, str(p))

    result = compute_volume_stats(p)
    assert "error" not in result
    assert result["volume_id"] == "vol"
    assert result["shape_d"] == 16
    assert result["shape_h"] == 16
    assert result["shape_w"] == 16
    assert result["p1"] <= result["fg_mean"] <= result["p99"]
    assert result["fg_std"] >= 0


def test_compute_volume_stats_4d_takes_first_frame(tmp_path):
    data = np.random.randn(16, 16, 16, 3).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / "vol4d.nii.gz"
    nib.save(img, str(p))
    result = compute_volume_stats(p)
    assert "error" not in result
    assert result["shape_d"] == 16


def test_compute_volume_stats_missing_file():
    result = compute_volume_stats(Path("/nonexistent/file.nii.gz"))
    assert "error" in result
    assert "path" in result


def test_compute_volume_stats_constant_volume(tmp_path):
    """Constant volume: foreground detection should fall back to full volume."""
    data = np.ones((8, 8, 8), dtype=np.float32) * 5.0
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / "const.nii.gz"
    nib.save(img, str(p))
    result = compute_volume_stats(p)
    assert "error" not in result


def test_compute_volume_stats_2d_error(tmp_path):
    """2D volume shape not 3D."""
    data = np.ones((8, 8), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    p = tmp_path / "vol2d.nii.gz"
    nib.save(img, str(p))
    result = compute_volume_stats(p)
    assert "error" in result
