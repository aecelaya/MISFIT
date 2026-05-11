"""Tests for misfit.preprocessing.indexer."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from misfit.preprocessing.indexer import INDEX_COLUMNS, assign_splits, build_index


def _make_nifti(path: Path):
    data = np.random.randn(8, 8, 8).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, str(path))


def test_build_index_success(tmp_path):
    p = tmp_path / "vol.nii.gz"
    _make_nifti(p)

    df, errors = build_index([p], output_path=None, num_workers=1)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert errors == []
    for col in INDEX_COLUMNS:
        assert col in df.columns


def test_build_index_writes_parquet(tmp_path):
    p = tmp_path / "vol.nii.gz"
    _make_nifti(p)
    out = tmp_path / "index.parquet"

    build_index([p], output_path=out, num_workers=1)

    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 1


def test_build_index_collects_errors(tmp_path):
    bad = tmp_path / "missing.nii.gz"
    df, errors = build_index([bad], output_path=None, num_workers=1)

    assert len(errors) == 1
    assert len(df) == 0
    assert list(df.columns) == INDEX_COLUMNS


def test_build_index_mixed_success_and_error(tmp_path):
    good = tmp_path / "good.nii.gz"
    bad = tmp_path / "bad.nii.gz"
    _make_nifti(good)

    df, errors = build_index([good, bad], output_path=None, num_workers=1)

    assert len(df) == 1
    assert len(errors) == 1


def test_build_index_no_paths(tmp_path):
    df, errors = build_index([], output_path=None, num_workers=1)
    assert len(df) == 0
    assert errors == []


def test_build_index_creates_parent_dirs(tmp_path):
    p = tmp_path / "vol.nii.gz"
    _make_nifti(p)
    out = tmp_path / "nested" / "dir" / "index.parquet"

    build_index([p], output_path=out, num_workers=1)
    assert out.exists()


def test_build_index_split_column_present(tmp_path):
    """Every row in the index must have a 'split' column."""
    p = tmp_path / "vol.nii.gz"
    _make_nifti(p)

    df, _ = build_index([p], output_path=None, num_workers=1)

    assert "split" in df.columns
    assert df["split"].iloc[0] in ("train", "val", "test")


def test_build_index_split_ratios_respected(tmp_path):
    """Custom split_ratios produce the expected row counts."""
    paths = []
    for i in range(10):
        p = tmp_path / f"vol{i}.nii.gz"
        _make_nifti(p)
        paths.append(p)

    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    df, _ = build_index(paths, output_path=None, num_workers=1,
                        split_ratios=ratios, split_seed=0)

    counts = df["split"].value_counts()
    assert counts.get("train", 0) == 6
    assert counts.get("val", 0) == 2
    assert counts.get("test", 0) == 2


# ---------------------------------------------------------------------------
# assign_splits
# ---------------------------------------------------------------------------

def test_assign_splits_empty_df():
    df = pd.DataFrame(columns=["volume_id"])
    result = assign_splits(df, {"train": 0.8, "val": 0.1, "test": 0.1})
    assert "split" in result.columns
    assert len(result) == 0


def test_assign_splits_all_rows_assigned(tmp_path):
    df = pd.DataFrame({"volume_id": [f"v{i}" for i in range(20)]})
    result = assign_splits(df, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=42)
    assert result["split"].notna().all()
    assert set(result["split"].unique()).issubset({"train", "val", "test"})


def test_assign_splits_reproducible():
    df = pd.DataFrame({"volume_id": [f"v{i}" for i in range(20)]})
    r1 = assign_splits(df, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=7)
    r2 = assign_splits(df, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=7)
    assert list(r1["split"]) == list(r2["split"])


def test_assign_splits_different_seeds_differ():
    df = pd.DataFrame({"volume_id": [f"v{i}" for i in range(50)]})
    r1 = assign_splits(df, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=1)
    r2 = assign_splits(df, {"train": 0.8, "val": 0.1, "test": 0.1}, seed=2)
    assert list(r1["split"]) != list(r2["split"])
