"""Tests for misfit.preprocessing.indexer."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from misfit.preprocessing.indexer import (
    DEFAULT_SPLIT_CONFIG,
    INDEX_COLUMNS,
    assign_splits,
    build_index,
    split_targets,
)


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


def test_build_index_legacy_split_ratios_respected(tmp_path):
    """A legacy train/val/test ratio config produces the expected row counts."""
    paths = []
    for i in range(10):
        p = tmp_path / f"vol{i}.nii.gz"
        _make_nifti(p)
        paths.append(p)

    ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    df, _ = build_index(paths, output_path=None, num_workers=1,
                        split_config=ratios, split_seed=0)

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


# ---------------------------------------------------------------------------
# Split sizes, caps, and groups
# ---------------------------------------------------------------------------

def _ids(n):
    return pd.DataFrame({"volume_id": [f"v{i}" for i in range(n)]})


def _counts(df):
    return df["split"].value_counts().to_dict()


def test_assign_splits_default_is_five_percent():
    result = assign_splits(_ids(1000), DEFAULT_SPLIT_CONFIG, seed=0)
    assert _counts(result) == {"train": 900, "val": 50, "test": 50}


def test_assign_splits_default_caps_at_500():
    result = assign_splits(_ids(20_000), DEFAULT_SPLIT_CONFIG, seed=0)
    assert _counts(result) == {"train": 19_000, "val": 500, "test": 500}


def test_assign_splits_null_cap_is_uncapped():
    config = {"val": 0.05, "test": 0.05, "max_val": None, "max_test": None}
    result = assign_splits(_ids(20_000), config, seed=0)
    assert _counts(result) == {"train": 18_000, "val": 1000, "test": 1000}


def test_assign_splits_integer_counts_are_exact_and_uncapped():
    config = {"val": 600, "test": 0, "max_val": 500}
    result = assign_splits(_ids(10_000), config, seed=0)
    assert _counts(result) == {"train": 9400, "val": 600}


def test_assign_splits_missing_keys_use_defaults():
    result = assign_splits(_ids(1000), {}, seed=0)
    assert _counts(result) == {"train": 900, "val": 50, "test": 50}


def test_assign_splits_nonzero_fraction_gets_at_least_one():
    result = assign_splits(_ids(10), DEFAULT_SPLIT_CONFIG, seed=0)
    assert _counts(result) == {"train": 8, "val": 1, "test": 1}


def test_assign_splits_zero_fraction_is_empty():
    config = {"val": 0.05, "test": 0.0}
    result = assign_splits(_ids(1000), config, seed=0)
    assert "test" not in _counts(result)


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, {"train": 1}),
        (2, {"train": 1, "val": 1}),
        (3, {"train": 1, "val": 1, "test": 1}),
    ],
)
def test_assign_splits_tiny_corpus_keeps_a_train_volume(n, expected):
    assert _counts(assign_splits(_ids(n), DEFAULT_SPLIT_CONFIG)) == expected


def test_split_targets_shrinks_val_after_test():
    """When val alone exceeds the corpus, test empties first, then val."""
    targets = split_targets({"val": 10, "test": 10}, 5)
    assert targets == [("val", 4), ("test", 0), ("train", None)]


def test_split_targets_legacy_config():
    targets = split_targets({"train": 0.8, "val": 0.1, "test": 0.1}, 100)
    assert targets == [("train", 80), ("val", 10), ("test", None)]


@pytest.mark.parametrize(
    "config",
    [
        {"val": 1.0},
        {"val": 1.5},
        {"val": -0.1},
        {"val": -1},
        {"val": "5%"},
        {"val": True},
        {"max_val": -1},
        {"max_val": 1.5},
        {"max_val": True},
    ],
)
def test_split_targets_rejects_bad_values(config):
    with pytest.raises(ValueError):
        split_targets(config, 100)


def test_assign_splits_keeps_groups_together():
    n_subjects = 200
    df = pd.DataFrame({
        "volume_id": [f"s{s}_m{m}" for s in range(n_subjects) for m in range(4)],
        "group": [f"s{s}" for s in range(n_subjects) for _ in range(4)],
    })
    result = assign_splits(df, DEFAULT_SPLIT_CONFIG, seed=0)
    assert (result.groupby("group")["split"].nunique() == 1).all()
    # 5% of 800 volumes is 40, dealt out in whole subjects of 4.
    assert _counts(result) == {"train": 720, "val": 40, "test": 40}


def test_assign_splits_null_groups_are_singletons():
    """Rows with no group are split individually, not pooled together."""
    df = pd.DataFrame({
        "volume_id": [f"v{i}" for i in range(100)],
        "group": [None] * 100,
    })
    grouped = assign_splits(df, DEFAULT_SPLIT_CONFIG, seed=3)
    ungrouped = assign_splits(df.drop(columns="group"), DEFAULT_SPLIT_CONFIG, seed=3)
    assert list(grouped["split"]) == list(ungrouped["split"])


def test_build_index_records_groups_and_keeps_input_order(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"vol{i}.nii.gz"
        _make_nifti(p)
        paths.append(p)
    groups = ["a", "a", "b", "b", None, "c"]

    df, _ = build_index(
        paths, output_path=None, num_workers=2, groups=groups, split_seed=0,
    )

    assert list(df["volume_id"]) == [f"vol{i}" for i in range(6)]
    assert list(df["group"]) == groups
    assert df[df["group"] == "a"]["split"].nunique() == 1
    assert df[df["group"] == "b"]["split"].nunique() == 1


def test_build_index_without_groups_has_null_group_column(tmp_path):
    p = tmp_path / "vol.nii.gz"
    _make_nifti(p)
    out = tmp_path / "index.parquet"

    build_index([p], output_path=out, num_workers=1)

    assert pd.read_parquet(out)["group"].isna().all()
