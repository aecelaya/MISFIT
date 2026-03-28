"""Tests for misfit.evaluation.evaluation_utils."""
import numpy as np
import pandas as pd
import pytest

from misfit.evaluation.evaluation_utils import (
    compute_results_stats,
    initialize_results_dataframe,
)


def test_initialize_results_dataframe_columns():
    df = initialize_results_dataframe(["metric_a", "metric_b"])
    assert list(df.columns) == ["volume_id", "metric_a", "metric_b"]
    assert len(df) == 0


def test_initialize_results_dataframe_empty():
    df = initialize_results_dataframe([])
    assert list(df.columns) == ["volume_id"]


def test_compute_results_stats_appends_rows():
    df = pd.DataFrame([
        {"volume_id": "vol1", "mae": 0.1, "psnr": 30.0},
        {"volume_id": "vol2", "mae": 0.2, "psnr": 28.0},
    ])
    result = compute_results_stats(df)
    # 2 original + 5 stats rows
    assert len(result) == 7
    labels = result["volume_id"].tolist()
    assert "Mean" in labels
    assert "Std" in labels
    assert "25th Percentile" in labels
    assert "Median" in labels
    assert "75th Percentile" in labels


def test_compute_results_stats_mean_correct():
    df = pd.DataFrame([
        {"volume_id": "vol1", "mae": 0.1},
        {"volume_id": "vol2", "mae": 0.3},
    ])
    result = compute_results_stats(df)
    mean_row = result[result["volume_id"] == "Mean"].iloc[0]
    assert mean_row["mae"] == pytest.approx(0.2, abs=1e-6)


def test_compute_results_stats_handles_nan():
    """All-NaN column should produce NaN stats without raising."""
    df = pd.DataFrame([
        {"volume_id": "vol1", "mae": np.nan},
        {"volume_id": "vol2", "mae": np.nan},
    ])
    result = compute_results_stats(df)
    mean_row = result[result["volume_id"] == "Mean"].iloc[0]
    assert np.isnan(mean_row["mae"])


def test_compute_results_stats_ignores_nan():
    df = pd.DataFrame([
        {"volume_id": "vol1", "mae": 0.1},
        {"volume_id": "vol2", "mae": np.nan},
        {"volume_id": "vol3", "mae": 0.3},
    ])
    result = compute_results_stats(df)
    mean_row = result[result["volume_id"] == "Mean"].iloc[0]
    assert mean_row["mae"] == pytest.approx(0.2, abs=1e-6)
