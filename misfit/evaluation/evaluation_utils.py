"""Utilities for MISFIT reconstruction evaluation.

Mirrors MIST's evaluation_utils pattern: DataFrame initialisation and
per-column summary statistics (mean, std, quartiles).
"""
import warnings
from functools import partial
from typing import Dict, List

import numpy as np
import pandas as pd


def initialize_results_dataframe(metric_names: List[str]) -> pd.DataFrame:
    """Return an empty DataFrame with the correct evaluation columns.

    Args:
        metric_names: Ordered list of metric names (e.g. ``["masked_mae",
            "ssim"]``).  One column is created per metric, preceded by a
            ``volume_id`` column.

    Returns:
        Empty DataFrame with columns ``["volume_id", *metric_names]``.
    """
    return pd.DataFrame(columns=["volume_id"] + list(metric_names))


def compute_results_stats(results_df: pd.DataFrame) -> pd.DataFrame:
    """Append summary statistics rows to the results DataFrame.

    Appends Mean, Std, 25th Percentile, Median, and 75th Percentile rows
    (ignoring NaN values) so the final CSV is self-contained.

    Args:
        results_df: Per-volume results with a ``volume_id`` string column
            followed by numeric metric columns.

    Returns:
        Updated DataFrame with five summary rows appended at the bottom.
    """
    stats_labels = [
        "Mean", "Std", "25th Percentile", "Median", "75th Percentile"
    ]
    stats_functions = [
        np.nanmean,
        np.nanstd,
        partial(np.nanpercentile, q=25),
        partial(np.nanpercentile, q=50),
        partial(np.nanpercentile, q=75),
    ]

    metric_cols = results_df.columns[1:]  # skip volume_id

    def _safe(func, col):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if results_df[col].isna().all():
                return np.nan
            return func(results_df[col])

    stats_rows = [
        {
            "volume_id": label,
            **{col: _safe(func, col) for col in metric_cols},
        }
        for label, func in zip(stats_labels, stats_functions)
    ]

    return pd.concat(
        [results_df, pd.DataFrame(stats_rows)],
        ignore_index=True,
    )
