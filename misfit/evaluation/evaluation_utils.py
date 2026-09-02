"""Utilities for MISFIT reconstruction evaluation.

Mirrors MIST's evaluation_utils pattern: DataFrame initialisation and
per-column summary statistics (mean, std, quartiles).
"""
import warnings
from functools import partial

import numpy as np
import pandas as pd


def initialize_results_dataframe(columns: list[str]) -> pd.DataFrame:
    """Return an empty DataFrame with a ``volume_id`` column plus *columns*.

    Args:
        columns: Ordered list of the numeric result columns — the caller passes
            the already-expanded list (e.g. ``["masked_mae", "masked_mae_naive",
            "masked_mae_skill", ...]``).

    Returns:
        Empty DataFrame with columns ``["volume_id", *columns]``.
    """
    return pd.DataFrame(columns=["volume_id"] + list(columns))


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
        for label, func in zip(stats_labels, stats_functions, strict=True)
    ]

    return pd.concat(
        [results_df, pd.DataFrame(stats_rows)],
        ignore_index=True,
    )
