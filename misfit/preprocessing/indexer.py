"""Parallel metadata index builder for MISFIT.

Walks a collection of NIfTI files and computes per-volume statistics
(spacing, foreground bbox, intensity percentiles, normalization constants)
in parallel, then writes the results to a Parquet file.

The resulting index is read by the MISFIT data loader at training time to
apply clip + z-score normalization on the fly without recomputing stats
per epoch.

Typical usage::

    from misfit.preprocessing.indexer import build_index
    from misfit.preprocessing.index_utils import collect_nifti_paths

    paths = collect_nifti_paths("/data/mdanderson/niftis")
    index_df, errors = build_index(paths, "index.parquet", num_workers=32)
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from misfit.preprocessing.index_utils import compute_volume_stats
from misfit.utils.console import console
from misfit.utils.progress_bar import get_progress_bar

# Columns produced by the parallel stat workers (no split yet).
_STATS_COLUMNS = [
    "volume_id",
    "path",
    "shape_d", "shape_h", "shape_w",
    "spacing_d", "spacing_h", "spacing_w",
    "affine",
    "fg_x_start", "fg_x_end",
    "fg_y_start", "fg_y_end",
    "fg_z_start", "fg_z_end",
    "p1", "p99",
    "fg_mean", "fg_std",
]

# Full parquet schema including the split assignment. ``group`` holds the
# optional grouping key from the input file (e.g. a subject ID), or null.
INDEX_COLUMNS = ["volume_id", "path", "split", "group"] + _STATS_COLUMNS[2:]

# Default split config: val and test each get 5% of the volumes, capped at
# 500, and train gets the rest. A fraction in [0, 1) is a share of the corpus;
# an integer is an exact volume count (not capped).
DEFAULT_SPLIT_CONFIG: dict[str, float | int | None] = {
    "val":      0.05,
    "test":     0.05,
    "max_val":  500,
    "max_test": 500,
}


def _held_out_size(
    config: dict, split: str, n: int,
) -> int:
    """Return the number of volumes for the ``val`` or ``test`` split.

    Args:
        config: Split config (see ``DEFAULT_SPLIT_CONFIG``).
        split: ``"val"`` or ``"test"``.
        n: Total number of volumes.

    Returns:
        Volume count for *split*. A nonzero fraction always yields at least
        one volume; ``max_<split>`` caps fractions only.

    Raises:
        ValueError: If the value is neither a fraction in [0, 1) nor a
            non-negative integer, or the cap is not a non-negative integer.
    """
    value = config.get(split, DEFAULT_SPLIT_CONFIG[split])
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"Split '{split}' count must be >= 0, got {value}.")
        return value
    if not isinstance(value, float) or not 0.0 <= value < 1.0:
        raise ValueError(
            f"Split '{split}' must be a fraction in [0, 1) or a whole number "
            f"of volumes, got {value!r}."
        )
    if value == 0.0:
        return 0
    size = max(1, int(round(n * value)))
    cap = config.get(f"max_{split}", DEFAULT_SPLIT_CONFIG[f"max_{split}"])
    if cap is None:
        return size
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
        raise ValueError(
            f"'max_{split}' must be a non-negative integer or null, got {cap!r}."
        )
    return min(size, cap)


def split_targets(config: dict, n: int) -> list[tuple[str, int | None]]:
    """Return ``(split, target_size)`` pairs in the order splits are filled.

    The last pair's target is None: that split takes every remaining volume.

    A legacy config (one with a ``"train"`` key, from MISFIT before split
    caps) keeps its old meaning: train and val are fractions of the corpus,
    filled in that order, and test takes the remainder. Otherwise val and test are sized by :func:`_held_out_size` and
    train takes the rest. Val and test shrink (test first) if needed so that
    train keeps at least one volume.

    Args:
        config: Split config (see ``DEFAULT_SPLIT_CONFIG``).
        n: Total number of volumes.

    Returns:
        List of ``(split, target_size)`` pairs.
    """
    if "train" in config:
        return [
            ("train", int(round(n * config["train"]))),
            ("val",   int(round(n * config.get("val", 0.1)))),
            ("test",  None),
        ]

    n_val = _held_out_size(config, "val", n)
    n_test = _held_out_size(config, "test", n)
    excess = n_val + n_test - (n - 1)
    if excess > 0:
        cut = min(excess, n_test)
        n_test -= cut
        n_val = max(0, n_val - (excess - cut))
    return [("val", n_val), ("test", n_test), ("train", None)]


def _group_codes(df: pd.DataFrame) -> np.ndarray:
    """Return an integer group code per row, numbered by first appearance.

    Rows sharing a non-null ``group`` value share a code; rows with a null
    (or no) ``group`` are their own singleton group. With no grouping the
    codes are ``0 .. n-1`` in row order.
    """
    if "group" not in df.columns:
        return np.arange(len(df))
    seen: dict[tuple, int] = {}
    codes = [
        seen.setdefault(("row", i) if pd.isna(g) else ("group", g), len(seen))
        for i, g in enumerate(df["group"])
    ]
    return np.asarray(codes)


def assign_splits(
    df: pd.DataFrame,
    config: dict,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign a ``split`` column (``"train"`` / ``"val"`` / ``"test"``) to *df*.

    Groups (see :func:`_group_codes`) are shuffled with *seed* and dealt out
    whole, in the fill order from :func:`split_targets`: each split takes
    groups until it holds at least its target number of volumes, so every
    volume of a group lands in the same split. With multi-volume groups a
    split can overshoot its target by less than one group.

    Args:
        df: DataFrame of volume stats (no ``split`` column yet), optionally
            with a ``group`` column.
        config: Split config (see ``DEFAULT_SPLIT_CONFIG``).
        seed: Random seed for the shuffle. Defaults to 42.

    Returns:
        Copy of *df* with a new ``"split"`` column.
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        df["split"] = pd.Series(dtype=object)
        return df

    codes = _group_codes(df)
    sizes = np.bincount(codes)
    targets = split_targets(config, n)

    rng = np.random.default_rng(seed)
    group_split = np.empty(len(sizes), dtype=object)
    k, filled = 0, 0
    for g in rng.permutation(len(sizes)):
        while targets[k][1] is not None and filled >= targets[k][1]:
            k, filled = k + 1, 0
        group_split[g] = targets[k][0]
        filled += sizes[g]

    df["split"] = group_split[codes]
    return df


def build_index(
    nifti_paths: list[Path],
    output_path: Path | None = None,
    num_workers: int = 32,
    split_config: dict | None = None,
    split_seed: int = 42,
    groups: list[str | None] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Build the MISFIT metadata index from a list of NIfTI files.

    Processes each file in a separate worker process to parallelize I/O and
    stat computation. Failed files are collected as error messages and skipped
    rather than aborting the entire run — at 1M+ scale, some files will
    always be corrupt or malformed.

    Args:
        nifti_paths: List of paths to .nii or .nii.gz files.
        output_path: Destination for the Parquet index file. If None, the
            index is returned but not written to disk. Defaults to None.
        num_workers: Number of parallel worker processes. For HPC nodes with
            fast parallel filesystems (Lustre/GPFS), 32 - 64 workers is a
            reasonable starting point. Defaults to 32.
        split_config: Split sizes (see ``DEFAULT_SPLIT_CONFIG``). Defaults
            to 5% val and 5% test, each capped at 500 volumes.
        split_seed: Random seed for the split shuffle. Defaults to 42.
        groups: Optional grouping key per path (aligned with
            *nifti_paths*, None for ungrouped). Volumes sharing a key are
            kept in the same split. Defaults to None (no grouping).

    Returns:
        Tuple of:
            index_df: DataFrame with one row per successfully processed
                volume, columns matching INDEX_COLUMNS (including ``split``).
            errors: List of error message strings for files that failed.
                Empty if all files succeeded.
    """
    records = []
    errors = []
    total = len(nifti_paths)

    console.print(
        f"[bold]Indexing {total:,} NIfTI files "
        f"with {num_workers} workers...[/bold]"
    )

    with get_progress_bar() as progress:
        task = progress.add_task("Building index", total=total)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(compute_volume_stats, p): i
                for i, p in enumerate(nifti_paths)
            }
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    errors.append(
                        f"  [red]FAILED[/red] {result['path']}: {result['error']}"
                    )
                else:
                    i = futures[future]
                    group = None if groups is None else groups[i]
                    records.append((i, {**result, "group": group}))
                progress.advance(task)

    # Restore input order: as_completed yields in completion order, which
    # varies run to run and would make the seeded split irreproducible.
    records = [record for _, record in sorted(records, key=lambda r: r[0])]

    if split_config is None:
        split_config = DEFAULT_SPLIT_CONFIG

    # Build DataFrame with a consistent column schema.
    if records:
        index_df = pd.DataFrame(records)[_STATS_COLUMNS + ["group"]]
        index_df = assign_splits(index_df, split_config, split_seed)
        index_df = index_df[INDEX_COLUMNS]
    else:
        index_df = pd.DataFrame(columns=INDEX_COLUMNS)

    # Summary.
    n_ok = len(records)
    n_err = len(errors)
    console.print(
        f"\n[green]Indexed {n_ok:,} volumes successfully.[/green]"
        + (f"  [yellow]{n_err:,} failed.[/yellow]" if n_err else "")
    )

    if errors:
        console.print("\n[bold yellow]Failed files:[/bold yellow]")
        for msg in errors:
            console.print(msg)

    # Write to Parquet.
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        index_df.to_parquet(output_path, index=False)
        console.print(
            f"\n[bold green]Index saved to {output_path}[/bold green]")

    return index_df, errors
