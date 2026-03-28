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
from typing import Dict, List, Optional, Tuple

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

# Full parquet schema including the split assignment.
INDEX_COLUMNS = ["volume_id", "path", "split"] + _STATS_COLUMNS[2:]

# Default 80 / 10 / 10 split ratios.
DEFAULT_SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.8,
    "val":   0.1,
    "test":  0.1,
}


def assign_splits(
    df: pd.DataFrame,
    ratios: Dict[str, float],
    seed: int = 42,
) -> pd.DataFrame:
    """Assign a ``split`` column (``"train"`` / ``"val"`` / ``"test"``) to *df*.

    Volumes are shuffled with *seed* and then divided according to *ratios*.
    The test set receives any remainder after rounding.

    Args:
        df: DataFrame of volume stats (no ``split`` column yet).
        ratios: Mapping with keys ``"train"``, ``"val"``, and ``"test"``
            whose values are proportions summing to 1.
        seed: Random seed for the shuffle. Defaults to 42.

    Returns:
        Copy of *df* with a new ``"split"`` column.
    """
    df = df.copy()
    n = len(df)
    if n == 0:
        df["split"] = pd.Series(dtype=object)
        return df

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(round(n * ratios.get("train", 0.8)))
    n_val   = int(round(n * ratios.get("val",   0.1)))
    # test gets the remainder so the total always equals n.

    labels = np.empty(n, dtype=object)
    labels[indices[:n_train]]                   = "train"
    labels[indices[n_train:n_train + n_val]]    = "val"
    labels[indices[n_train + n_val:]]           = "test"

    df["split"] = labels
    return df


def build_index(
    nifti_paths: List[Path],
    output_path: Optional[Path] = None,
    num_workers: int = 32,
    split_ratios: Optional[Dict[str, float]] = None,
    split_seed: int = 42,
) -> Tuple[pd.DataFrame, List[str]]:
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
            fast parallel filesystems (Lustre/GPFS), 32–64 workers is a
            reasonable starting point. Defaults to 32.
        split_ratios: Train/val/test proportions. Defaults to
            ``DEFAULT_SPLIT_RATIOS`` (80 / 10 / 10).
        split_seed: Random seed for the split shuffle. Defaults to 42.

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
                executor.submit(compute_volume_stats, p): p
                for p in nifti_paths
            }
            for future in as_completed(futures):
                result = future.result()
                if "error" in result:
                    errors.append(
                        f"  [red]FAILED[/red] {result['path']}: {result['error']}"
                    )
                else:
                    records.append(result)
                progress.advance(task)

    if split_ratios is None:
        split_ratios = DEFAULT_SPLIT_RATIOS

    # Build DataFrame with a consistent column schema.
    if records:
        index_df = pd.DataFrame(records)[_STATS_COLUMNS]
        index_df = assign_splits(index_df, split_ratios, split_seed)
        index_df = index_df[INDEX_COLUMNS]
    else:
        index_df = pd.DataFrame(columns=INDEX_COLUMNS)

    # Summary.
    n_ok  = len(records)
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
        console.print(f"\n[bold green]Index saved to {output_path}[/bold green]")

    return index_df, errors
