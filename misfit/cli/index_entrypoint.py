"""CLI entrypoint for misfit_index — builds the MISFIT metadata index."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from misfit.preprocessing.index_utils import collect_nifti_paths
from misfit.preprocessing.indexer import build_index, console


def _parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="misfit_index",
        description=(
            "Build the MISFIT metadata index from a collection of NIfTI files.\n\n"
            "Computes per-volume statistics (spacing, foreground bbox, intensity\n"
            "percentiles, normalization constants) in parallel and writes the\n"
            "results to a Parquet file. Run this once before training."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Input source (mutually exclusive) ---
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-dir",
        type=str,
        metavar="DIR",
        help=(
            "Root directory to recursively search for .nii and .nii.gz files. "
            "All NIfTI files found under this directory are indexed."
        ),
    )
    source.add_argument(
        "--manifest",
        type=str,
        metavar="CSV",
        help=(
            "CSV file with a 'path' column listing absolute paths to NIfTI "
            "files. Use this when your dataset spans multiple directories or "
            "when you want explicit control over which files are indexed."
        ),
    )

    # --- Output ---
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="PARQUET",
        help="Destination path for the output Parquet index file.",
    )

    # --- Performance ---
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        metavar="N",
        help=(
            "Number of parallel worker processes. On HPC nodes with fast "
            "parallel filesystems (Lustre/GPFS), 32–64 is a reasonable "
            "starting point. Defaults to 32."
        ),
    )

    return parser.parse_args(args)


def index_entry(args=None) -> None:
    """Entrypoint for the misfit_index CLI command."""
    ns = _parse_args(args)

    # --- Collect paths ---
    if ns.data_dir is not None:
        data_dir = Path(ns.data_dir)
        if not data_dir.is_dir():
            console.print(
                f"[bold red]Error:[/bold red] --data-dir '{data_dir}' "
                f"is not a directory."
            )
            sys.exit(1)
        nifti_paths = collect_nifti_paths(data_dir)
        if not nifti_paths:
            console.print(
                f"[bold red]Error:[/bold red] No .nii or .nii.gz files "
                f"found under '{data_dir}'."
            )
            sys.exit(1)
        console.print(
            f"Found [bold]{len(nifti_paths):,}[/bold] NIfTI files "
            f"under '{data_dir}'."
        )
    else:
        manifest_path = Path(ns.manifest)
        if not manifest_path.exists():
            console.print(
                f"[bold red]Error:[/bold red] --manifest '{manifest_path}' "
                f"does not exist."
            )
            sys.exit(1)
        manifest_df = pd.read_csv(manifest_path)
        if "path" not in manifest_df.columns:
            console.print(
                "[bold red]Error:[/bold red] --manifest CSV must contain "
                "a 'path' column."
            )
            sys.exit(1)
        nifti_paths = [Path(p) for p in manifest_df["path"].tolist()]
        console.print(
            f"Loaded [bold]{len(nifti_paths):,}[/bold] paths from "
            f"'{manifest_path}'."
        )

    # --- Build index ---
    _, errors = build_index(
        nifti_paths=nifti_paths,
        output_path=Path(ns.output),
        num_workers=ns.num_workers,
    )

    sys.exit(1 if errors else 0)
