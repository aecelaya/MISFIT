"""CLI entrypoint for misfit_index — builds the MISFIT metadata index.

Usage::

    misfit_index --data-dir /data/niftis --output index.parquet
    misfit_index --manifest paths.csv --output index.parquet --num-workers 64
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import Optional, List

import pandas as pd

from misfit.cli.args import ArgParser, add_index_args
from misfit.preprocessing.index_utils import collect_nifti_paths
from misfit.preprocessing.indexer import build_index
from misfit.utils.console import console, print_error, print_info


def _parse_args(args=None):
    parser = ArgParser(
        prog="misfit_index",
        description=(
            "Build the MISFIT metadata index from a collection of NIfTI files.\n\n"
            "Computes per-volume statistics (spacing, foreground bbox, intensity\n"
            "percentiles, normalization constants) in parallel and writes the\n"
            "results to a Parquet file. Run this once before training."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_index_args(parser, source_required=True)
    return parser.parse_args(args)


def index_entry(args=None) -> None:
    """Entrypoint for the misfit_index CLI command."""
    ns = _parse_args(args)

    # --- Collect paths ---
    if ns.data_dir is not None:
        data_dir = Path(ns.data_dir)
        if not data_dir.is_dir():
            print_error(f"--data-dir '{data_dir}' is not a directory.")
            sys.exit(1)
        nifti_paths = collect_nifti_paths(data_dir)
        if not nifti_paths:
            print_error(f"No .nii or .nii.gz files found under '{data_dir}'.")
            sys.exit(1)
        print_info(
            f"Found [bold]{len(nifti_paths):,}[/bold] NIfTI files "
            f"under '{data_dir}'."
        )
    else:
        manifest_path = Path(ns.manifest)
        if not manifest_path.exists():
            print_error(f"--manifest '{manifest_path}' does not exist.")
            sys.exit(1)
        manifest_df = pd.read_csv(manifest_path)
        if "path" not in manifest_df.columns:
            print_error("--manifest CSV must contain a 'path' column.")
            sys.exit(1)
        nifti_paths = [Path(p) for p in manifest_df["path"].tolist()]
        print_info(
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
