"""CLI entrypoint for misfit_index — builds the MISFIT metadata index.

Usage::

    misfit_index --input paths.csv    --output index.parquet
    misfit_index --input paths.parquet --output index.parquet --num-workers 64

The ``--input`` file must contain a ``path`` column with absolute paths to
NIfTI files, and may contain a ``group`` column (e.g. a subject ID): volumes
sharing a group are kept in the same split. Both CSV and Parquet formats are
accepted.

A ``<output_stem>_config.json`` file is written alongside the Parquet index
the first time the command is run. It records the val/test split sizes and
seed used to assign the ``split`` column. Edit this file and re-run
``misfit_index`` to obtain a different train/val/test split.
"""
import json
import sys
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

import pandas as pd

from misfit.cli.args import ArgParser, add_index_args
from misfit.preprocessing.indexer import (
    DEFAULT_SPLIT_CONFIG,
    build_index,
    split_targets,
)
from misfit.utils.console import print_error, print_info


def _split_config_path(output_path: Path) -> Path:
    """Return the companion split-config path for *output_path*.

    Example: ``/data/index.parquet`` → ``/data/index_config.json``.
    """
    return output_path.parent / (output_path.stem + "_config.json")


def _read_or_create_split_config(output_path: Path) -> dict:
    """Load the split config if it exists, otherwise create it with defaults.

    The config file lives at ``<output_stem>_config.json`` next to the parquet.
    Edit the file and re-run ``misfit_index`` to change split sizes.

    Returns:
        Dict with the split sizes (see ``DEFAULT_SPLIT_CONFIG``; older files
        may hold ``"train"``/``"val"``/``"test"`` ratios instead) and
        ``"seed"`` (int).
    """
    config_path = _split_config_path(output_path)
    if config_path.exists():
        with open(config_path) as fh:
            config = json.load(fh)
        print_info(f"Using split config from [bold]{config_path}[/bold]")
    else:
        config = {**DEFAULT_SPLIT_CONFIG, "seed": 42}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as fh:
            json.dump(config, fh, indent=2)
        print_info(
            f"Split config written to [bold]{config_path}[/bold]  "
            "(edit and re-run to change split sizes)"
        )
    return config


def _load_input_paths(
    input_path: Path,
) -> tuple[list[Path], list[str | None] | None]:
    """Load NIfTI paths, and optional groups, from a CSV or Parquet file.

    The file must contain a ``path`` column and may contain a ``group``
    column. Both ``.csv`` and ``.parquet`` extensions are supported; any
    other extension is read as Parquet.

    Args:
        input_path: Path to the input file.

    Returns:
        Tuple of the :class:`Path` objects from the ``path`` column and the
        ``group`` values as strings (None where empty), or None if there is
        no ``group`` column.

    Raises:
        SystemExit: If the file does not exist, cannot be read, or lacks a
            ``path`` column.
    """
    if not input_path.exists():
        print_error(f"--input '{input_path}' does not exist.")
        sys.exit(1)

    suffix = input_path.suffix.lower()
    try:
        if suffix == ".csv":
            # Read groups as text so subject IDs keep their exact spelling
            # (e.g. "0042" stays "0042", and a blank row doesn't turn 101 into 101.0).
            df = pd.read_csv(input_path, dtype={"group": str})
        else:
            # .parquet or unknown — treat as Parquet.
            df = pd.read_parquet(input_path)
    except Exception as exc:  # noqa: BLE001
        print_error(f"Could not read --input '{input_path}': {exc}")
        sys.exit(1)

    if "path" not in df.columns:
        print_error(f"--input '{input_path}' must contain a 'path' column.")
        sys.exit(1)

    paths = [Path(p) for p in df["path"].tolist()]
    if "group" not in df.columns:
        return paths, None
    # Stringify so a numeric group column (possible in Parquet) still works.
    groups = [None if pd.isna(g) else str(g) for g in df["group"].tolist()]
    return paths, groups


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
    add_index_args(parser, input_required=True)
    return parser.parse_args(args)


def index_entry(args=None) -> None:
    """Entrypoint for the misfit_index CLI command."""
    ns = _parse_args(args)

    # --- Collect paths ---
    input_path = Path(ns.input)
    nifti_paths, groups = _load_input_paths(input_path)
    if not nifti_paths:
        print_error(f"--input '{input_path}' contains no paths.")
        sys.exit(1)
    print_info(
        f"Loaded [bold]{len(nifti_paths):,}[/bold] NIfTI paths "
        f"from '{input_path}'."
    )
    if groups is not None:
        n_groups = len({g for g in groups if g is not None})
        print_info(
            f"Keeping each of [bold]{n_groups:,}[/bold] groups in one split."
        )

    # --- Split config ---
    output_path = Path(ns.output)
    split_cfg = _read_or_create_split_config(output_path)
    split_config = {k: v for k, v in split_cfg.items() if k != "seed"}
    split_seed = int(split_cfg.get("seed", 42))
    # Validate before the (possibly hours-long) stats pass, not after it.
    try:
        split_targets(split_config, len(nifti_paths))
    except ValueError as exc:
        print_error(f"Invalid split config: {exc}")
        sys.exit(1)

    # --- Build index ---
    _, errors = build_index(
        nifti_paths=nifti_paths,
        output_path=output_path,
        num_workers=ns.num_workers_index,
        split_config=split_config,
        split_seed=split_seed,
        groups=groups,
    )

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    index_entry()
