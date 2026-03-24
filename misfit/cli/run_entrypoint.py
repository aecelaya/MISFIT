"""CLI entrypoint for misfit_run — index then train in one command.

Chains ``misfit_index`` → ``misfit_train`` so that a full pretraining run
can be launched with a single command::

    misfit_run \\
        --data-dir /data/niftis \\
        --output index.parquet \\
        --index-train index.parquet --index-val val.parquet \\
        --results /runs/exp1

Each stage receives only its own subset of arguments, derived from the
merged namespace via ``_ns_to_argv``.
"""
import argparse
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from typing import List, Optional

from misfit.cli import args as argmod
from misfit.cli.index_entrypoint import index_entry
from misfit.cli.train_entrypoint import train_entry


def _ns_to_argv(ns: argparse.Namespace, keys: List[str]) -> List[str]:
    """Convert a subset of Namespace fields into an ``argv``-style list.

    Handles booleans (flags), lists/tuples (nargs), and scalar values.
    """
    argv: List[str] = []
    for k in keys:
        if not hasattr(ns, k):
            continue
        v = getattr(ns, k)
        if v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                argv.append(flag)
        elif isinstance(v, (list, tuple)):
            if not v:
                continue
            argv.append(flag)
            argv.extend(str(x) for x in v)
        else:
            argv.extend([flag, str(v)])
    return argv


def _parse_run_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argmod.ArgParser(
        prog="misfit_run",
        description=(
            "Run misfit_index → misfit_train in one go. "
            "Accepts all arguments from both commands."
        ),
        conflict_handler="resolve",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    argmod.add_index_args(parser, source_required=True)
    argmod.add_train_args(parser)
    return parser.parse_args(argv)


def run_entry(argv: Optional[List[str]] = None) -> None:
    """Entrypoint for the misfit_run CLI command."""
    ns = _parse_run_args(argv)

    index_keys = [
        "data_dir", "manifest",
        "output", "num_workers",
    ]
    train_keys = [
        "index_train", "index_val", "num_workers",
        "results",
        "model", "in_channels", "patch_size", "mask_patch_size", "mask_ratio",
        "loss",
        "epochs", "batch_size", "optimizer", "learning_rate", "weight_decay",
        "lr_scheduler", "warmup_epochs", "amp",
        "seed", "resume",
    ]

    index_entry(_ns_to_argv(ns, index_keys))
    train_entry(_ns_to_argv(ns, train_keys))


if __name__ == "__main__":
    run_entry()  # pragma: no cover
