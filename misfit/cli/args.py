"""Shared argument parsing helpers for MISFIT CLI commands.

Defines validator types, the ``ArgParser`` convenience subclass, and
``add_*_args`` functions so that argument definitions are written once and
shared between the individual entrypoints and the ``misfit_run`` chained
command.
"""
import argparse
from argparse import ArgumentParser
from typing import Union

import misfit.loss_functions  # noqa: F401 — trigger registrations
import misfit.models  # noqa: F401 — trigger registrations
from misfit.loss_functions.loss_registry import list_registered_losses
from misfit.metrics.metrics_registry import list_registered_metrics
from misfit.models.model_registry import list_registered_models
from misfit.training.lr_schedulers.lr_scheduler_registry import list_lr_schedulers
from misfit.training.optimizers.optimizer_registry import list_optimizers


# ---------------------------------------------------------------------------
# Validator types
# ---------------------------------------------------------------------------

def positive_int(value: Union[str, int]) -> int:
    """Argparse type: integer > 0."""
    v = int(value)
    if v <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer but got {value}."
        )
    return v


def positive_float(value: Union[str, float]) -> float:
    """Argparse type: float > 0."""
    v = float(value)
    if v <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a positive float but got {value}."
        )
    return v


def non_negative_int(value: Union[str, int]) -> int:
    """Argparse type: integer >= 0."""
    v = int(value)
    if v < 0:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative integer but got {value}."
        )
    return v


def float_0_1(value: Union[str, float]) -> float:
    """Argparse type: float in [0, 1]."""
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(
            f"Expected a float in [0, 1] but got {value}."
        )
    return v


# ---------------------------------------------------------------------------
# ArgParser convenience subclass
# ---------------------------------------------------------------------------

class ArgParser(ArgumentParser):
    """ArgumentParser with shorthand helpers for common patterns."""

    def arg(self, *args, **kwargs):
        """Shorthand for :meth:`add_argument`."""
        return super().add_argument(*args, **kwargs)

    def flag(self, *args, **kwargs):
        """Add a ``store_true`` boolean flag."""
        return super().add_argument(*args, action="store_true", **kwargs)


# ---------------------------------------------------------------------------
# Argument group builders
# ---------------------------------------------------------------------------

def add_index_args(parser: ArgParser, source_required: bool = True) -> None:
    """Add ``misfit_index`` arguments to *parser*.

    Adds a mutually exclusive ``--data-dir`` / ``--manifest`` source group
    plus ``--output`` and ``--num-workers``.

    Args:
        parser: The :class:`ArgParser` to populate.
        source_required: Whether the source group is required. Set to
            ``False`` when composing inside ``misfit_run`` if you want to
            make both stages optional. Defaults to ``True``.
    """
    source = parser.add_mutually_exclusive_group(required=source_required)
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

    g = parser.add_argument_group("Index")
    g.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="PARQUET",
        help="Destination path for the output Parquet index file.",
    )
    g.add_argument(
        "--num-workers",
        type=positive_int,
        default=32,
        metavar="N",
        help=(
            "Number of parallel worker processes for indexing. "
            "32–64 is a reasonable starting point on HPC nodes. Defaults to 32."
        ),
    )


def add_train_args(parser: ArgParser) -> None:
    """Add ``misfit_train`` arguments to *parser*.

    Args:
        parser: The :class:`ArgParser` to populate.
    """
    # --- Data ---
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--index-train", required=True, metavar="PARQUET",
        help="Path to training Parquet index (from misfit_index).",
    )
    data.add_argument(
        "--index-val", required=True, metavar="PARQUET",
        help="Path to validation Parquet index (from misfit_index).",
    )
    data.add_argument(
        "--num-workers-train", type=positive_int, default=8, metavar="N",
        dest="num_workers",
        help="DataLoader worker processes per GPU.",
    )

    # --- Output ---
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--results", required=True, metavar="DIR",
        help="Output directory for checkpoints, best model, and TensorBoard logs.",
    )

    # --- Model ---
    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model", default="swinmae-base",
        choices=list_registered_models(),
        help="SwinMAE variant (controls feature_size: 24 / 48 / 96).",
    )
    model.add_argument(
        "--in-channels", type=positive_int, default=1, metavar="C",
        help="Number of input image channels.",
    )
    model.add_argument(
        "--patch-size", type=positive_int, nargs=3, default=[96, 96, 96],
        metavar=("D", "H", "W"),
        help="Spatial crop size fed to the model. Must be divisible by 32.",
    )
    model.add_argument(
        "--mask-patch-size", type=positive_int, default=16, metavar="P",
        help="Edge length of each masked 3D cube (voxels).",
    )
    model.add_argument(
        "--mask-ratio", type=float_0_1, default=0.75, metavar="R",
        help="Fraction of patches to mask during pretraining.",
    )

    # --- Loss ---
    loss = parser.add_argument_group("Loss")
    loss.add_argument(
        "--loss", default="normalized_masked_mse",
        choices=list_registered_losses(),
        help=(
            "Reconstruction loss. normalized_masked_mse is recommended "
            "for multi-modality datasets (CT + MRI)."
        ),
    )

    # --- Optimisation ---
    opt = parser.add_argument_group("Optimisation")
    opt.add_argument("--epochs",        type=non_negative_int, default=200)
    opt.add_argument(
        "--batch-size", type=positive_int, default=2, metavar="N",
        help="Batch size per GPU.",
    )
    opt.add_argument(
        "--optimizer", default="adamw",
        choices=list_optimizers(),
    )
    opt.add_argument(
        "--learning-rate", type=positive_float, default=1e-4, metavar="LR",
    )
    opt.add_argument(
        "--weight-decay", type=positive_float, default=0.05, metavar="WD",
    )
    opt.add_argument(
        "--lr-scheduler", default="cosine",
        choices=list_lr_schedulers(),
    )
    opt.add_argument(
        "--warmup-epochs", type=non_negative_int, default=20, metavar="N",
        help="Linear warmup epochs (recommended ≥10 for SwinUNETR-V2).",
    )
    opt.add_argument(
        "--amp", action="store_true",
        help="Enable automatic mixed precision (float16) training.",
    )

    # --- Reproducibility & resumption ---
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument("--seed", type=non_negative_int, default=42)
    misc.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint in --results/checkpoints/.",
    )


def add_evaluate_args(parser: ArgParser) -> None:
    """Add ``misfit_evaluate`` arguments to *parser*.

    Args:
        parser: The :class:`ArgParser` to populate.
    """
    # --- Input ---
    inp = parser.add_argument_group("Evaluate: Input")
    inp.add_argument(
        "--checkpoint", required=True, metavar="PT",
        help="Path to a pretrained checkpoint (.pt) produced by misfit_train.",
    )
    inp.add_argument(
        "--index-val", required=True, metavar="PARQUET",
        help="Path to the validation Parquet index (from misfit_index).",
    )

    # --- Output ---
    out = parser.add_argument_group("Evaluate: Output")
    out.add_argument(
        "--results", required=True, metavar="DIR",
        help="Output directory for evaluation_results.csv.",
    )

    # --- Metrics ---
    met = parser.add_argument_group("Evaluate: Metrics")
    met.add_argument(
        "--metrics",
        nargs="+",
        default=list_registered_metrics(),
        choices=list_registered_metrics(),
        metavar="METRIC",
        help=(
            "Metrics to compute. Defaults to all registered metrics: "
            f"{list_registered_metrics()}."
        ),
    )

    # --- Inference ---
    inf = parser.add_argument_group("Evaluate: Inference")
    inf.add_argument(
        "--device", type=str, default=None, metavar="DEVICE",
        help=(
            "Torch device for inference (e.g. 'cuda:0', 'cpu'). "
            "Defaults to 'cuda' if available, else 'cpu'."
        ),
    )
    inf.add_argument(
        "--amp", action="store_true",
        help="Use automatic mixed precision for inference.",
    )
