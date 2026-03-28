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

def add_index_args(parser: ArgParser, input_required: bool = True) -> None:
    """Add ``misfit_index`` arguments to *parser*.

    Args:
        parser: The :class:`ArgParser` to populate.
        input_required: Whether ``--input`` is required. Set to ``False``
            when composing inside ``misfit_run``. Defaults to ``True``.
    """
    g = parser.add_argument_group("Index")
    g.add_argument(
        "--input",
        type=str,
        required=input_required,
        metavar="FILE",
        help=(
            "CSV or Parquet file with a 'path' column listing absolute paths "
            "to NIfTI files. CSV files must have a header row."
        ),
    )
    g.add_argument(
        "--output",
        type=str,
        required=True,
        metavar="PARQUET",
        help="Destination path for the output Parquet index file.",
    )
    g.add_argument(
        "--num-workers-index",
        type=positive_int,
        default=32,
        metavar="N",
        dest="num_workers_index",
        help=(
            "Number of parallel worker processes for indexing. "
            "32–64 is a reasonable starting point on HPC nodes. Defaults to 32."
        ),
    )


def add_train_args(parser: ArgParser, index_required: bool = True) -> None:
    """Add ``misfit_train`` arguments to *parser*.

    Args:
        parser: The :class:`ArgParser` to populate.
        index_required: Whether ``--index`` is a required argument. Set to
            ``False`` when composing inside ``misfit_run``, where the index
            path is derived from ``--output`` at runtime. Defaults to True.
    """
    # --- Data ---
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--index", required=index_required, metavar="PARQUET",
        help=(
            "Path to the Parquet index produced by misfit_index. "
            "Volumes are split into train/val/test subsets using the "
            "'split' column assigned at index time."
        ),
    )
    data.add_argument(
        "--num-cpu-workers", type=positive_int, default=8, metavar="N",
        dest="num_cpu_workers",
        help="Number of CPU worker processes for data loading. Defaults to 8.",
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
    # --- Reproducibility & resumption ---
    misc = parser.add_argument_group("Miscellaneous")
    misc.add_argument("--seed", type=non_negative_int, default=42)

    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume", action="store_true",
        help=(
            "Resume training from the latest checkpoint and config.json in "
            "--results.  Model architecture and patch size must match the "
            "saved config; other hyperparameter changes emit warnings."
        ),
    )
    run_mode.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Overwrite an existing --results directory and start fresh.  "
            "Without this flag, misfit_train refuses to run if config.json "
            "already exists in --results (use --resume to continue instead)."
        ),
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
        "--index", required=True, metavar="PARQUET",
        help="Path to the Parquet index (from misfit_index). Only 'val' split rows are evaluated.",
    )
    inp.add_argument(
        "--config", required=True, metavar="JSON",
        help=(
            "Path to the config.json produced by misfit_train. "
            "Metrics to compute are read from the 'evaluation' section."
        ),
    )

    # --- Output ---
    out = parser.add_argument_group("Evaluate: Output")
    out.add_argument(
        "--output-csv", required=True, metavar="CSV",
        help="Path where the evaluation results CSV will be written.",
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


def add_embed_args(parser: ArgParser) -> None:
    """Add ``misfit_embed`` arguments to *parser*.

    Args:
        parser: The :class:`ArgParser` to populate.
    """
    # --- Input ---
    inp = parser.add_argument_group("Embed: Input")
    inp.add_argument(
        "--encoder-checkpoint", required=True, metavar="PT",
        dest="encoder_checkpoint",
        help="Path to a pretrained MISFIT encoder checkpoint (.pt).",
    )
    inp.add_argument(
        "--index", required=True, metavar="PARQUET",
        help="Path to the Parquet index of volumes to embed.",
    )

    # --- Output ---
    out = parser.add_argument_group("Embed: Output")
    out.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help=(
            "Directory where per-volume .npz files are saved.  "
            "Each file contains 'features (N_crops, C)' and "
            "'positions (N_crops, 3)'."
        ),
    )

    # --- Model config ---
    cfg = parser.add_argument_group("Embed: Configuration")
    cfg.add_argument(
        "--config", required=True, metavar="JSON",
        help=(
            "Path to the config.json produced by misfit_train. "
            "Model architecture and patch size are read from the 'model' section."
        ),
    )
    cfg.add_argument(
        "--aggregator",
        default="mean_pool",
        metavar="NAME",
        help=(
            "Aggregator to use for producing the global embedding.  "
            "Defaults to 'mean_pool' (zero-shot).  Use 'attention_pool' "
            "after training with misfit_embed_train."
        ),
    )
    cfg.add_argument(
        "--aggregator-checkpoint",
        default=None,
        metavar="PT",
        help=(
            "Path to a trained aggregator checkpoint produced by "
            "misfit_embed_train.  Required when --aggregator=attention_pool."
        ),
    )
    cfg.add_argument(
        "--device", type=str, default=None, metavar="DEVICE",
        help="Torch device (e.g. 'cuda:0', 'cpu').",
    )


def add_embed_train_args(parser: ArgParser) -> None:
    """Add ``misfit_embed_train`` arguments to *parser*.

    Args:
        parser: The :class:`ArgParser` to populate.
    """
    import misfit.embedding  # noqa: F401 — trigger registrations
    from misfit.embedding.aggregators.aggregator_registry import list_aggregators
    from misfit.embedding.objectives.objective_registry import list_objectives

    # --- Input ---
    inp = parser.add_argument_group("Embed Train: Input")
    inp.add_argument(
        "--input", required=True, metavar="CSV",
        help=(
            "Unified CSV with columns: volume_id, split, features_path, label.  "
            "Only rows where split='train' are used for training.  "
            "features_path must be the absolute path to each volume's .npz file "
            "produced by misfit_embed."
        ),
    )

    # --- Output ---
    out = parser.add_argument_group("Embed Train: Output")
    out.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Output directory for the trained aggregator.pt and logs.",
    )

    # --- Model ---
    mdl = parser.add_argument_group("Embed Train: Model")
    mdl.add_argument(
        "--aggregator",
        default="attention_pool",
        choices=list_aggregators(),
        help="Aggregator architecture to train. Defaults to 'attention_pool'.",
    )
    mdl.add_argument(
        "--objective",
        default="classification",
        choices=list_objectives(),
        help=(
            "Training objective.  'classification': cross-entropy.  "
            "'contrastive': Supervised Contrastive (K=2 pairs per group).  "
            "Defaults to 'classification'."
        ),
    )
    mdl.add_argument(
        "--embed-dim", type=positive_int, required=True, metavar="C",
        help=(
            "Dimensionality of the encoder bottleneck features (C).  "
            "Must match the feature files produced by misfit_embed."
        ),
    )
    mdl.add_argument(
        "--no-position-encoding", action="store_true",
        help=(
            "Disable learned 3-D position encoding in AttentionPoolAggregator."
        ),
    )

    # --- Training ---
    trn = parser.add_argument_group("Embed Train: Training")
    trn.add_argument(
        "--epochs", type=positive_int, default=50, metavar="N",
        help="Number of training epochs. Defaults to 50.",
    )
    trn.add_argument(
        "--batch-size", type=positive_int, default=32, metavar="N",
        help=(
            "Batch size.  For contrastive training this must be even "
            "(M × 2 pairs). Defaults to 32."
        ),
    )
    trn.add_argument(
        "--learning-rate", type=positive_float, default=1e-3, metavar="LR",
        help="Initial learning rate for Adam. Defaults to 1e-3.",
    )
    trn.add_argument(
        "--num-workers-embed", type=positive_int, default=4, metavar="N",
        dest="num_workers",
        help="DataLoader worker processes. Defaults to 4.",
    )
    trn.add_argument(
        "--device", type=str, default="cuda", metavar="DEVICE",
        help="Torch device (e.g. 'cuda', 'cpu'). Defaults to 'cuda'.",
    )


def add_inspect_args(parser: ArgParser) -> None:
    """Add ``misfit_inspect`` arguments to *parser*.

    Args:
        parser: The :class:`ArgParser` to populate.
    """
    # --- Input ---
    inp = parser.add_argument_group("Inspect: Input")
    inp.add_argument(
        "--checkpoint", required=True, metavar="PT",
        help="Path to a pretrained MISFIT checkpoint (.pt).",
    )
    inp.add_argument(
        "--index", required=True, metavar="PARQUET",
        help="Path to the Parquet index of volumes to reconstruct.",
    )

    # --- Output ---
    out = parser.add_argument_group("Inspect: Output")
    out.add_argument(
        "--output-dir", required=True, metavar="DIR",
        help="Directory where reconstructed .nii.gz files are written.",
    )

    # --- Inference config ---
    cfg = parser.add_argument_group("Inspect: Configuration")
    cfg.add_argument(
        "--config", required=True, metavar="JSON",
        help=(
            "Path to the config.json produced by misfit_train. "
            "Model architecture and patch size are read from the 'model' section."
        ),
    )
    cfg.add_argument(
        "--device", type=str, default=None, metavar="DEVICE",
        help=(
            "Torch device (e.g. 'cuda:0', 'cpu'). "
            "Defaults to 'cuda' if available, else 'cpu'."
        ),
    )
