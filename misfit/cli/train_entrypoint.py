"""CLI entrypoint for misfit_train — MAE pretraining.

Single-GPU::

    misfit_train --index-train train.parquet --index-val val.parquet \\
        --results /runs/exp1 --model swinmae-base

Multi-GPU (single node)::

    torchrun --nproc_per_node=4 $(which misfit_train) \\
        --index-train train.parquet --index-val val.parquet \\
        --results /runs/exp1 --model swinmae-base

Multi-node (4 nodes × 8 GPUs)::

    torchrun --nproc_per_node=8 --nnodes=4 \\
             --node_rank=<rank> --master_addr=<addr> --master_port=29500 \\
             $(which misfit_train) \\
        --index-train train.parquet --index-val val.parquet \\
        --results /runs/exp1 --model swinmae-base
"""
import argparse

from misfit.training.trainers.mae_trainer import MAETrainer


def _parse_args(args=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="misfit_train",
        description="MISFIT MAE pretraining — single-GPU to multi-node.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    data = p.add_argument_group("Data")
    data.add_argument(
        "--index-train", required=True, metavar="PARQUET",
        help="Path to training Parquet index (from misfit_index).",
    )
    data.add_argument(
        "--index-val", required=True, metavar="PARQUET",
        help="Path to validation Parquet index (from misfit_index).",
    )
    data.add_argument(
        "--num-workers", type=int, default=8, metavar="N",
        help="DataLoader worker processes per GPU.",
    )

    # --- Output ---
    out = p.add_argument_group("Output")
    out.add_argument(
        "--results", required=True, metavar="DIR",
        help="Output directory for checkpoints, best model, and logs.",
    )

    # --- Model ---
    model = p.add_argument_group("Model")
    model.add_argument(
        "--model", default="swinmae-base",
        choices=["swinmae-small", "swinmae-base", "swinmae-large"],
        help="SwinMAE variant (controls feature_size: 24 / 48 / 96).",
    )
    model.add_argument(
        "--in-channels", type=int, default=1, metavar="C",
        help="Number of input image channels.",
    )
    model.add_argument(
        "--patch-size", type=int, nargs=3, default=[96, 96, 96],
        metavar=("D", "H", "W"),
        help="Spatial crop size fed to the model. Must be divisible by 32.",
    )
    model.add_argument(
        "--mask-patch-size", type=int, default=16, metavar="P",
        help="Edge length of each masked 3D cube (voxels).",
    )
    model.add_argument(
        "--mask-ratio", type=float, default=0.75, metavar="R",
        help="Fraction of patches to mask during pretraining.",
    )

    # --- Loss ---
    loss = p.add_argument_group("Loss")
    loss.add_argument(
        "--loss", default="normalized_masked_mse",
        choices=["masked_mse", "masked_l1", "normalized_masked_mse"],
        help="Reconstruction loss. normalized_masked_mse is recommended "
             "for multi-modality datasets (CT + MRI).",
    )

    # --- Optimisation ---
    opt = p.add_argument_group("Optimisation")
    opt.add_argument("--epochs",        type=int,   default=200)
    opt.add_argument("--batch-size",    type=int,   default=2,    metavar="N",
                     help="Batch size per GPU.")
    opt.add_argument("--optimizer",     default="adamw",
                     choices=["adam", "adamw", "sgd"])
    opt.add_argument("--learning-rate", type=float, default=1e-4, metavar="LR")
    opt.add_argument("--weight-decay",  type=float, default=0.05, metavar="WD")
    opt.add_argument("--lr-scheduler",  default="cosine",
                     choices=["cosine", "polynomial", "constant"])
    opt.add_argument(
        "--warmup-epochs", type=int, default=20, metavar="N",
        help="Linear warmup epochs (recommended ≥10 for SwinUNETR-V2).",
    )
    opt.add_argument(
        "--amp", action="store_true",
        help="Enable automatic mixed precision (float16) training.",
    )

    # --- Reproducibility & resumption ---
    misc = p.add_argument_group("Miscellaneous")
    misc.add_argument("--seed",   type=int, default=42)
    misc.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint in --results/checkpoints/.",
    )

    return p.parse_args(args)


def train_entry(args=None) -> None:
    """Entrypoint for the misfit_train CLI command."""
    ns = _parse_args(args)
    trainer = MAETrainer(ns)
    trainer.train()
