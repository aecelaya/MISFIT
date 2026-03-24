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
from argparse import ArgumentDefaultsHelpFormatter
from typing import Optional, List

from misfit.cli.args import ArgParser, add_train_args
from misfit.training.trainers.mae_trainer import MAETrainer


def _parse_args(args=None):
    parser = ArgParser(
        prog="misfit_train",
        description="MISFIT MAE pretraining — single-GPU to multi-node.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_train_args(parser)
    return parser.parse_args(args)


def train_entry(args=None) -> None:
    """Entrypoint for the misfit_train CLI command."""
    ns = _parse_args(args)
    trainer = MAETrainer(ns)
    trainer.train()
