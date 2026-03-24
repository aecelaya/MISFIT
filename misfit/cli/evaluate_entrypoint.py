"""CLI entrypoint for misfit_evaluate — reconstruction quality evaluation.

Usage::

    misfit_evaluate \\
        --checkpoint /runs/exp1/models/best_model.pt \\
        --index-val val.parquet \\
        --results /runs/exp1/eval

    # Select specific metrics
    misfit_evaluate \\
        --checkpoint best_model.pt --index-val val.parquet \\
        --results /runs/exp1/eval \\
        --metrics masked_mae ssim

    # AMP inference on a specific GPU
    misfit_evaluate \\
        --checkpoint best_model.pt --index-val val.parquet \\
        --results /runs/exp1/eval \\
        --device cuda:0 --amp
"""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

from misfit.cli.args import ArgParser, add_evaluate_args
from misfit.evaluation.evaluator import ReconstructionEvaluator


def _parse_args(args=None):
    parser = ArgParser(
        prog="misfit_evaluate",
        description=(
            "Evaluate MISFIT pretraining quality by running reconstruction "
            "inference on a held-out NIfTI index and computing SSIM, PSNR, "
            "MAE, and MSE on the masked patches."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_evaluate_args(parser)
    return parser.parse_args(args)


def evaluate_entry(args=None) -> None:
    """Entrypoint for the misfit_evaluate CLI command."""
    ns = _parse_args(args)

    evaluator = ReconstructionEvaluator(
        checkpoint_path=Path(ns.checkpoint),
        index_path=Path(ns.index_val),
        results_dir=Path(ns.results),
        metrics=ns.metrics,
        device=ns.device,
        amp=ns.amp,
    )
    evaluator.run()
