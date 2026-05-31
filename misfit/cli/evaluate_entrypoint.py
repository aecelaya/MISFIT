"""CLI entrypoint for misfit_evaluate — reconstruction quality evaluation.

Usage::

    misfit_evaluate \\
        --checkpoint /runs/exp1/models/best_model.pt \\
        --config     /runs/exp1/config.json \\
        --index      index.parquet \\
        --output-csv /runs/exp1/eval/evaluation_results.csv

    # Run on a specific GPU
    misfit_evaluate \\
        --checkpoint /runs/exp1/models/best_model.pt \\
        --config     /runs/exp1/config.json \\
        --index      index.parquet \\
        --output-csv /runs/exp1/eval/evaluation_results.csv \\
        --device cuda:0

Metrics are read from the ``evaluation`` section of the config JSON::

    "evaluation": {
        "masked_mae":  {},
        "masked_mse":  {},
        "ssim":        {},
        "masked_psnr": {}
    }
"""
import sys
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

from misfit.cli.args import ArgParser, add_evaluate_args
from misfit.evaluation.evaluator import ReconstructionEvaluator
from misfit.utils.console import print_error
from misfit.utils.io import read_json_file


def _parse_args(args=None):
    parser = ArgParser(
        prog="misfit_evaluate",
        description=(
            "Evaluate MISFIT pretraining quality by running reconstruction "
            "inference on a held-out NIfTI index and computing metrics on "
            "the masked patches. Metrics are configured via config.json."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_evaluate_args(parser)
    return parser.parse_args(args)


def evaluate_entry(args=None) -> None:
    """Entrypoint for the misfit_evaluate CLI command."""
    ns = _parse_args(args)

    config_path = Path(ns.config)
    if not config_path.exists():
        print_error(f"--config '{config_path}' does not exist.")
        sys.exit(1)

    config = read_json_file(config_path)
    eval_section = config.get("evaluation", {})
    if not eval_section:
        print_error(
            f"'evaluation' section missing or empty in '{config_path}'. "
            "Re-run misfit_train to regenerate a valid config.json."
        )
        sys.exit(1)

    metrics = list(eval_section.keys())
    model_config = config.get("model", {})
    training_config = config.get("training", {})

    evaluator = ReconstructionEvaluator(
        checkpoint_path=Path(ns.checkpoint),
        index_path=Path(ns.index),
        output_csv_path=Path(ns.output_csv),
        model_config=model_config,
        metrics=metrics,
        device=ns.device,
        split=ns.split or None,
        training_config=training_config,
    )
    evaluator.run()
