"""CLI entrypoint for misfit_infer — encoder feature extraction and reconstruction.

Usage::

    # Extract encoder bottleneck features (saved as .npy)
    misfit_infer \\
        --checkpoint best_model.pt \\
        --index index.parquet \\
        --mode features \\
        --output-dir /runs/exp1/features

    # Reconstruct volumes (saved as .nii.gz)
    misfit_infer \\
        --checkpoint best_model.pt \\
        --index index.parquet \\
        --mode reconstruct \\
        --output-dir /runs/exp1/reconstructions

    # With all-flip TTA and AMP
    misfit_infer \\
        --checkpoint best_model.pt --index index.parquet \\
        --mode features --output-dir /runs/exp1/features \\
        --tta all_flips --amp
"""
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path

from misfit.cli.args import ArgParser, add_infer_args
from misfit.inference.inference_runners import extract_features, reconstruct


def _parse_args(args=None):
    parser = ArgParser(
        prog="misfit_infer",
        description=(
            "Run MISFIT inference on a NIfTI index. "
            "Use --mode features to extract encoder representations "
            "or --mode reconstruct to generate MAE reconstructions."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    add_infer_args(parser)
    return parser.parse_args(args)


def infer_entry(args=None) -> None:
    """Entrypoint for the misfit_infer CLI command."""
    ns = _parse_args(args)

    kwargs = dict(
        index_path=Path(ns.index),
        checkpoint_path=Path(ns.checkpoint),
        output_dir=Path(ns.output_dir),
        tta_strategy=ns.tta_strategy,
        inferer_name=ns.inferer,
        device=ns.device,
        amp=ns.amp,
    )

    if ns.mode == "features":
        extract_features(**kwargs)
    else:
        reconstruct(**kwargs)
