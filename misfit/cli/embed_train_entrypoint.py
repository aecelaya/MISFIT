"""CLI entrypoint for ``misfit_embed_train`` — train a crop aggregator."""
from pathlib import Path

from misfit.cli.args import ArgParser, add_embed_train_args
from misfit.utils import print_error


def embed_train_entry(args=None) -> None:
    """Train a crop aggregator on cached per-crop ``.npz`` feature files."""
    parser = ArgParser(
        prog="misfit_embed_train",
        description=(
            "Train a crop aggregator (mean_pool or attention_pool) on "
            "per-crop feature files produced by misfit_embed. "
            "Supports classification (cross-entropy) and contrastive "
            "(Supervised Contrastive, K=2) training objectives."
        ),
    )
    add_embed_train_args(parser)
    ns = parser.parse_args(args)

    import misfit.embedding  # noqa: F401 — trigger registrations
    from misfit.embedding.embed_trainer import EmbedTrainer

    try:
        trainer = EmbedTrainer(
            features_dir=Path(ns.features_dir),
            labels_csv=Path(ns.labels_csv),
            label_col=ns.label_col,
            objective_name=ns.objective,
            aggregator_name=ns.aggregator,
            embed_dim=ns.embed_dim,
            output_dir=Path(ns.output_dir),
            num_epochs=ns.epochs,
            batch_size=ns.batch_size,
            learning_rate=ns.learning_rate,
            num_workers=ns.num_workers,
            device=ns.device,
            use_position_encoding=not ns.no_position_encoding,
        )
        trainer.run()
    except (ValueError, RuntimeError) as exc:
        print_error(str(exc))
        raise SystemExit(1) from exc


if __name__ == "__main__":
    embed_train_entry()
