"""CLI entrypoint for ``misfit_embed``.

Encode a volume and aggregate to a single embedding vector.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from misfit.cli.args import ArgParser, add_embed_args
from misfit.utils import console, get_progress_bar, print_error
from misfit.utils.io import read_json_file


def embed_entry(args=None) -> None:
    """Encode each volume and save a global embedding vector as a ``.npz`` file."""
    parser = ArgParser(
        prog="misfit_embed",
        description=(
            "Tile each volume into crops, encode with the MISFIT encoder, "
            "and aggregate to a single global embedding vector.  "
            "With --aggregator=mean_pool (default) no trained aggregator is "
            "needed.  Pass --aggregator=attention_pool and "
            "--aggregator-checkpoint to use a trained aggregator."
        ),
    )
    add_embed_args(parser)
    ns = parser.parse_args(args)

    config_path = Path(ns.config)
    if not config_path.exists():
        print_error(f"--config '{config_path}' does not exist.")
        sys.exit(1)

    config = read_json_file(config_path)
    model_config = config.get("model", {})
    patch_size = tuple(model_config["patch_size"])

    # Lazy imports so CLI startup is fast.
    import misfit.embedding  # noqa: F401 — trigger registrations
    from misfit.embedding.aggregators.aggregator_registry import get_aggregator
    from misfit.embedding.embedder import Embedder
    from misfit.inference.inference_utils import (
        build_model_from_checkpoint,
        get_default_device,
        load_and_normalise,
        load_checkpoint,
    )

    device = torch.device(ns.device) if ns.device else get_default_device()
    output_dir = Path(ns.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build encoder.
    checkpoint = load_checkpoint(ns.encoder_checkpoint, device)
    model = build_model_from_checkpoint(checkpoint, model_config, device)
    model.eval()

    def encoder_fn(x: torch.Tensor) -> torch.Tensor:
        return model.encoder(x)[-1]

    # Build aggregator.
    aggregator_cls = get_aggregator(ns.aggregator)
    embed_dim: int = _infer_embed_dim(model, patch_size, device)

    agg_kwargs: dict = {"embed_dim": embed_dim}

    # Load the trained aggregator checkpoint first (if any) so the aggregator
    # can be rebuilt with the same architecture it was trained with. The
    # attention_pool aggregator only creates its position-projection layer when
    # position encoding is enabled, so constructing it with the wrong flag would
    # make load_state_dict fail on a missing/unexpected key.
    agg_ckpt: dict | None = None
    if ns.aggregator_checkpoint:
        agg_ckpt = torch.load(ns.aggregator_checkpoint, map_location=device)
        if ns.aggregator == "attention_pool" and "use_position_encoding" in agg_ckpt:
            agg_kwargs["use_position_encoding"] = agg_ckpt["use_position_encoding"]

    aggregator = aggregator_cls(**agg_kwargs).to(device)
    aggregator.eval()

    if agg_ckpt is not None:
        aggregator.load_state_dict(agg_ckpt["aggregator_state"])

    embedder = Embedder(
        encoder_fn=encoder_fn,
        aggregator=aggregator,
        patch_size=patch_size,
        device=device,
    )

    # Load index and optionally filter to a single split.
    index_df = pd.read_parquet(ns.index)
    if ns.split and "split" in index_df.columns:
        index_df = index_df[index_df["split"] == ns.split].reset_index(drop=True)

    console.print(
        f"[bold]Embedding {len(index_df)} volumes[/bold]  "
        f"(aggregator={ns.aggregator}, patch_size={patch_size})"
    )

    with get_progress_bar() as progress:
        task = progress.add_task("Embedding", total=len(index_df))
        for _, row in index_df.iterrows():
            out_path = output_dir / f"{row['volume_id']}.npz"
            if out_path.exists():
                progress.advance(task)
                continue

            volume = load_and_normalise(
                row["path"], row["p1"], row["p99"], row["fg_mean"], row["fg_std"]
            )
            if volume is None:
                print_error(
                    f"Skipping {row['volume_id']}: could not load {row['path']}"
                )
                progress.advance(task)
                continue

            try:
                volume_t = torch.from_numpy(volume).float().unsqueeze(0)  # (1, D, H, W)
                embedding = embedder.embed(volume_t)  # (C,) global vector
                np.savez_compressed(out_path, embedding=embedding.cpu().numpy())
            except Exception as exc:
                print_error(f"Skipping {row['volume_id']}: {exc}")

            progress.advance(task)

    console.print(f"[green]Embeddings saved to {output_dir}[/green]")


def _infer_embed_dim(
    model, patch_size: int | tuple[int, int, int], device: torch.device
) -> int:
    """Run a dummy forward pass to discover the encoder output channels."""
    ps = (
        (patch_size, patch_size, patch_size)
        if isinstance(patch_size, int)
        else tuple(patch_size)
    )
    with torch.no_grad():
        dummy = torch.zeros(1, 1, *ps, device=device)
        out = model.encoder(dummy)[-1]
    return out.shape[1]


if __name__ == "__main__":
    embed_entry()
