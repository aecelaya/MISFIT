"""CLI entrypoint for ``misfit_encode`` — extract raw spatial encoder features.

Unlike ``misfit_embed``, which GAP-pools each crop to a single ``(C,)`` vector,
``misfit_encode`` saves the full bottleneck feature map for every crop, giving
``(N_crops, C, D', H', W')`` where ``D'/H'/W' = patch_size / 32`` (e.g. 3 for
a 96-voxel crop).  These richer spatial features are useful for training
aggregators that need to reason about spatial structure within each crop.

The cache deliberately retains this within-crop spatial structure for
downstream flexibility, even though the current embedding path
(``misfit_embed_train`` via ``CropFeaturesDataset``) GAP-pools it away to a
single ``(C,)`` vector per crop. Keeping the full map means future
spatial-aware consumers can be added without re-running encoding.

Each volume is saved as a compressed ``.npz`` file with two arrays:

- ``feature_map``: ``(N_crops, C, D', H', W')`` float32
- ``positions``:   ``(N_crops, 3)`` float32 — normalised 3-D crop centres

Usage::

    misfit_encode \\
        --encoder-checkpoint /runs/exp1/models/best_model.pt \\
        --index              /data/index.parquet \\
        --config             /runs/exp1/config.json \\
        --output-dir         /data/encodings
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from misfit.cli.args import ArgParser, add_encode_args
from misfit.utils import console, get_progress_bar, print_error
from misfit.utils.io import read_json_file


def encode_entry(args=None) -> None:
    """Extract per-crop spatial encoder feature maps and save as ``.npz`` files."""
    parser = ArgParser(
        prog="misfit_encode",
        description=(
            "Tile each volume into crops and save the full encoder bottleneck "
            "feature map for every crop as a compressed .npz file.  "
            "Produces (N_crops, C, D', H', W') feature maps suitable for "
            "training spatial aggregators."
        ),
    )
    add_encode_args(parser)
    ns = parser.parse_args(args)

    config_path = Path(ns.config)
    if not config_path.exists():
        print_error(f"--config '{config_path}' does not exist.")
        sys.exit(1)

    config = read_json_file(config_path)
    model_config = config.get("model", {})
    patch_size = tuple(model_config["patch_size"])

    from misfit.inference.inference_utils import (
        build_model_from_checkpoint,
        get_default_device,
        load_and_normalise,
        load_checkpoint,
    )

    device = torch.device(ns.device) if ns.device else get_default_device()
    output_dir = Path(ns.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(ns.encoder_checkpoint, device)
    model = build_model_from_checkpoint(checkpoint, model_config, device)
    model.eval()

    # Load index and optionally filter to a single split.
    index_df = pd.read_parquet(ns.index)
    if ns.split and "split" in index_df.columns:
        index_df = index_df[index_df["split"] == ns.split].reset_index(drop=True)

    console.print(
        f"[bold]Encoding {len(index_df)} volumes[/bold]  "
        f"(patch_size={patch_size})"
    )

    with get_progress_bar() as progress:
        task = progress.add_task("Encoding", total=len(index_df))
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
                feature_map, positions = _encode_volume(
                    volume, model, patch_size, device
                )
                np.savez_compressed(
                    out_path,
                    feature_map=feature_map,
                    positions=positions,
                )
            except Exception as exc:
                print_error(f"Skipping {row['volume_id']}: {exc}")

            progress.advance(task)

    console.print(f"[green]Encodings saved to {output_dir}[/green]")


def _encode_volume(
    volume: np.ndarray,
    model: torch.nn.Module,
    patch_size: int | tuple[int, int, int],
    device: torch.device,
) -> tuple:
    """Tile *volume* into crops and return raw bottleneck feature maps.

    Args:
        volume: Normalised float32 array of shape ``(D, H, W)``.
        model: Pretrained MISFIT model with a ``.encoder`` attribute.
        patch_size: Crop size in voxels. Either a single int (cubic crop) or a
            ``(D, H, W)`` sequence for anisotropic crops.
        device: Torch device for inference.

    Returns:
        feature_map: ``(N_crops, C, D', H', W')`` float32 ndarray.
        positions:   ``(N_crops, 3)`` float32 ndarray of normalised crop centres.
    """
    pd_, ph_, pw_ = (
        (patch_size, patch_size, patch_size)
        if isinstance(patch_size, int)
        else tuple(patch_size)
    )
    volume_t = torch.from_numpy(volume).float().unsqueeze(0).to(device)  # (1, D, H, W)

    # Pad so every dimension is divisible by its patch_size.
    _, D, H, W = volume_t.shape
    pad_d = (pd_ - D % pd_) % pd_
    pad_h = (ph_ - H % ph_) % ph_
    pad_w = (pw_ - W % pw_) % pw_
    if pad_d or pad_h or pad_w:
        volume_t = torch.nn.functional.pad(volume_t, (0, pad_w, 0, pad_h, 0, pad_d))

    _, D_pad, H_pad, W_pad = volume_t.shape
    nd, nh, nw = D_pad // pd_, H_pad // ph_, W_pad // pw_

    feature_maps = []
    positions = []

    with torch.no_grad():
        for id_ in range(nd):
            for ih in range(nh):
                for iw in range(nw):
                    d0, h0, w0 = id_ * pd_, ih * ph_, iw * pw_
                    crop = volume_t[:, d0:d0 + pd_, h0:h0 + ph_, w0:w0 + pw_]
                    crop = crop.unsqueeze(0)  # (1, 1, Pd, Ph, Pw)
                    feat = model.encoder(crop)[-1]  # (1, C, D', H', W')
                    # (C, D', H', W')
                    feature_maps.append(feat.squeeze(0).cpu().numpy())

                    cd = (d0 + pd_ / 2) / D_pad
                    ch = (h0 + ph_ / 2) / H_pad
                    cw = (w0 + pw_ / 2) / W_pad
                    positions.append([cd, ch, cw])

    return (
        np.stack(feature_maps, axis=0).astype(np.float32),  # (N, C, D', H', W')
        np.array(positions, dtype=np.float32),               # (N, 3)
    )


if __name__ == "__main__":
    encode_entry()
