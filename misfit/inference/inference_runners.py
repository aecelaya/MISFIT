"""High-level batch reconstruction runner for MISFIT.

``reconstruct``
    Runs the full MAE forward pass on every non-overlapping patch of each
    volume, stitches the results back into the original coordinate space,
    denormalises the intensities, and saves a ``.nii.gz`` file.  Used by
    ``misfit_inspect`` to visually assess pretraining quality.
"""
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from misfit.inference import inference_utils
from misfit.utils.console import print_section_header, print_success, print_warning
from misfit.utils.progress_bar import get_progress_bar


def _tiled_reconstruct(
    padded: np.ndarray,
    patch_size: Tuple[int, int, int],
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    device: Union[str, torch.device],
) -> np.ndarray:
    """Reconstruct *padded* volume patch-by-patch and stitch results.

    Args:
        padded: Zero-padded volume of shape (D', H', W') where each dimension
            is an exact multiple of the corresponding *patch_size* element.
        patch_size: Patch dimensions (pd, ph, pw).
        model_fn: Callable that maps a ``(1, 1, pd, ph, pw)`` tensor to a
            reconstructed tensor of the same shape.
        device: Torch device used for inference tensors.

    Returns:
        Reconstructed numpy array of shape (D', H', W').
    """
    pd_, ph_, pw_ = patch_size
    D, H, W = padded.shape
    output = np.zeros_like(padded)
    amp_ctx = (
        torch.amp.autocast("cuda")
        if torch.device(device).type == "cuda"
        else nullcontext()
    )

    for di in range(0, D, pd_):
        for hi in range(0, H, ph_):
            for wi in range(0, W, pw_):
                patch = padded[di:di + pd_, hi:hi + ph_, wi:wi + pw_]
                tensor = (
                    torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                )
                with torch.no_grad(), amp_ctx:
                    recon = model_fn(tensor)
                output[di:di + pd_, hi:hi + ph_, wi:wi + pw_] = (
                    recon.squeeze().cpu().numpy()
                )

    return output


def reconstruct(
    index_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_config: Dict,
    device: Optional[Union[str, torch.device]] = None,
) -> None:
    """Reconstruct every volume in *index_path* and save as NIfTI.

    For each volume the function:

    1. Loads and z-score normalises the NIfTI data.
    2. Zero-pads to the nearest multiple of *patch_size* in every dimension.
    3. Tiles the padded volume into non-overlapping patches and runs MAE
       reconstruction on each patch.
    4. Stitches the reconstructed patches back into the full padded volume.
    5. Trims padding to restore the original voxel dimensions.
    6. Denormalises intensities (``recon × fg_std + fg_mean``).
    7. Saves ``<volume_id>.nii.gz`` under *output_dir* using the affine stored
       in the index.

    Args:
        index_path: Parquet index produced by ``misfit_index``.
        checkpoint_path: Pretrained MISFIT checkpoint (``.pt``).
        output_dir: Directory where ``<volume_id>.nii.gz`` files are written.
        model_config: Model configuration dict (``config["model"]`` from
            ``config.json``).
        device: Torch device. Defaults to CUDA if available, else CPU.
    """
    device = device or inference_utils.get_default_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = inference_utils.load_checkpoint(checkpoint_path, device)
    patch_size = tuple(model_config["patch_size"])

    model = inference_utils.build_model_from_checkpoint(checkpoint, model_config, device)
    model_fn = lambda x: model(x)["reconstruction"]  # noqa: E731

    index_df = pd.read_parquet(index_path)
    n_total = len(index_df)
    errors: List[str] = []

    print_section_header(
        f"Reconstructing: {n_total:,} volumes → {output_dir}"
    )

    with get_progress_bar() as progress:
        task = progress.add_task("Reconstruction", total=n_total)

        for _, row in index_df.iterrows():
            volume_id = row["volume_id"]
            out_path = output_dir / f"{volume_id}.nii.gz"

            volume = inference_utils.load_and_normalise(
                row["path"], row["p1"], row["p99"], row["fg_mean"], row["fg_std"]
            )
            if volume is None:
                errors.append(f"Could not load: {row['path']}")
                progress.advance(task)
                continue

            try:
                affine = np.array(json.loads(row["affine"]), dtype=np.float64)

                padded, original_shape = inference_utils.pad_to_multiple(volume, patch_size)
                recon_padded = _tiled_reconstruct(padded, patch_size, model_fn, device)

                d, h, w = original_shape
                recon_np = recon_padded[:d, :h, :w]

                fg_std  = float(row["fg_std"])
                fg_mean = float(row["fg_mean"])
                recon_np = recon_np * fg_std + fg_mean

                nib.save(nib.Nifti1Image(recon_np.astype(np.float32), affine=affine), out_path)
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"Reconstruction failed for {volume_id}: {exc}")

            progress.advance(task)

    if errors:
        print_warning("\n".join(errors))
    print_success(
        f"Reconstructions saved for {n_total - len(errors):,}/{n_total:,} volumes."
    )
