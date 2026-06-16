"""High-level batch reconstruction runner for MISFIT.

``reconstruct``
    Runs the full MAE forward pass on every non-overlapping patch of each
    volume, stitches the results back into the original coordinate space,
    denormalises the intensities, and saves outputs under two subdirectories:

    - ``reconstructions/`` — full-volume reconstruction NIfTIs.
    - ``masks/`` — binary masks using the model convention (1 = masked /
      reconstructed by the model, 0 = visible to the encoder), aggregated
      across all patches.

    Used by ``misfit_inspect`` to visually assess pretraining quality.
"""
import json
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from misfit.inference import inference_utils
from misfit.utils.console import print_section_header, print_success, print_warning
from misfit.utils.progress_bar import get_progress_bar


_NORMALIZED_MSE_LOSS = "normalized_masked_mse"


def _tiled_reconstruct(
    padded: np.ndarray,
    patch_size: tuple[int, int, int],
    model_fn: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    device: str | torch.device,
    denorm_patches: bool = False,
    amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct *padded* volume patch-by-patch and stitch results.

    Args:
        padded: Zero-padded volume of shape (D', H', W') where each dimension
            is an exact multiple of the corresponding *patch_size* element.
        patch_size: Patch dimensions (pd, ph, pw).
        model_fn: Callable that maps a ``(1, 1, pd, ph, pw)`` tensor to a
            dict with keys ``"reconstruction"`` and ``"mask"``.
        device: Torch device used for inference tensors.
        denorm_patches: When ``True``, undo the per-patch normalization from
            the model output before stitching. Set this when the model was
            trained with ``normalized_masked_mse`` so that the stitched
            reconstruction is in the same z-score space as the input (and can
            be correctly denormalised to original intensities afterwards).
            Defaults to ``False``.
        amp: When ``True`` (default), run the forward pass under bfloat16
            autocast on CUDA devices. Has no effect on CPU.

    Returns:
        Tuple of ``(reconstruction, mask)`` numpy arrays, each of shape
        (D', H', W').  *mask* uses the model convention: 1 = masked voxels,
        0 = visible voxels.
    """
    pd_, ph_, pw_ = patch_size
    D, H, W = padded.shape
    recon_out = np.zeros_like(padded)
    mask_out = np.zeros_like(padded)
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if amp and torch.device(device).type == "cuda"
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
                    output = model_fn(tensor)
                recon = output["reconstruction"].squeeze().cpu().numpy()
                if denorm_patches:
                    # The model was trained to predict per-patch-normalised values
                    # (zero mean, unit variance per patch cube). Undo that using the
                    # target patch's own statistics so the stitched reconstruction is
                    # back in the volume-level z-score space.
                    patch_std = float(patch.std()) + 1e-6
                    patch_mean = float(patch.mean())
                    recon = recon * patch_std + patch_mean
                recon_out[di:di + pd_, hi:hi + ph_, wi:wi + pw_] = recon
                mask_out[di:di + pd_, hi:hi + ph_, wi:wi + pw_] = (
                    output["mask"].squeeze().cpu().numpy()
                )

    return recon_out, mask_out


def reconstruct(
    index_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    model_config: dict,
    training_config: dict | None = None,
    device: str | torch.device | None = None,
    split: str | None = None,
) -> None:
    """Reconstruct every volume in *index_path* and save as NIfTI.

    For each volume the function:

    1. Loads and z-score normalises the NIfTI data.
    2. Zero-pads to the nearest multiple of *patch_size* in every dimension.
    3. Tiles the padded volume into non-overlapping patches and runs MAE
       reconstruction on each patch.
    4. Stitches the reconstructed patches and masks back into the full padded
       volume.
    5. Trims padding to restore the original voxel dimensions.
    6. Denormalises reconstruction intensities (``recon × fg_std + fg_mean``).
       When ``training_config["loss"]`` is ``"normalized_masked_mse"``, the
       per-patch normalisation applied by the loss is first undone (step 4a)
       before the volume-level denormalisation so that saved intensities are
       in the original physical range.
    7. Saves outputs under two subdirectories of *output_dir*:
       - ``reconstructions/<volume_id>.nii.gz``
       - ``masks/<volume_id>.nii.gz`` — 1 = masked (reconstructed by model),
         0 = visible (seen by encoder). Overlay in a viewer to highlight
         the regions the model had to fill in.

    Args:
        index_path: Parquet index produced by ``misfit_index``.
        checkpoint_path: Pretrained MISFIT checkpoint (``.pt``).
        output_dir: Root output directory.  ``reconstructions/`` and
            ``masks/`` subdirectories are created automatically.
        model_config: Model configuration dict (``config["model"]`` from
            ``config.json``).
        training_config: Training configuration dict (``config["training"]``
            from ``config.json``). The ``loss`` name selects the correct
            denormalisation, and the ``amp`` flag toggles bfloat16 autocast
            (defaults to enabled when absent). When ``None``,
            volume-level-only denormalisation is applied (correct for
            ``masked_mse`` and ``masked_l1``) with autocast enabled.
        device: Torch device. Defaults to CUDA if available, else CPU.
        split: If the index contains a ``split`` column, only rows whose
            split matches this value are processed.  ``None`` processes all
            rows.
    """
    tcfg = training_config or {}
    loss_name = tcfg.get("loss", "")
    denorm_patches = loss_name == _NORMALIZED_MSE_LOSS
    use_amp = tcfg.get("amp", True)

    device = device or inference_utils.get_default_device()
    output_dir = Path(output_dir)
    recon_dir = output_dir / "reconstructions"
    mask_dir = output_dir / "masks"
    recon_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = inference_utils.load_checkpoint(checkpoint_path, device)
    patch_size = tuple(model_config["patch_size"])

    model = inference_utils.build_model_from_checkpoint(
        checkpoint, model_config, device
    )
    model_fn = model  # noqa: E731 — _tiled_reconstruct calls model_fn(tensor)

    index_df = pd.read_parquet(index_path)
    if split and "split" in index_df.columns:
        index_df = index_df[index_df["split"] == split].reset_index(drop=True)
    n_total = len(index_df)
    errors: list[str] = []

    print_section_header(
        f"Reconstructing: {n_total:,} volumes → {output_dir}"
    )

    with get_progress_bar() as progress:
        task = progress.add_task("Reconstruction", total=n_total)

        for _, row in index_df.iterrows():
            volume_id = row["volume_id"]
            recon_path = recon_dir / f"{volume_id}.nii.gz"
            mask_path = mask_dir / f"{volume_id}.nii.gz"

            volume = inference_utils.load_and_normalise(
                row["path"], row["p1"], row["p99"], row["fg_mean"], row["fg_std"]
            )
            if volume is None:
                errors.append(f"Could not load: {row['path']}")
                progress.advance(task)
                continue

            try:
                affine = np.array(json.loads(row["affine"]), dtype=np.float64)

                padded, original_shape = inference_utils.pad_to_multiple(
                    volume, patch_size
                )
                recon_padded, mask_padded = _tiled_reconstruct(
                    padded, patch_size, model_fn, device,
                    denorm_patches=denorm_patches, amp=use_amp,
                )

                d, h, w = original_shape
                recon_np = recon_padded[:d, :h, :w]
                mask_np = mask_padded[:d, :h, :w]

                # Denormalize reconstruction to original intensity space.
                fg_std = float(row["fg_std"])
                fg_mean = float(row["fg_mean"])
                recon_np = recon_np * fg_std + fg_mean

                nib.save(
                    nib.Nifti1Image(recon_np.astype(np.float32), affine=affine),
                    recon_path,
                )
                # Save mask with model convention: 1 = masked (reconstructed),
                # 0 = visible (seen by encoder). Overlay in a viewer to highlight
                # the regions the model had to reconstruct from context.
                nib.save(
                    nib.Nifti1Image(mask_np.astype(np.float32), affine=affine),
                    mask_path,
                )
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"Reconstruction failed for {volume_id}: {exc}")

            progress.advance(task)

    if errors:
        print_warning("\n".join(errors))
    print_success(
        f"Reconstructions saved for {n_total - len(errors):,}/{n_total:,} volumes."
    )
