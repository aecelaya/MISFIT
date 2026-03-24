"""High-level batch inference runners for MISFIT.

Provides two entry points that mirror MIST's ``infer_from_dataframe``:

``extract_features``
    Runs the pretrained encoder over a Parquet index and saves per-volume
    bottleneck feature maps as ``.npy`` files.  These are the representations
    you pass to downstream fine-tuning pipelines (e.g. MIST).

``reconstruct``
    Runs the full MAE forward pass and saves the reconstruction of each
    volume as a ``.nii.gz`` file in the original normalised space.  Useful
    for visualising what the model has learned.

Both runners use the :class:`~misfit.inference.predictor.Predictor` so
TTA and inferer choice are consistently configurable.
"""
from pathlib import Path
from typing import List, Optional, Union

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from misfit.inference import inference_utils
from misfit.inference.ensemblers.ensembler_registry import get_ensembler
from misfit.inference.inferers.inferer_registry import get_inferer
from misfit.inference.predictor import Predictor
from misfit.inference.tta.strategies import get_strategy
from misfit.utils.console import print_error, print_section_header, print_success, print_warning
from misfit.utils.progress_bar import get_progress_bar


def _build_predictor(
    checkpoint: dict,
    inferer_name: str,
    tta_strategy: str,
    model_fn_key: str,
    device: Union[str, torch.device],
    patch_size: tuple,
    amp: bool,
) -> Predictor:
    """Shared Predictor construction for both runners."""
    model = inference_utils.build_model_from_checkpoint(checkpoint, device)

    if model_fn_key == "reconstruction":
        model_fn = lambda x: model(x)["reconstruction"]  # noqa: E731
    elif model_fn_key == "features":
        model_fn = lambda x: model.encoder(x)[-1]  # noqa: E731
    else:
        raise ValueError(f"Unknown model_fn_key: '{model_fn_key}'")

    inferer_cls = get_inferer(inferer_name)
    if inferer_name == "whole_volume":
        inferer = inferer_cls(patch_size=patch_size, amp=amp, device=device)
    else:
        inferer = inferer_cls(patch_size=patch_size, device=device)

    ensembler = get_ensembler("mean")
    tta_transforms = get_strategy(tta_strategy)()

    return Predictor(
        model_fn=model_fn,
        inferer=inferer,
        ensembler=ensembler,
        tta_transforms=tta_transforms,
        device=device,
    )


def extract_features(
    index_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    tta_strategy: str = "none",
    inferer_name: str = "whole_volume",
    device: Optional[Union[str, torch.device]] = None,
    amp: bool = False,
) -> None:
    """Extract encoder bottleneck features for every volume in *index_path*.

    For each volume, loads and normalises the NIfTI, runs the encoder
    (optionally with TTA), and saves the bottleneck feature map as a
    ``.npy`` file under *output_dir*.

    The saved array has shape ``(C, D', H', W')`` where ``C`` is the
    bottleneck channel count and ``D'/H'/W'`` are ``patch_size / 32``.

    Args:
        index_path: Parquet index produced by ``misfit_index``.
        checkpoint_path: Pretrained MISFIT checkpoint (``.pt``).
        output_dir: Directory where ``<volume_id>.npy`` files are written.
        tta_strategy: Name of the TTA strategy to use (``"none"`` or
            ``"all_flips"``). Defaults to ``"none"``.
        inferer_name: Inferer to use (``"whole_volume"`` or
            ``"sliding_window"``). Defaults to ``"whole_volume"``.
        device: Torch device. Defaults to CUDA if available, else CPU.
        amp: Use automatic mixed precision. Defaults to False.
    """
    device = device or inference_utils.get_default_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = inference_utils.load_checkpoint(checkpoint_path, device)
    patch_size = tuple(checkpoint["args"]["patch_size"])

    predictor = _build_predictor(
        checkpoint=checkpoint,
        inferer_name=inferer_name,
        tta_strategy=tta_strategy,
        model_fn_key="features",
        device=device,
        patch_size=patch_size,
        amp=amp,
    )

    index_df = pd.read_parquet(index_path)
    n_total = len(index_df)
    errors: List[str] = []

    print_section_header(
        f"Extracting features: {n_total:,} volumes → {output_dir}"
    )

    with get_progress_bar() as progress:
        task = progress.add_task("Feature extraction", total=n_total)

        for _, row in index_df.iterrows():
            volume_id = row["volume_id"]
            out_path = output_dir / f"{volume_id}.npy"

            volume = inference_utils.load_and_normalise(
                row["path"], row["p1"], row["p99"], row["fg_mean"], row["fg_std"]
            )
            if volume is None:
                errors.append(f"Could not load: {row['path']}")
                progress.advance(task)
                continue

            volume = inference_utils.centre_crop_or_pad(volume, patch_size)
            tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()

            try:
                features = predictor(tensor)           # (1, C, D', H', W')
                np.save(out_path, features.squeeze(0).cpu().numpy())
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"Inference failed for {volume_id}: {exc}")

            progress.advance(task)

    if errors:
        print_warning("\n".join(errors))
    print_success(
        f"Features saved for {n_total - len(errors):,}/{n_total:,} volumes."
    )


def reconstruct(
    index_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    tta_strategy: str = "none",
    inferer_name: str = "whole_volume",
    device: Optional[Union[str, torch.device]] = None,
    amp: bool = False,
) -> None:
    """Reconstruct every volume in *index_path* and save as NIfTI.

    Runs the full MAE forward pass (encode masked input → decode) and
    writes the reconstruction as ``<volume_id>.nii.gz`` under *output_dir*.
    The NIfTI is in the normalised, patch-cropped space — intensities are
    z-score normalised values, not original HU / signal units.

    Args:
        index_path: Parquet index produced by ``misfit_index``.
        checkpoint_path: Pretrained MISFIT checkpoint (``.pt``).
        output_dir: Directory where ``<volume_id>.nii.gz`` files are written.
        tta_strategy: TTA strategy name. Defaults to ``"none"``.
        inferer_name: Inferer name. Defaults to ``"whole_volume"``.
        device: Torch device. Defaults to CUDA if available, else CPU.
        amp: Use automatic mixed precision. Defaults to False.
    """
    device = device or inference_utils.get_default_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = inference_utils.load_checkpoint(checkpoint_path, device)
    patch_size = tuple(checkpoint["args"]["patch_size"])

    predictor = _build_predictor(
        checkpoint=checkpoint,
        inferer_name=inferer_name,
        tta_strategy=tta_strategy,
        model_fn_key="reconstruction",
        device=device,
        patch_size=patch_size,
        amp=amp,
    )

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

            volume = inference_utils.centre_crop_or_pad(volume, patch_size)
            tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).float()

            try:
                recon = predictor(tensor)                # (1, 1, D, H, W)
                recon_np = recon.squeeze().cpu().numpy()  # (D, H, W)
                nib.save(nib.Nifti1Image(recon_np, affine=np.eye(4)), out_path)
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f"Reconstruction failed for {volume_id}: {exc}")

            progress.advance(task)

    if errors:
        print_warning("\n".join(errors))
    print_success(
        f"Reconstructions saved for {n_total - len(errors):,}/{n_total:,} volumes."
    )
