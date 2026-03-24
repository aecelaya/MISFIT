"""ReconstructionEvaluator for MISFIT pretraining quality assessment.

Loads a pretrained checkpoint, runs inference over a validation index, and
computes per-volume reconstruction quality metrics (SSIM, PSNR, MAE, MSE).

Unlike MIST's Evaluator — which reads pre-computed predictions from disk —
this evaluator runs the model forward pass inline, since reconstruction
quality can only be measured by running the model.

Typical usage::

    evaluator = ReconstructionEvaluator(
        checkpoint_path=Path("best_model.pt"),
        index_path=Path("val.parquet"),
        results_dir=Path("/runs/exp1/eval"),
        metrics=["masked_mae", "masked_psnr", "ssim"],
    )
    evaluator.run()
"""
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import misfit.models  # noqa: F401 — trigger model registrations
from misfit.evaluation import evaluation_utils
from misfit.metrics.metrics_registry import get_metric, list_registered_metrics
from misfit.models.model_registry import get_model_from_registry
from misfit.utils.console import console, print_error, print_success, print_warning
from misfit.utils.progress_bar import get_progress_bar


class ReconstructionEvaluator:
    """Evaluate reconstruction quality of a pretrained MISFIT checkpoint.

    Loads each volume from a Parquet index, applies clip + z-score
    normalisation using the precomputed per-volume statistics, runs a
    single forward pass through the model, and computes the requested
    metrics on the masked patches.

    Args:
        checkpoint_path: Path to a checkpoint produced by ``MAETrainer``
            (contains ``model``, ``args``, and optional ``scaler`` keys).
        index_path: Path to the Parquet index built by ``misfit_index``.
        results_dir: Directory where ``evaluation_results.csv`` is written.
        metrics: List of metric names to compute. Defaults to all registered
            metrics.
        device: Torch device string (e.g. ``"cuda:0"``). Defaults to
            ``"cuda"`` if available, else ``"cpu"``.
        amp: Use automatic mixed precision for inference. Defaults to False.
        patch_size: Spatial crop size (D, H, W) applied before inference.
            If None, read from the checkpoint's saved args. Defaults to None.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        index_path: Path,
        results_dir: Path,
        metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
        amp: bool = False,
        patch_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.index_path = Path(index_path)
        self.results_dir = Path(results_dir)
        self.metrics = metrics or list_registered_metrics()
        self.amp = amp

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load index.
        self.index_df = pd.read_parquet(self.index_path)

        # Load checkpoint and build model.
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.saved_args: Dict = self.checkpoint["args"]
        self.patch_size: Tuple[int, int, int] = (
            patch_size
            if patch_size is not None
            else tuple(self.saved_args["patch_size"])
        )

        self.model: nn.Module = self._build_model()
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Reconstruct the model from checkpoint args and load weights."""
        args = self.saved_args
        model = get_model_from_registry(
            args["model"],
            in_channels=args["in_channels"],
            img_size=tuple(args["patch_size"]),
            mask_patch_size=args["mask_patch_size"],
            mask_ratio=args["mask_ratio"],
        )
        model.load_state_dict(self.checkpoint["model"])
        model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Per-volume helpers
    # ------------------------------------------------------------------

    def _load_and_normalise(
        self, row: pd.Series
    ) -> Optional[np.ndarray]:
        """Load a NIfTI volume and apply clip + z-score normalisation.

        Returns None (and logs a warning) if the file cannot be read.
        """
        try:
            img = nib.load(row["path"])
            data = np.asarray(img.dataobj, dtype=np.float32)
        except Exception as exc:  # pylint: disable=broad-except
            print_warning(f"Could not load {row['path']}: {exc}")
            return None

        if data.ndim == 4:
            data = data[..., 0]

        p1, p99 = float(row["p1"]), float(row["p99"])
        fg_mean, fg_std = float(row["fg_mean"]), float(row["fg_std"])

        data = np.clip(data, p1, p99)
        data = (data - fg_mean) / max(fg_std, 1e-8)
        return data

    def _pad_or_crop(self, volume: np.ndarray) -> np.ndarray:
        """Centre-crop (or pad then crop) the volume to ``self.patch_size``."""
        target = self.patch_size
        result = volume

        # Pad any axis that is smaller than the target.
        pad_width = []
        for dim_size, t in zip(result.shape, target):
            deficit = max(0, t - dim_size)
            pad_before = deficit // 2
            pad_after = deficit - pad_before
            pad_width.append((pad_before, pad_after))
        if any(p[0] + p[1] > 0 for p in pad_width):
            result = np.pad(result, pad_width, mode="constant", constant_values=0)

        # Centre crop.
        slices = []
        for dim_size, t in zip(result.shape, target):
            start = (dim_size - t) // 2
            slices.append(slice(start, start + t))
        return result[tuple(slices)]

    def _run_inference(
        self, volume: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one forward pass and return (reconstruction, mask) as numpy.

        Args:
            volume: Normalised, cropped volume of shape (D, H, W).

        Returns:
            Tuple of (reconstruction, mask), each shape (D, H, W).
        """
        tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        tensor = tensor.to(self.device)

        amp_ctx = torch.amp.autocast("cuda") if self.amp else nullcontext()
        with torch.no_grad(), amp_ctx:
            output = self.model(tensor)

        recon = output["reconstruction"].squeeze().cpu().numpy()   # (D,H,W)
        mask  = output["mask"].squeeze().cpu().numpy()             # (D,H,W)
        return recon, mask

    def _compute_metrics(
        self,
        reconstruction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all requested metrics for one volume."""
        results = {}
        for metric_name in self.metrics:
            metric = get_metric(metric_name)
            try:
                val = metric(reconstruction, target, mask)
                results[metric_name] = val if np.isfinite(val) else metric.worst
            except Exception as exc:  # pylint: disable=broad-except
                print_warning(f"Metric '{metric_name}' failed: {exc}")
                results[metric_name] = metric.worst
        return results

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Evaluate all volumes in the index and write results to CSV.

        Returns:
            DataFrame with per-volume metric values and summary statistics.
        """
        output_csv = self.results_dir / "evaluation_results.csv"
        results_df = evaluation_utils.initialize_results_dataframe(self.metrics)

        n_total = len(self.index_df)
        n_errors = 0
        rows = []

        console.print(
            f"\n[bold]Evaluating {n_total:,} volumes "
            f"on {self.device} ...[/bold]\n"
        )

        with get_progress_bar() as progress:
            task = progress.add_task("Reconstruction eval", total=n_total)

            for _, row in self.index_df.iterrows():
                volume_id = row["volume_id"]

                # 1. Load + normalise.
                volume = self._load_and_normalise(row)
                if volume is None:
                    n_errors += 1
                    progress.advance(task)
                    continue

                # 2. Crop / pad to patch size.
                volume = self._pad_or_crop(volume)

                # 3. Inference.
                try:
                    reconstruction, mask = self._run_inference(volume)
                except Exception as exc:  # pylint: disable=broad-except
                    print_warning(f"Inference failed for {volume_id}: {exc}")
                    n_errors += 1
                    progress.advance(task)
                    continue

                # 4. Metrics.
                metric_values = self._compute_metrics(reconstruction, volume, mask)
                rows.append({"volume_id": volume_id, **metric_values})

                progress.advance(task)

        # Build DataFrame and append summary stats.
        if rows:
            results_df = pd.DataFrame(rows, columns=results_df.columns)
            results_df = evaluation_utils.compute_results_stats(results_df)
        else:
            print_error("No volumes were successfully evaluated.")

        results_df.to_csv(output_csv, index=False)

        n_ok = n_total - n_errors
        if n_errors:
            print_warning(f"{n_errors:,} volume(s) failed and were skipped.")
        print_success(
            f"Evaluated {n_ok:,}/{n_total:,} volumes. "
            f"Results saved to {output_csv}"
        )

        return results_df
