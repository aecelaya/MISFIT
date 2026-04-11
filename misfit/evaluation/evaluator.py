"""ReconstructionEvaluator for MISFIT pretraining quality assessment.

Loads a pretrained checkpoint, runs tiled inference over a validation index,
and computes per-volume reconstruction quality metrics (SSIM, PSNR, MAE, MSE)
averaged across all non-overlapping patches.

Unlike MIST's Evaluator — which reads pre-computed predictions from disk —
this evaluator runs the model forward pass inline, since reconstruction
quality can only be measured by running the model.

Typical usage::

    evaluator = ReconstructionEvaluator(
        checkpoint_path=Path("best_model.pt"),
        index_path=Path("val.parquet"),
        output_csv_path=Path("/runs/exp1/eval/evaluation_results.csv"),
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
from misfit.inference.inference_utils import pad_to_multiple
from misfit.metrics.metrics_registry import get_metric, list_registered_metrics
from misfit.models.model_registry import get_model_from_registry
from misfit.utils.console import console, print_error, print_success, print_warning
from misfit.utils.progress_bar import get_progress_bar


class ReconstructionEvaluator:
    """Evaluate reconstruction quality of a pretrained MISFIT checkpoint.

    For each volume in a Parquet index the evaluator:

    1. Loads and z-score normalises the NIfTI data.
    2. Zero-pads to the nearest multiple of *patch_size* in every dimension.
    3. Tiles the padded volume into non-overlapping patches and runs a full
       MAE forward pass on each patch.
    4. Computes the requested masked metrics on each patch independently
       (comparing reconstruction against the unmasked original on the masked
       positions only — the actual MAE pretraining objective).
    5. Averages the per-patch metrics to produce one row per volume.

    Args:
        checkpoint_path: Path to a checkpoint produced by ``MAETrainer``
            (contains ``model`` and optional ``scaler`` keys).
        index_path: Path to the Parquet index built by ``misfit_index``.
        output_csv_path: Path where the evaluation results CSV will be written.
            Parent directory is created automatically if it does not exist.
        model_config: Model configuration dict (``config["model"]`` from
            ``config.json``).  Must contain ``name``, ``patch_size``,
            ``mask_patch_size``, and ``mask_ratio``.
        metrics: List of metric names to compute. Defaults to all registered
            metrics.
        device: Torch device string (e.g. ``"cuda:0"``). Defaults to
            ``"cuda"`` if available, else ``"cpu"``.
        amp: Use automatic mixed precision for inference. Always True.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        index_path: Path,
        output_csv_path: Path,
        model_config: Dict,
        metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
        split: Optional[str] = "val",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.index_path = Path(index_path)
        self.output_csv_path = Path(output_csv_path)
        self.model_config = model_config
        self.metrics = metrics or list_registered_metrics()
        self.amp = True

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load index and optionally filter to a single split.
        self.index_df = pd.read_parquet(self.index_path)
        if split and "split" in self.index_df.columns:
            self.index_df = (
                self.index_df[self.index_df["split"] == split].reset_index(drop=True)
            )

        # Load checkpoint and build model.
        self.checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.patch_size: Tuple[int, int, int] = tuple(model_config["patch_size"])

        self.model: nn.Module = self._build_model()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Reconstruct the model from model config and load weights."""
        cfg = self.model_config
        model = get_model_from_registry(
            cfg["name"],
            in_channels=1,
            img_size=tuple(cfg["patch_size"]),
            mask_patch_size=cfg["mask_patch_size"],
            mask_ratio=cfg["mask_ratio"],
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

    def _run_inference(
        self, patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run one forward pass on a single patch and return (reconstruction, mask).

        Args:
            patch: Normalised patch of shape ``patch_size``.

        Returns:
            Tuple of ``(reconstruction, mask)``, each shape ``patch_size``.
        """
        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        tensor = tensor.to(self.device)

        amp_ctx = torch.amp.autocast("cuda") if self.amp else nullcontext()
        with torch.no_grad(), amp_ctx:
            output = self.model(tensor)

        recon = output["reconstruction"].squeeze().cpu().numpy()   # (D,H,W)
        mask  = output["mask"].squeeze().cpu().numpy()             # (D,H,W)
        return recon, mask

    def _run_tiled_inference(
        self, volume: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Tile *volume* into non-overlapping patches and run inference on each.

        Pads the volume to the nearest multiple of ``patch_size`` in each
        dimension, then iterates over the resulting grid of non-overlapping
        patches.  Each patch is passed through the full MAE forward pass to
        obtain its reconstruction and mask.

        Args:
            volume: Normalised volume of shape (D, H, W).

        Returns:
            List of ``(reconstruction, target, mask)`` tuples — one per patch
            — each of shape ``patch_size``.
        """
        padded, _ = pad_to_multiple(volume, self.patch_size)
        pd_, ph_, pw_ = self.patch_size
        D, H, W = padded.shape
        results = []
        for di in range(0, D, pd_):
            for hi in range(0, H, ph_):
                for wi in range(0, W, pw_):
                    patch = padded[di:di + pd_, hi:hi + ph_, wi:wi + pw_]
                    recon, mask = self._run_inference(patch)
                    results.append((recon, patch, mask))
        return results

    def _compute_metrics(
        self,
        reconstruction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> Dict[str, float]:
        """Compute all requested metrics for one patch."""
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

        For each volume, masked metrics are computed per patch and averaged
        across all patches to produce a single per-volume score.

        Returns:
            DataFrame with per-volume metric values and summary statistics.
        """
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
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

                # 2. Tile + inference (per-patch masked forward pass).
                try:
                    patch_results = self._run_tiled_inference(volume)
                except Exception as exc:  # pylint: disable=broad-except
                    print_warning(f"Inference failed for {volume_id}: {exc}")
                    n_errors += 1
                    progress.advance(task)
                    continue

                # 3. Per-patch masked metrics, averaged across all patches.
                all_patch_metrics = [
                    self._compute_metrics(recon, target, mask)
                    for recon, target, mask in patch_results
                ]
                metric_values = {
                    k: float(np.mean([m[k] for m in all_patch_metrics]))
                    for k in self.metrics
                }
                rows.append({"volume_id": volume_id, **metric_values})

                progress.advance(task)

        # Build DataFrame and append summary stats.
        if rows:
            results_df = pd.DataFrame(rows, columns=results_df.columns)
            results_df = evaluation_utils.compute_results_stats(results_df)
        else:
            print_error("No volumes were successfully evaluated.")

        results_df.to_csv(self.output_csv_path, index=False)

        n_ok = n_total - n_errors
        if n_errors:
            print_warning(f"{n_errors:,} volume(s) failed and were skipped.")
        print_success(
            f"Evaluated {n_ok:,}/{n_total:,} volumes. "
            f"Results saved to {self.output_csv_path}"
        )

        return results_df
