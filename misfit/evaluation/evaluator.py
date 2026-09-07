"""ReconstructionEvaluator for MISFIT pretraining quality assessment.

Loads a pretrained checkpoint, runs tiled inference over a validation index,
and computes per-volume masked reconstruction metrics (MAE, MSE, PSNR)
averaged across all non-overlapping patches.

Metrics are computed in the **training loss's space**: when the run used
``normalized_masked_mse`` the target is normalised per ``mask_patch_size`` cube
(zero mean, unit variance) — exactly as the loss does — so ``masked_mse`` is
directly comparable to ``best_val_loss``. Because a per-region-normalised MSE is
~1.0 for *any* constant predictor, every metric is also reported for a **naive
baseline** (impute masked voxels with the visible-region mean) plus a ``_skill``
column so "did pretraining beat trivial?" is a single readable number.

Unlike MIST's Evaluator — which reads pre-computed predictions from disk —
this evaluator runs the model forward pass inline, since reconstruction
quality can only be measured by running the model. Masks are drawn
deterministically from ``seed`` so runs are reproducible.

Typical usage::

    evaluator = ReconstructionEvaluator(
        checkpoint_path=Path("best_model.pt"),
        index_path=Path("val.parquet"),
        output_csv_path=Path("/runs/exp1/eval/evaluation_results.csv"),
        model_config=config["model"],
        training_config=config["training"],
    )
    evaluator.run()
"""
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import misfit.models  # noqa: F401 — trigger model registrations
from misfit.evaluation import evaluation_utils
from misfit.inference.inference_runners import _NORMALIZED_MSE_LOSS
from misfit.inference.inference_utils import get_row_spacing, pad_to_multiple
from misfit.metrics.metrics_registry import (
    DEFAULT_METRICS,
    get_metric,
)
from misfit.models.model_registry import get_model_from_registry
from misfit.utils.console import console, print_error, print_success, print_warning
from misfit.utils.hardware import autocast_context, resolve_amp
from misfit.utils.normalization import normalize_patchwise
from misfit.utils.progress_bar import get_progress_bar

# Numerically-safe floor for the naive value in the lower-is-better skill ratio.
_SKILL_EPS = 1e-12


class ReconstructionEvaluator:
    """Evaluate reconstruction quality of a pretrained MISFIT checkpoint.

    For each volume in a Parquet index the evaluator:

    1. Loads, z-score normalises, and crops to the index's foreground bounding
       box (matching ``MISFITDataset``).
    2. Zero-pads to the nearest multiple of *patch_size* in every dimension and
       tiles into non-overlapping patches.
    3. Runs a full MAE forward pass on each patch with a deterministic mask
       (``seed + tile_index``).
    4. Puts the target in the training loss's space (per ``mask_patch_size``
       cube for ``normalized_masked_mse``), then computes the requested masked
       metrics for the model and for a naive baseline (visible-region mean).
    5. Averages per-patch, one row per volume, with ``_naive``/``_skill``
       columns per metric.

    Args:
        checkpoint_path: Path to a checkpoint produced by ``MAETrainer``
            (contains a ``model`` key).
        index_path: Path to the Parquet index built by ``misfit_index``.
        output_csv_path: Path where the evaluation results CSV will be written.
            Parent directory is created automatically if it does not exist.
        model_config: Model configuration dict (``config["model"]`` from
            ``config.json``).  Must contain ``architecture``, ``patch_size``,
            ``mask_patch_size``, and ``mask_ratio``.
        metrics: List of metric names to compute. Defaults to
            ``DEFAULT_METRICS`` (``masked_mae``, ``masked_mse``, ``masked_psnr``).
        device: Torch device string (e.g. ``"cuda:0"``). Defaults to
            ``"cuda"`` if available, else ``"cpu"``.
        split: If the index has a ``split`` column, only rows matching this
            value are evaluated. Defaults to ``"val"``.
        training_config: Training configuration dict (``config["training"]``
            from ``config.json``). The ``loss`` name selects the metric space
            (per-``mask_patch_size``-cube normalisation for
            ``normalized_masked_mse``, whole-volume z-score otherwise) and the
            ``amp`` flag toggles autocast (defaults to enabled when absent, then
            resolved against the current hardware — BF16 needs an NVIDIA
            Ampere+ or AMD CDNA/RDNA3+ GPU, otherwise it falls back to FP32).
        seed: Base RNG seed. Each patch's mask is drawn from ``seed +
            patch_index`` so the whole evaluation is reproducible.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        index_path: Path,
        output_csv_path: Path,
        model_config: dict,
        metrics: list[str] | None = None,
        device: str | None = None,
        split: str | None = "val",
        training_config: dict | None = None,
        seed: int = 42,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.index_path = Path(index_path)
        self.output_csv_path = Path(output_csv_path)
        self.model_config = model_config
        self.metrics = list(metrics) if metrics else list(DEFAULT_METRICS)
        self.seed = seed
        self.mask_patch_size = int(model_config["mask_patch_size"])
        tcfg = training_config or {}
        # Resolve the config's AMP request against this machine's hardware —
        # evaluation may run on a different GPU (or CPU) than training did.
        self.amp = resolve_amp(tcfg.get("amp", True))
        loss_name = tcfg.get("loss", "")
        self.normalize_target_patches = loss_name == _NORMALIZED_MSE_LOSS

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
        self.patch_size: tuple[int, int, int] = tuple(model_config["patch_size"])

        # Voxel spacing for the volume currently being evaluated. Set per-volume
        # in run() so the decoder is conditioned exactly as it was in training.
        self._current_spacing: torch.Tensor | None = None

        self.model: nn.Module = self._build_model()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Reconstruct the model from model config and load weights."""
        cfg = self.model_config
        model = get_model_from_registry(
            cfg["architecture"],
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
    ) -> np.ndarray | None:
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
        return self._crop_to_foreground(data, row)

    @staticmethod
    def _crop_to_foreground(volume: np.ndarray, row: pd.Series) -> np.ndarray:
        """Crop to the index's foreground bounding box.

        ``MISFITDataset`` crops training patches to this box before sampling, so
        the evaluator does too — otherwise the tiled patches would include air
        and zero-padding the model was never trained on, diluting every metric.
        No-op when the index predates the ``fg_*`` columns.
        """
        cols = ("fg_x_start", "fg_x_end", "fg_y_start", "fg_y_end",
                "fg_z_start", "fg_z_end")
        if not all(c in row for c in cols):
            return volume
        x0, x1 = int(row["fg_x_start"]), int(row["fg_x_end"]) + 1
        y0, y1 = int(row["fg_y_start"]), int(row["fg_y_end"]) + 1
        z0, z1 = int(row["fg_z_start"]), int(row["fg_z_end"]) + 1
        return volume[x0:x1, y0:y1, z0:z1]

    def _run_inference(
        self, patch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one forward pass on a single patch and return (reconstruction, mask).

        Args:
            patch: Normalised patch of shape ``patch_size``.

        Returns:
            Tuple of ``(reconstruction, mask)``, each shape ``patch_size``.
        """
        tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        tensor = tensor.to(self.device)

        amp_ctx = autocast_context(self.amp)
        with torch.no_grad(), amp_ctx:
            output = self.model(tensor, spacing=self._current_spacing)

        recon = output["reconstruction"].squeeze().cpu().numpy()  # (D,H,W)
        mask = output["mask"].squeeze().cpu().numpy()              # (D,H,W)
        return recon, mask

    def _run_tiled_inference(
        self, volume: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Tile *volume* into non-overlapping patches and run inference on each.

        Pads the volume to the nearest multiple of ``patch_size`` in each
        dimension, then iterates over the resulting grid of non-overlapping
        patches.  Each patch is passed through the full MAE forward pass to
        obtain its reconstruction and mask.

        When the model was trained with ``normalized_masked_mse`` the target
        patch is normalised **per ``mask_patch_size`` cube** (via
        :func:`misfit.utils.normalization.normalize_patchwise`) — the exact
        transform the loss applies — so ``masked_mse`` is measured in the space
        the model was optimised in. Other losses keep the whole-volume z-score
        target.

        The mask for tile *i* is drawn from ``seed + i`` so results are
        reproducible.

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
        patch_index = 0
        for di in range(0, D, pd_):
            for hi in range(0, H, ph_):
                for wi in range(0, W, pw_):
                    patch = padded[di:di + pd_, hi:hi + ph_, wi:wi + pw_]
                    # Deterministic mask per tile → reproducible evaluation.
                    torch.manual_seed(self.seed + patch_index)
                    patch_index += 1
                    recon, mask = self._run_inference(patch)
                    if self.normalize_target_patches:
                        patch = normalize_patchwise(
                            torch.from_numpy(np.ascontiguousarray(patch)),
                            self.mask_patch_size,
                        ).numpy()
                    results.append((recon, patch, mask))
        return results

    def _compute_metrics(
        self,
        reconstruction: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> dict[str, float]:
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

    @staticmethod
    def _naive_prediction(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Impute the masked region with the mean of the visible voxels.

        The canonical no-information inpainting baseline: any metric the model
        cannot beat here means pretraining learned nothing useful for
        reconstruction. Falls back to the whole-patch mean if the patch happens
        to have no visible voxels.
        """
        visible = target[mask == 0]
        fill = float(visible.mean()) if visible.size else float(target.mean())
        return np.full_like(target, fill)

    def _expanded_columns(self) -> list[str]:
        """``[m, m_naive, m_skill, ...]`` for every requested metric."""
        return [
            col
            for m in self.metrics
            for col in (m, f"{m}_naive", f"{m}_skill")
        ]

    @staticmethod
    def _skill(metric_name: str, model_value: float, naive_value: float) -> float:
        """Signed improvement of the model over the naive baseline.

        Positive = beats trivial, 0 = tied, negative = worse. For
        lower-is-better metrics (MAE, MSE) this is the fraction of the trivial
        error removed (``1 - model/naive``); for higher-is-better metrics
        (PSNR, SSIM) it is the additive gain (``model - naive``, e.g. dB for
        PSNR).
        """
        metric = get_metric(metric_name)
        lower_is_better = metric.best < metric.worst
        if lower_is_better:
            if abs(naive_value) < _SKILL_EPS:
                return float("nan")
            return 1.0 - model_value / naive_value
        return model_value - naive_value

    def _print_summary(self, rows: list[dict]) -> None:
        """Print a per-metric ``model | naive | skill`` summary + a verdict.

        Called only with a non-empty *rows* (guarded by the caller).
        """
        console.print("\n[bold]Summary[/bold] (masked voxels, loss space)")
        for m in self.metrics:
            model_mean = float(np.nanmean([r[m] for r in rows]))
            naive_mean = float(np.nanmean([r[f"{m}_naive"] for r in rows]))
            skill_mean = float(np.nanmean([r[f"{m}_skill"] for r in rows]))
            lower_is_better = get_metric(m).best < get_metric(m).worst
            if lower_is_better:
                verdict = (
                    f"beats trivial ({skill_mean:+.1%} of its error removed)"
                    if skill_mean > 0.005
                    else "tied with trivial"
                    if skill_mean > -0.005
                    else f"worse than trivial ({skill_mean:+.1%})"
                )
            else:
                verdict = (
                    f"+{skill_mean:.3f} over trivial"
                    if skill_mean > 0
                    else f"{skill_mean:.3f} vs trivial"
                )
            console.print(
                f"  {m:<14} model={model_mean:.4f}  "
                f"naive={naive_mean:.4f}  {verdict}"
            )

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame:
        """Evaluate all volumes in the index and write results to CSV.

        For each volume, masked metrics are computed per patch (for the model
        and for the naive baseline) and averaged across all patches to produce a
        single per-volume score. The CSV has three columns per metric —
        ``<metric>``, ``<metric>_naive``, ``<metric>_skill`` — plus the usual
        summary rows.

        Returns:
            DataFrame with per-volume metric values and summary statistics.
        """
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df = evaluation_utils.initialize_results_dataframe(
            self._expanded_columns()
        )

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

                # Condition the decoder on this volume's voxel spacing, exactly
                # as during training (no-op for indexes without spacing columns).
                self._current_spacing = get_row_spacing(row, self.device)

                # 2. Tile + inference (per-patch masked forward pass).
                try:
                    patch_results = self._run_tiled_inference(volume)
                except Exception as exc:  # pylint: disable=broad-except
                    print_warning(f"Inference failed for {volume_id}: {exc}")
                    n_errors += 1
                    progress.advance(task)
                    continue

                # 3. Per-patch metrics for the model and the naive baseline,
                #    averaged across all patches.
                model_per_patch = []
                naive_per_patch = []
                for recon, target, mask in patch_results:
                    model_per_patch.append(
                        self._compute_metrics(recon, target, mask)
                    )
                    naive_per_patch.append(
                        self._compute_metrics(
                            self._naive_prediction(target, mask), target, mask
                        )
                    )

                row = {"volume_id": volume_id}
                for m in self.metrics:
                    model_avg = float(np.mean([p[m] for p in model_per_patch]))
                    naive_avg = float(np.mean([p[m] for p in naive_per_patch]))
                    row[m] = model_avg
                    row[f"{m}_naive"] = naive_avg
                    row[f"{m}_skill"] = self._skill(m, model_avg, naive_avg)
                rows.append(row)

                progress.advance(task)

        # Build DataFrame and append summary stats.
        if rows:
            self._print_summary(rows)
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
