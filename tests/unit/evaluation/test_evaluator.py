"""Tests for misfit.evaluation.evaluator.ReconstructionEvaluator."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

import misfit.models  # noqa — trigger registrations
from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

MODEL_CONFIG = {
    "architecture": "_tiny_test_model",
    "patch_size": [32, 32, 32],
    "mask_patch_size": 16,
    "mask_ratio": 0.75,
}


def _make_tiny_checkpoint(tmp_path: Path, patch_size=(32, 32, 32)) -> Path:
    """Build and save a tiny SwinMAE checkpoint for tests (feature_size=12)."""
    model = SwinMAE(
        in_channels=1,
        feature_size=12,
        img_size=patch_size,
        mask_patch_size=16,
        mask_ratio=0.75,
    )
    ckpt_path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    return ckpt_path


def _build_tiny_model(checkpoint, model_config, device):
    """Rebuild the tiny SwinMAE from a checkpoint saved by _make_tiny_checkpoint."""
    model = SwinMAE(
        in_channels=1,
        feature_size=12,
        img_size=tuple(model_config["patch_size"]),
        mask_patch_size=model_config["mask_patch_size"],
        mask_ratio=model_config["mask_ratio"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def _make_index(tmp_path: Path, nifti_path: Path) -> Path:
    df = pd.DataFrame([{
        "volume_id": "vol",
        "path": str(nifti_path),
        "shape_d": 32, "shape_h": 32, "shape_w": 32,
        "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
        "affine": json.dumps(np.eye(4).tolist()),
        "fg_x_start": 0, "fg_x_end": 31,
        "fg_y_start": 0, "fg_y_end": 31,
        "fg_z_start": 0, "fg_z_end": 31,
        "p1": -2.0, "p99": 2.0,
        "fg_mean": 0.0, "fg_std": 1.0,
    }])
    p = tmp_path / "index.parquet"
    df.to_parquet(p, index=False)
    return p


@pytest.fixture
def evaluator_setup(tmp_path):
    """Set up a tiny evaluator with real files on disk."""
    data = np.random.randn(32, 32, 32).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nifti_path = tmp_path / "vol.nii.gz"
    nib.save(img, str(nifti_path))

    index_path = _make_index(tmp_path, nifti_path)
    ckpt_path = _make_tiny_checkpoint(tmp_path)
    output_csv = tmp_path / "results" / "evaluation_results.csv"

    return ckpt_path, index_path, output_csv


def _make_evaluator(ckpt, idx, res, metrics=None):
    """Create a ReconstructionEvaluator with _build_model patched to use the tiny checkpoint."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator
    # We need to patch _build_model so it uses our tiny feature_size=12 model.
    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt,
            index_path=idx,
            output_csv_path=res,
            model_config=MODEL_CONFIG,
            metrics=metrics or ["masked_mae"],
            device="cpu",
        )
    return ev


def test_evaluator_init(evaluator_setup):
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res)
    assert ev.device == torch.device("cpu")


def test_evaluator_run_produces_csv(evaluator_setup):
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res)
    df = ev.run()
    assert res.exists()
    assert isinstance(df, pd.DataFrame)


def test_evaluator_load_and_normalise_missing_file(evaluator_setup):
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res)
    row = pd.Series({"path": "/nonexistent/file.nii.gz", "p1": -1, "p99": 1,
                     "fg_mean": 0, "fg_std": 1})
    result = ev._load_and_normalise(row)
    assert result is None


def test_evaluator_patch_size_from_model_config(tmp_path):
    """patch_size is read from model_config."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    data = np.random.randn(32, 32, 32).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nifti_path = tmp_path / "vol.nii.gz"
    nib.save(img, str(nifti_path))

    idx = _make_index(tmp_path, nifti_path)
    ckpt = _make_tiny_checkpoint(tmp_path)
    res = tmp_path / "res"

    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt, index_path=idx, output_csv_path=res,
            model_config=MODEL_CONFIG, metrics=["masked_mae"], device="cpu",
        )
    assert ev.patch_size == (32, 32, 32)


def test_evaluator_device_none_auto_detects(evaluator_setup):
    """device=None should auto-detect (resolves to cpu in CI)."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    ckpt, idx, res = evaluator_setup
    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt, index_path=idx, output_csv_path=res,
            model_config=MODEL_CONFIG, metrics=["masked_mae"], device=None,
        )
    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert ev.device == expected


def test_build_model_loads_weights(evaluator_setup):
    """_build_model() is called without patching — exercises lines 106-117."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    ckpt, idx, res = evaluator_setup
    tiny_model = SwinMAE(
        in_channels=1, feature_size=12, img_size=(32, 32, 32),
        mask_patch_size=16, mask_ratio=0.75,
    )
    # Patch get_model_from_registry so _build_model runs but doesn't hit the
    # real registry (which would need feature_size=24 for swinunetr-small).
    with patch(
        "misfit.evaluation.evaluator.get_model_from_registry",
        return_value=tiny_model,
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt, index_path=idx, output_csv_path=res,
            model_config=MODEL_CONFIG, metrics=["masked_mae"], device="cpu",
        )
    assert ev.model is not None
    assert ev.model.training is False   # eval() was called


def test_load_and_normalise_4d_nifti(evaluator_setup, tmp_path):
    """4-D NIfTI (e.g. time series) should be squeezed to 3-D."""
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res)

    data_4d = np.random.randn(16, 16, 16, 1).astype(np.float32)
    nifti_4d = tmp_path / "vol4d.nii.gz"
    nib.save(nib.Nifti1Image(data_4d, np.eye(4)), str(nifti_4d))

    row = pd.Series({
        "path": str(nifti_4d),
        "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0,
    })
    result = ev._load_and_normalise(row)
    assert result is not None
    assert result.ndim == 3


def test_compute_metrics_exception_uses_worst(evaluator_setup):
    """When a metric raises, _compute_metrics falls back to metric.worst."""
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res, metrics=["masked_mae"])

    recon = np.random.randn(8, 8, 8).astype(np.float32)
    target = np.random.randn(8, 8, 8).astype(np.float32)
    mask = np.ones((8, 8, 8), dtype=np.float32)

    with patch(
        "misfit.evaluation.evaluator.get_metric",
        return_value=MagicMock(
            side_effect=RuntimeError("metric boom"),
            worst=float("inf"),
        ),
    ):
        result = ev._compute_metrics(recon, target, mask)

    assert result["masked_mae"] == float("inf")


def test_run_tiled_inference_patch_count(evaluator_setup):
    """A 64³ volume with 32³ patches triggers exactly 8 inference calls."""
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res)

    call_count = {"n": 0}
    orig_run = ev._run_inference

    def counting_run_inference(patch):
        call_count["n"] += 1
        return orig_run(patch)

    vol = np.zeros((64, 64, 64), dtype=np.float32)
    with patch.object(ev, "_run_inference", side_effect=counting_run_inference):
        results = ev._run_tiled_inference(vol)

    assert call_count["n"] == 8
    assert len(results) == 8


def test_run_tiled_inference_single_patch(evaluator_setup):
    """A volume exactly equal to patch_size triggers exactly 1 inference call."""
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res)

    call_count = {"n": 0}
    orig_run = ev._run_inference

    def counting_run(patch):
        call_count["n"] += 1
        return orig_run(patch)

    vol = np.zeros((32, 32, 32), dtype=np.float32)
    with patch.object(ev, "_run_inference", side_effect=counting_run):
        results = ev._run_tiled_inference(vol)

    assert call_count["n"] == 1
    assert len(results) == 1


def test_run_skips_volume_on_load_failure(evaluator_setup):
    """When _load_and_normalise returns None the volume is skipped."""
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res)

    with patch.object(ev, "_load_and_normalise", return_value=None):
        ev.run()

    # CSV exists but contains only summary rows (no per-volume data row)
    assert res.exists()


def test_run_skips_volume_on_inference_failure(evaluator_setup):
    """When _run_tiled_inference raises the volume is skipped."""
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res)

    with patch.object(
        ev, "_run_tiled_inference", side_effect=RuntimeError("inference boom")
    ):
        ev.run()  # must not propagate the exception

    assert res.exists()


def test_run_all_volumes_fail_prints_error(evaluator_setup):
    """When every volume fails rows is empty → print_error path."""
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res)

    with patch.object(ev, "_load_and_normalise", return_value=None), \
         patch("misfit.evaluation.evaluator.print_error") as mock_err:
        ev.run()

    mock_err.assert_called_once()
    assert "No volumes" in mock_err.call_args[0][0]


def test_split_filtering_keeps_only_matching_rows(tmp_path):
    """When split='val' the evaluator should drop non-val rows from index_df."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    data = np.random.randn(32, 32, 32).astype(np.float32)
    nifti_path = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(nifti_path))

    base_row = {
        "path": str(nifti_path),
        "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0,
    }
    df = pd.DataFrame([
        {"volume_id": "train_vol", "split": "train", **base_row},
        {"volume_id": "val_vol",   "split": "val",   **base_row},
        {"volume_id": "test_vol",  "split": "test",  **base_row},
    ])
    idx = tmp_path / "split_index.parquet"
    df.to_parquet(idx, index=False)
    ckpt = _make_tiny_checkpoint(tmp_path)
    res = tmp_path / "res.csv"

    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt, index_path=idx, output_csv_path=res,
            model_config=MODEL_CONFIG, metrics=["masked_mae"], device="cpu",
            split="val",
        )

    assert len(ev.index_df) == 1
    assert ev.index_df.iloc[0]["volume_id"] == "val_vol"


def test_split_none_keeps_all_rows(tmp_path):
    """When split=None the evaluator should keep all rows regardless of split column."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    data = np.random.randn(32, 32, 32).astype(np.float32)
    nifti_path = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(nifti_path))

    base_row = {
        "path": str(nifti_path),
        "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0,
    }
    df = pd.DataFrame([
        {"volume_id": "train_vol", "split": "train", **base_row},
        {"volume_id": "val_vol",   "split": "val",   **base_row},
    ])
    idx = tmp_path / "all_index.parquet"
    df.to_parquet(idx, index=False)
    ckpt = _make_tiny_checkpoint(tmp_path)
    res = tmp_path / "res.csv"

    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt, index_path=idx, output_csv_path=res,
            model_config=MODEL_CONFIG, metrics=["masked_mae"], device="cpu",
            split=None,
        )

    assert len(ev.index_df) == 2


# ---------------------------------------------------------------------------
# training_config / normalized_masked_mse target-patch normalisation
# ---------------------------------------------------------------------------

def test_normalized_mse_evaluator_normalizes_target_patches(evaluator_setup):
    """With training_config={"loss": "normalized_masked_mse"}, the target patches
    returned by _run_tiled_inference should be approximately zero-mean, unit-std."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    ckpt, idx, res = evaluator_setup

    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt,
            index_path=idx,
            output_csv_path=res,
            model_config=MODEL_CONFIG,
            metrics=["masked_mae"],
            device="cpu",
            training_config={"loss": "normalized_masked_mse"},
        )

    assert ev.normalize_target_patches is True

    # Feed a volume with a known, non-trivial distribution.
    rng = np.random.default_rng(0)
    vol = rng.normal(loc=5.0, scale=3.0, size=(32, 32, 32)).astype(np.float32)
    patch_results = ev._run_tiled_inference(vol)

    # There is only one patch (32^3 volume, 32^3 patch_size).
    assert len(patch_results) == 1
    _, target_patch, _ = patch_results[0]

    # The normalised target should be ~zero-mean, ~unit-std.
    assert abs(target_patch.mean()) < 0.1
    assert abs(target_patch.std() - 1.0) < 0.1


def test_non_normalized_mse_evaluator_keeps_target_unchanged(evaluator_setup):
    """Without normalized_masked_mse, target patches must not be altered."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    ckpt, idx, res = evaluator_setup

    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt,
            index_path=idx,
            output_csv_path=res,
            model_config=MODEL_CONFIG,
            metrics=["masked_mae"],
            device="cpu",
            training_config={"loss": "masked_mse"},
        )

    assert ev.normalize_target_patches is False

    rng = np.random.default_rng(1)
    vol = rng.normal(loc=5.0, scale=3.0, size=(32, 32, 32)).astype(np.float32)
    padded_vol = vol.copy()  # pad_to_multiple is a no-op here (already 32^3)

    patch_results = ev._run_tiled_inference(vol)
    assert len(patch_results) == 1
    _, target_patch, _ = patch_results[0]

    # Target should be the raw padded slice, not normalised.
    np.testing.assert_allclose(target_patch, padded_vol, rtol=1e-5)


def test_no_training_config_defaults_normalize_target_false(evaluator_setup):
    """training_config=None should default normalize_target_patches to False."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    ckpt, idx, res = evaluator_setup

    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt,
            index_path=idx,
            output_csv_path=res,
            model_config=MODEL_CONFIG,
            metrics=["masked_mae"],
            device="cpu",
            training_config=None,
        )

    assert ev.normalize_target_patches is False
    # No training_config → AMP requested by default, then resolved against the
    # current hardware (False on a CPU / pre-Ampere test machine).
    from misfit.utils.hardware import bf16_supported
    assert ev.amp is bf16_supported()


def test_amp_flag_read_from_training_config(evaluator_setup):
    """training_config={"amp": False} always disables AMP; a request is resolved
    against the current hardware."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator
    from misfit.utils.hardware import bf16_supported

    ckpt, idx, res = evaluator_setup

    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev_off = ReconstructionEvaluator(
            checkpoint_path=ckpt, index_path=idx, output_csv_path=res,
            model_config=MODEL_CONFIG, metrics=["masked_mae"], device="cpu",
            training_config={"amp": False},
        )
        ev_on = ReconstructionEvaluator(
            checkpoint_path=ckpt, index_path=idx, output_csv_path=res,
            model_config=MODEL_CONFIG, metrics=["masked_mae"], device="cpu",
            training_config={"loss": "masked_mse"},
        )

    assert ev_off.amp is False
    assert ev_on.amp is bf16_supported()


def test_amp_request_honoured_on_bf16_capable_hardware(evaluator_setup, monkeypatch):
    """When the hardware supports BF16, a config AMP request is honoured."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    monkeypatch.setattr("misfit.evaluation.evaluator.resolve_amp", lambda requested: requested)

    ckpt, idx, res = evaluator_setup
    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt, index_path=idx, output_csv_path=res,
            model_config=MODEL_CONFIG, metrics=["masked_mae"], device="cpu",
            training_config={"loss": "masked_mse"},
        )
    assert ev.amp is True


def test_run_inference_amp_disabled_uses_nullcontext(evaluator_setup):
    """With amp disabled, _run_inference still returns a recon/mask pair on CPU."""
    ckpt, idx, res = evaluator_setup
    ev = _make_evaluator(ckpt, idx, res)
    ev.amp = False

    patch_arr = np.zeros((32, 32, 32), dtype=np.float32)
    recon, mask = ev._run_inference(patch_arr)
    assert recon.shape == (32, 32, 32)
    assert mask.shape == (32, 32, 32)


def test_run_partial_errors_prints_warning(evaluator_setup, tmp_path):
    """When some volumes fail n_errors > 0 → print_warning path."""
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    # Build an index with two volumes: one good, one bad path
    data = np.random.randn(32, 32, 32).astype(np.float32)
    nifti_good = tmp_path / "good.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(nifti_good))

    df = pd.DataFrame([
        {"volume_id": "good", "path": str(nifti_good),
         "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0},
        {"volume_id": "bad",  "path": "/nonexistent/vol.nii.gz",
         "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0},
    ])
    idx2 = tmp_path / "index2.parquet"
    df.to_parquet(idx2, index=False)

    ckpt = _make_tiny_checkpoint(tmp_path)
    res2 = tmp_path / "res2" / "evaluation_results.csv"

    with patch.object(
        ReconstructionEvaluator,
        "_build_model",
        lambda self: _build_tiny_model(self.checkpoint, self.model_config, self.device),
    ):
        ev = ReconstructionEvaluator(
            checkpoint_path=ckpt, index_path=idx2, output_csv_path=res2,
            model_config=MODEL_CONFIG, metrics=["masked_mae"], device="cpu",
        )

    with patch("misfit.evaluation.evaluator.print_warning") as mock_warn:
        ev.run()

    # At least one warning about skipped volumes
    warning_texts = [str(c) for c in mock_warn.call_args_list]
    assert any("failed" in t for t in warning_texts)
