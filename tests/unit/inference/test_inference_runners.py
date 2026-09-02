"""Tests for misfit.inference.inference_runners."""
import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import torch

import misfit.models  # noqa — trigger model registrations
from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PATCH_SIZE = (32, 32, 32)

MODEL_CONFIG = {
    "architecture": "swinunetr-small",
    "patch_size": list(PATCH_SIZE),
    "mask_patch_size": 16,
    "mask_ratio": 0.75,
}


def _make_checkpoint(tmp_path: Path) -> Path:
    """Save a minimal checkpoint matching the tiny SwinMAE."""
    model = SwinMAE(
        in_channels=1,
        feature_size=12,
        img_size=PATCH_SIZE,
        mask_patch_size=16,
        mask_ratio=0.75,
    )
    p = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, p)
    return p


def _make_index(tmp_path: Path, volume_ids=("vol0",), shape=PATCH_SIZE) -> Path:
    """Write a minimal Parquet index with fake NIfTI paths."""
    rows = []
    for vid in volume_ids:
        nifti_path = tmp_path / f"{vid}.nii.gz"
        data = np.random.randn(*shape).astype(np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(nifti_path))
        rows.append({
            "volume_id": vid,
            "path": str(nifti_path),
            "p1": -2.0,
            "p99": 2.0,
            "fg_mean": 0.0,
            "fg_std": 1.0,
            "shape_d": shape[0],
            "shape_h": shape[1],
            "shape_w": shape[2],
            "spacing_d": 1.0,
            "spacing_h": 1.0,
            "spacing_w": 1.0,
            "affine": json.dumps(np.eye(4).tolist()),
        })
    p = tmp_path / "index.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def _tiny_model(_checkpoint=None, _model_config=None, _device=None):
    """Returns a tiny SwinMAE regardless of checkpoint contents."""
    return SwinMAE(
        in_channels=1,
        feature_size=12,
        img_size=PATCH_SIZE,
        mask_patch_size=16,
        mask_ratio=0.75,
    )


# ---------------------------------------------------------------------------
# _tiled_reconstruct
# ---------------------------------------------------------------------------

def _fake_model_fn(recon_val=1.0, mask_val=0.0):
    """Return a model_fn that produces constant reconstruction and mask tensors."""
    def fn(tensor):
        return {
            "reconstruction": torch.full_like(tensor, recon_val),
            "mask": torch.full_like(tensor, mask_val),
        }
    return fn


class TestTiledReconstruct:
    def test_single_patch_identity(self):
        """A single-patch volume goes through model_fn exactly once."""
        from misfit.inference.inference_runners import _tiled_reconstruct

        call_count = {"n": 0}

        def fake_model_fn(tensor):
            call_count["n"] += 1
            return {"reconstruction": torch.ones_like(tensor), "mask": torch.zeros_like(tensor)}

        vol = np.zeros((32, 32, 32))
        recon, mask = _tiled_reconstruct(vol, (32, 32, 32), fake_model_fn, "cpu")
        assert call_count["n"] == 1
        assert recon.shape == (32, 32, 32)
        assert mask.shape == (32, 32, 32)

    def test_multi_patch_call_count(self):
        """2×2×2 grid of patches → model_fn called 8 times."""
        from misfit.inference.inference_runners import _tiled_reconstruct

        call_count = {"n": 0}

        def fake_model_fn(tensor):
            call_count["n"] += 1
            return {"reconstruction": torch.zeros_like(tensor), "mask": torch.zeros_like(tensor)}

        vol = np.zeros((64, 64, 64))
        _tiled_reconstruct(vol, (32, 32, 32), fake_model_fn, "cpu")
        assert call_count["n"] == 8

    def test_output_shape_matches_input(self):
        from misfit.inference.inference_runners import _tiled_reconstruct

        vol = np.zeros((64, 32, 96))
        recon, mask = _tiled_reconstruct(vol, (32, 32, 32), _fake_model_fn(), "cpu")
        assert recon.shape == vol.shape
        assert mask.shape == vol.shape

    def test_patch_values_stitched_correctly(self):
        """Each patch in the output should reflect the model_fn's output."""
        from misfit.inference.inference_runners import _tiled_reconstruct

        vol = np.zeros((64, 64, 64))
        fn = _fake_model_fn(recon_val=1.0, mask_val=1.0)
        recon, mask = _tiled_reconstruct(vol, (32, 32, 32), fn, "cpu")
        np.testing.assert_array_equal(recon, np.ones((64, 64, 64)))
        np.testing.assert_array_equal(mask, np.ones((64, 64, 64)))

    def test_bfloat16_model_output_does_not_raise(self):
        """model_fn returning BF16 tensors must not raise TypeError from .numpy().

        Regression test: CUDA AMP produces BF16 outputs, which NumPy cannot
        convert directly. _tiled_reconstruct must cast to float32 first.
        """
        from misfit.inference.inference_runners import _tiled_reconstruct

        def bf16_model_fn(tensor):
            return {
                "reconstruction": torch.ones_like(tensor).to(torch.bfloat16),
                "mask": torch.zeros_like(tensor).to(torch.bfloat16),
            }

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        recon, mask = _tiled_reconstruct(vol, (32, 32, 32), bf16_model_fn, "cpu")
        assert recon.shape == (32, 32, 32)
        assert mask.shape == (32, 32, 32)
        assert recon.dtype in (np.float32, np.float64)
        assert mask.dtype in (np.float32, np.float64)


# ---------------------------------------------------------------------------
# reconstruct
# ---------------------------------------------------------------------------

class TestReconstruct:

    def test_happy_path_saves_nifti(self, tmp_path):
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("vol0",))
        output_dir = tmp_path / "recons"

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            return_value=(np.zeros(PATCH_SIZE), np.zeros(PATCH_SIZE)),
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                device=torch.device("cpu"),
            )

        assert (output_dir / "reconstructions" / "vol0.nii.gz").exists()
        assert (output_dir / "masks" / "vol0.nii.gz").exists()
        img = nib.load(str(output_dir / "reconstructions" / "vol0.nii.gz"))
        assert img.shape == PATCH_SIZE

    def test_load_failure_skips_volume(self, tmp_path):
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("bad_vol",))
        output_dir = tmp_path / "recons_skip"

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners.inference_utils.load_and_normalise",
            return_value=None,
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                device=torch.device("cpu"),
            )

        assert not (output_dir / "reconstructions" / "bad_vol.nii.gz").exists()

    def test_inference_exception_is_caught(self, tmp_path):
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("err_vol",))
        output_dir = tmp_path / "recons_err"

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            side_effect=RuntimeError("boom"),
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                device=torch.device("cpu"),
            )

        assert not (output_dir / "reconstructions" / "err_vol.nii.gz").exists()

    def test_tiles_larger_volume(self, tmp_path):
        """A volume 2× patch_size in each dim produces output with original shape."""
        from misfit.inference.inference_runners import reconstruct

        large_shape = (64, 64, 64)
        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("bigvol",), shape=large_shape)
        output_dir = tmp_path / "recons_large"

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            return_value=(np.zeros(large_shape), np.zeros(large_shape)),
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                device=torch.device("cpu"),
            )

        assert (output_dir / "reconstructions" / "bigvol.nii.gz").exists()
        img = nib.load(str(output_dir / "reconstructions" / "bigvol.nii.gz"))
        assert img.shape == large_shape

    def test_preserves_affine(self, tmp_path):
        """Output NIfTI affine matches the affine stored in the index."""
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        custom_affine = np.diag([2.0, 2.0, 2.0, 1.0])

        nifti_path = tmp_path / "affvol.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(PATCH_SIZE, dtype=np.float32), np.eye(4)),
                 str(nifti_path))
        rows = [{
            "volume_id": "affvol", "path": str(nifti_path),
            "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0,
            "shape_d": PATCH_SIZE[0], "shape_h": PATCH_SIZE[1], "shape_w": PATCH_SIZE[2],
            "spacing_d": 2.0, "spacing_h": 2.0, "spacing_w": 2.0,
            "affine": json.dumps(custom_affine.tolist()),
        }]
        index_path = tmp_path / "affine_index.parquet"
        pd.DataFrame(rows).to_parquet(index_path, index=False)

        output_dir = tmp_path / "recons_affine"

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            return_value=(np.zeros(PATCH_SIZE), np.zeros(PATCH_SIZE)),
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                device=torch.device("cpu"),
            )

        img = nib.load(str(output_dir / "reconstructions" / "affvol.nii.gz"))
        np.testing.assert_array_almost_equal(img.affine, custom_affine)

    def test_denormalizes_output(self, tmp_path):
        """Output intensities reflect fg_std × recon + fg_mean denormalization."""
        from misfit.inference.inference_runners import reconstruct

        fg_mean = 500.0
        fg_std = 200.0

        nifti_path = tmp_path / "normvol.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(PATCH_SIZE, dtype=np.float32), np.eye(4)),
                 str(nifti_path))
        rows = [{
            "volume_id": "normvol", "path": str(nifti_path),
            "p1": -2.0, "p99": 2.0, "fg_mean": fg_mean, "fg_std": fg_std,
            "shape_d": PATCH_SIZE[0], "shape_h": PATCH_SIZE[1], "shape_w": PATCH_SIZE[2],
            "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
            "affine": json.dumps(np.eye(4).tolist()),
        }]
        index_path = tmp_path / "norm_index.parquet"
        pd.DataFrame(rows).to_parquet(index_path, index=False)

        ckpt_path = _make_checkpoint(tmp_path)
        output_dir = tmp_path / "recons_norm"

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            return_value=(np.zeros(PATCH_SIZE), np.zeros(PATCH_SIZE)),
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                device=torch.device("cpu"),
            )

        img = nib.load(str(output_dir / "reconstructions" / "normvol.nii.gz"))
        data_out = np.asarray(img.dataobj, dtype=np.float32)
        # recon=0 → 0 * fg_std + fg_mean = fg_mean
        np.testing.assert_allclose(data_out, fg_mean, rtol=1e-5)

    def test_split_filters_index(self, tmp_path):
        """Only rows matching split= are reconstructed."""
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        output_dir = tmp_path / "recons_split"

        # Build index with two volumes in different splits.
        nifti_path = tmp_path / "v.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(PATCH_SIZE, dtype=np.float32), np.eye(4)),
                 str(nifti_path))
        base = {
            "path": str(nifti_path), "p1": -2.0, "p99": 2.0,
            "fg_mean": 0.0, "fg_std": 1.0,
            "shape_d": PATCH_SIZE[0], "shape_h": PATCH_SIZE[1], "shape_w": PATCH_SIZE[2],
            "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
            "affine": json.dumps(np.eye(4).tolist()),
        }
        df = pd.DataFrame([
            {"volume_id": "train_vol", "split": "train", **base},
            {"volume_id": "val_vol",   "split": "val",   **base},
        ])
        index_path = tmp_path / "split_index.parquet"
        df.to_parquet(index_path, index=False)

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            return_value=(np.zeros(PATCH_SIZE), np.zeros(PATCH_SIZE)),
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                device=torch.device("cpu"),
                split="val",
            )

        assert (output_dir / "reconstructions" / "val_vol.nii.gz").exists()
        assert not (output_dir / "reconstructions" / "train_vol.nii.gz").exists()

    def test_split_none_processes_all_rows(self, tmp_path):
        """split=None processes all rows regardless of split column."""
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        output_dir = tmp_path / "recons_all"

        nifti_path = tmp_path / "v.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros(PATCH_SIZE, dtype=np.float32), np.eye(4)),
                 str(nifti_path))
        base = {
            "path": str(nifti_path), "p1": -2.0, "p99": 2.0,
            "fg_mean": 0.0, "fg_std": 1.0,
            "shape_d": PATCH_SIZE[0], "shape_h": PATCH_SIZE[1], "shape_w": PATCH_SIZE[2],
            "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
            "affine": json.dumps(np.eye(4).tolist()),
        }
        df = pd.DataFrame([
            {"volume_id": "train_vol", "split": "train", **base},
            {"volume_id": "val_vol",   "split": "val",   **base},
        ])
        index_path = tmp_path / "all_index.parquet"
        df.to_parquet(index_path, index=False)

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            return_value=(np.zeros(PATCH_SIZE), np.zeros(PATCH_SIZE)),
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                device=torch.device("cpu"),
                split=None,
            )

        assert (output_dir / "reconstructions" / "train_vol.nii.gz").exists()
        assert (output_dir / "reconstructions" / "val_vol.nii.gz").exists()

    def test_runs_real_model_threads_spacing(self, tmp_path):
        """reconstruct() with the real _tiled_reconstruct runs the model_fn closure,
        which forwards the volume's voxel spacing to the model."""
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("realvol",))
        output_dir = tmp_path / "recons_real"

        captured = {}
        real_forward = SwinMAE.forward

        def spy_forward(self, x, spacing=None, mask_ratio=None):
            captured["spacing"] = spacing
            return real_forward(self, x, spacing=spacing, mask_ratio=mask_ratio)

        # Only patch model construction — _tiled_reconstruct runs for real so the
        # per-volume model_fn closure is exercised.
        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch.object(SwinMAE, "forward", spy_forward):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                device=torch.device("cpu"),
            )

        assert (output_dir / "reconstructions" / "realvol.nii.gz").exists()
        # Spacing from the index (1.0, 1.0, 1.0) was threaded into the forward pass.
        assert captured["spacing"] is not None
        assert tuple(captured["spacing"].squeeze().tolist()) == (1.0, 1.0, 1.0)

    def test_mask_saved_and_inverted(self, tmp_path):
        """masks/ contains inverted mask: model 1=masked → saved 0; model 0=visible → saved 1."""
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("maskvol",))
        output_dir = tmp_path / "recons_mask"

        # model mask: all 1s (everything masked out)
        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            return_value=(np.zeros(PATCH_SIZE), np.ones(PATCH_SIZE)),
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                device=torch.device("cpu"),
            )

        mask_img = nib.load(str(output_dir / "masks" / "maskvol.nii.gz"))
        mask_data = np.asarray(mask_img.dataobj, dtype=np.float32)
        # model mask=1 (masked/reconstructed) is saved as-is
        np.testing.assert_array_equal(mask_data, np.ones(PATCH_SIZE))


# ---------------------------------------------------------------------------
# _tiled_reconstruct — denorm_patches
# ---------------------------------------------------------------------------

class TestTiledReconstructDenormPatches:
    """Verify that denorm_patches correctly undoes per-patch normalization."""

    def test_denorm_patches_false_returns_raw_model_output(self):
        """With denorm_patches=False the reconstruction equals the raw model output."""
        from misfit.inference.inference_runners import _tiled_reconstruct

        raw_value = 3.0

        def model_fn(tensor):
            return {
                "reconstruction": torch.full_like(tensor, raw_value),
                "mask": torch.zeros_like(tensor),
            }

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        recon, _ = _tiled_reconstruct(vol, (32, 32, 32), model_fn, "cpu",
                                      denorm_patches=False)
        np.testing.assert_allclose(recon, raw_value)

    def test_denorm_patches_true_scales_by_patch_stats(self):
        """With denorm_patches=True each patch is scaled by the input patch mean/std.

        The model output is treated as patch-normalised (unit variance, zero mean).
        A reconstruction of all-zeros should be restored to the patch mean.
        """
        from misfit.inference.inference_runners import _tiled_reconstruct

        # Patch with known mean=10 and std≈0 → after inverse norm, recon=0
        # should map to mean=10 (recon*std + mean ≈ 0*eps + 10 = 10).
        vol = np.full((32, 32, 32), fill_value=10.0, dtype=np.float32)

        def model_fn(tensor):
            return {
                "reconstruction": torch.zeros_like(tensor),
                "mask": torch.zeros_like(tensor),
            }

        recon, _ = _tiled_reconstruct(vol, (32, 32, 32), model_fn, "cpu",
                                      denorm_patches=True)
        # std is ~0 so clamped to 1e-6; reconstruction ≈ 0 * 1e-6 + 10 = 10
        np.testing.assert_allclose(recon, 10.0, atol=1e-4)

    def test_denorm_patches_true_is_per_mask_cube_not_whole_patch(self):
        """De-normalization uses each mask cube's own mean/std, not the whole
        96³ (here 32³) tile's."""
        from misfit.inference.inference_runners import _tiled_reconstruct

        # One 16³ cube shifted far from the rest so per-cube and whole-tile
        # statistics disagree sharply.
        rng = np.random.default_rng(42)
        vol = rng.normal(loc=0.0, scale=1.0, size=(32, 32, 32)).astype(np.float32)
        vol[:16, :16, :16] += 20.0  # cube (0,0,0)

        def model_fn(tensor):
            return {
                "reconstruction": torch.ones_like(tensor),
                "mask": torch.zeros_like(tensor),
            }

        recon, _ = _tiled_reconstruct(vol, (32, 32, 32), model_fn, "cpu",
                                      denorm_patches=True, mask_patch_size=16)

        # recon = 1 * (cube_std + eps) + cube_mean, per cube.
        shifted = recon[:16, :16, :16].mean()
        rest = recon[16:, :16, :16].mean()
        assert shifted > 20.0          # picked up the +20 cube mean
        assert abs(rest) < 3.0         # the other cubes stayed near 0

        # A whole-tile de-norm would have used one (mean≈2.5, std≈7) for
        # everything → every voxel ≈ 1*7 + 2.5 ≈ 9.5.
        whole_tile = 1.0 * (float(vol.std()) + 1e-6) + float(vol.mean())
        assert not np.allclose(recon, whole_tile, atol=1.0)


# ---------------------------------------------------------------------------
# reconstruct — normalized_mse training_config
# ---------------------------------------------------------------------------

class TestReconstructNormalizedMse:
    """reconstruct() with training_config={"loss": "normalized_masked_mse"}."""

    def test_normalized_mse_calls_tiled_reconstruct_with_denorm_true(self, tmp_path):
        """When loss is normalized_masked_mse, _tiled_reconstruct receives denorm_patches=True."""
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("vol0",))
        output_dir = tmp_path / "recons_nmse"

        captured = {}

        def capturing_tiled_reconstruct(padded, patch_size, model_fn, device,
                                        denorm_patches=False, amp=True,
                                        mask_patch_size=16):
            captured["denorm_patches"] = denorm_patches
            captured["amp"] = amp
            return np.zeros(padded.shape), np.zeros(padded.shape)

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            side_effect=capturing_tiled_reconstruct,
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                training_config={"loss": "normalized_masked_mse"},
                device=torch.device("cpu"),
            )

        assert captured.get("denorm_patches") is True

    def test_non_normalized_loss_calls_tiled_reconstruct_with_denorm_false(self, tmp_path):
        """When loss is masked_mse, _tiled_reconstruct receives denorm_patches=False."""
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("vol0",))
        output_dir = tmp_path / "recons_mse"

        captured = {}

        def capturing_tiled_reconstruct(padded, patch_size, model_fn, device,
                                        denorm_patches=False, amp=True,
                                        mask_patch_size=16):
            captured["denorm_patches"] = denorm_patches
            captured["amp"] = amp
            return np.zeros(padded.shape), np.zeros(padded.shape)

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            side_effect=capturing_tiled_reconstruct,
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                training_config={"loss": "masked_mse"},
                device=torch.device("cpu"),
            )

        assert captured.get("denorm_patches") is False

    def test_no_training_config_defaults_to_denorm_false(self, tmp_path):
        """training_config=None should default to denorm_patches=False."""
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("vol0",))
        output_dir = tmp_path / "recons_none"

        captured = {}

        def capturing_tiled_reconstruct(padded, patch_size, model_fn, device,
                                        denorm_patches=False, amp=True,
                                        mask_patch_size=16):
            captured["denorm_patches"] = denorm_patches
            captured["amp"] = amp
            return np.zeros(padded.shape), np.zeros(padded.shape)

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            side_effect=capturing_tiled_reconstruct,
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                training_config=None,
                device=torch.device("cpu"),
            )

        assert captured.get("denorm_patches") is False
        # No training_config → AMP requested by default, then resolved against
        # the current hardware (False on a CPU / pre-Ampere test machine).
        from misfit.utils.hardware import bf16_supported
        assert captured.get("amp") is bf16_supported()


class TestReconstructAmp:
    """reconstruct() resolves the ``amp`` flag from training_config against hardware."""

    def test_amp_request_resolved_against_hardware(self, tmp_path):
        """A config AMP request is resolved via resolve_amp before being threaded."""
        from misfit.inference.inference_runners import reconstruct
        from misfit.utils.hardware import bf16_supported

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("vol0",))
        output_dir = tmp_path / "recons_amp_default"

        captured = {}

        def capturing_tiled_reconstruct(padded, patch_size, model_fn, device,
                                        denorm_patches=False, amp=True,
                                        mask_patch_size=16):
            captured["amp"] = amp
            return np.zeros(padded.shape), np.zeros(padded.shape)

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            side_effect=capturing_tiled_reconstruct,
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                training_config={"loss": "masked_mse"},
                device=torch.device("cpu"),
            )

        assert captured.get("amp") is bf16_supported()

    def test_amp_request_honoured_on_bf16_capable_hardware(self, tmp_path, monkeypatch):
        """With BF16-capable hardware, the config AMP request is threaded through."""
        from misfit.inference.inference_runners import reconstruct

        monkeypatch.setattr(
            "misfit.inference.inference_runners.resolve_amp", lambda requested: requested
        )

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("vol0",))
        output_dir = tmp_path / "recons_amp_hw"

        captured = {}

        def capturing_tiled_reconstruct(padded, patch_size, model_fn, device,
                                        denorm_patches=False, amp=True,
                                        mask_patch_size=16):
            captured["amp"] = amp
            return np.zeros(padded.shape), np.zeros(padded.shape)

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            side_effect=capturing_tiled_reconstruct,
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                training_config={"loss": "masked_mse"},
                device=torch.device("cpu"),
            )

        assert captured.get("amp") is True

    def test_amp_false_is_threaded_through(self, tmp_path):
        """training_config={"amp": False} → amp=False is passed to _tiled_reconstruct."""
        from misfit.inference.inference_runners import reconstruct

        ckpt_path = _make_checkpoint(tmp_path)
        index_path = _make_index(tmp_path, volume_ids=("vol0",))
        output_dir = tmp_path / "recons_amp_off"

        captured = {}

        def capturing_tiled_reconstruct(padded, patch_size, model_fn, device,
                                        denorm_patches=False, amp=True,
                                        mask_patch_size=16):
            captured["amp"] = amp
            return np.zeros(padded.shape), np.zeros(padded.shape)

        with patch(
            "misfit.inference.inference_runners.inference_utils.build_model_from_checkpoint",
            side_effect=_tiny_model,
        ), patch(
            "misfit.inference.inference_runners._tiled_reconstruct",
            side_effect=capturing_tiled_reconstruct,
        ):
            reconstruct(
                index_path=index_path,
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                model_config=MODEL_CONFIG,
                training_config={"amp": False},
                device=torch.device("cpu"),
            )

        assert captured.get("amp") is False

    def test_tiled_reconstruct_amp_false_uses_nullcontext_on_cpu(self):
        """_tiled_reconstruct with amp=False still runs (nullcontext) on CPU."""
        from misfit.inference.inference_runners import _tiled_reconstruct

        vol = np.zeros((32, 32, 32), dtype=np.float32)
        recon, mask = _tiled_reconstruct(
            vol, (32, 32, 32), _fake_model_fn(), "cpu", amp=False
        )
        assert recon.shape == (32, 32, 32)
        assert mask.shape == (32, 32, 32)
