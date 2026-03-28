"""Tests for misfit.inference.inference_runners."""
import json
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

import misfit.models    # noqa — trigger model registrations
from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PATCH_SIZE = (32, 32, 32)

MODEL_CONFIG = {
    "name": "swinmae-small",
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
        recon, mask = _tiled_reconstruct(vol, (32, 32, 32), _fake_model_fn(recon_val=1.0, mask_val=1.0), "cpu")
        np.testing.assert_array_equal(recon, np.ones((64, 64, 64)))
        np.testing.assert_array_equal(mask, np.ones((64, 64, 64)))


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
        nib.save(nib.Nifti1Image(np.zeros(PATCH_SIZE, dtype=np.float32), np.eye(4)), str(nifti_path))
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
        nib.save(nib.Nifti1Image(np.zeros(PATCH_SIZE, dtype=np.float32), np.eye(4)), str(nifti_path))
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
        nib.save(nib.Nifti1Image(np.zeros(PATCH_SIZE, dtype=np.float32), np.eye(4)), str(nifti_path))
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
        nib.save(nib.Nifti1Image(np.zeros(PATCH_SIZE, dtype=np.float32), np.eye(4)), str(nifti_path))
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
