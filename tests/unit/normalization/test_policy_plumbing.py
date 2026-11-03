"""--normalization-policy reaches every path that turns a NIfTI into model input."""
import json
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

import misfit.models  # noqa: F401 — trigger registrations
from misfit.normalization.base import IntensityScale
from misfit.normalization.policies import ClipAndZScore, ZScoreOnly

POLICY = "z_score_only"
MODEL_CONFIG = {
    "architecture": "swinunetr-small", "patch_size": [32, 32, 32],
    "mask_patch_size": 16, "mask_ratio": 0.75,
}


@pytest.fixture
def files(tmp_path):
    """A one-volume index and a config.json that selects z_score_only."""
    nifti = tmp_path / "vol.nii.gz"
    nib.save(nib.Nifti1Image(
        (np.random.randn(32, 32, 32) * 50 + 200).astype(np.float32), np.eye(4)
    ), str(nifti))
    index = tmp_path / "index.parquet"
    row = {
        "volume_id": "vol", "path": str(nifti), "split": "val",
        "shape_d": 32, "shape_h": 32, "shape_w": 32,
        "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
        "affine": json.dumps(np.eye(4).tolist()),
        "fg_x_start": 0, "fg_x_end": 31, "fg_y_start": 0, "fg_y_end": 31,
        "fg_z_start": 0, "fg_z_end": 31,
        "p1": 100.0, "p99": 300.0, "fg_mean": 200.0, "fg_std": 50.0,
    }
    pd.DataFrame([row, {**row, "volume_id": "vol_train", "split": "train"}]).to_parquet(
        index, index=False
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "model": MODEL_CONFIG, "data": {"normalization_policy": POLICY},
    }))
    return index, config


def test_dataset_applies_its_policy(files):
    from misfit.data_loading.dataset import MISFITDataset

    index, _ = files
    assert isinstance(MISFITDataset(index, (32, 32, 32)).normalization_policy, ClipAndZScore)
    ds = MISFITDataset(index, (32, 32, 32), normalization_policy=POLICY)
    volume = np.asarray(nib.load(ds.index_df.iloc[0]["path"]).dataobj, np.float32)
    out = ds._normalize(volume, ds.index_df.iloc[0])
    assert abs(out.mean()) < 1e-4 and abs(out.std() - 1) < 1e-4  # no clipping


@pytest.mark.parametrize("factory", ["get_training_dataloader", "get_validation_dataloader"])
def test_dataloaders_forward_the_policy(files, factory):
    import misfit.data_loading.dataloader as dl

    index, _ = files
    loader = getattr(dl, factory)(index, (32, 32, 32), num_workers=0,
                                  normalization_policy=POLICY)
    assert isinstance(loader.dataset.normalization_policy, ZScoreOnly)


def test_evaluator_applies_the_policy(files):
    from misfit.evaluation.evaluator import ReconstructionEvaluator

    index, _ = files
    with patch.object(ReconstructionEvaluator, "_build_model", lambda self: None):
        ev = ReconstructionEvaluator(
            checkpoint_path=_checkpoint(index.parent), index_path=index,
            output_csv_path=index.parent / "out.csv", model_config=MODEL_CONFIG,
            device="cpu", normalization_policy=POLICY,
        )
    data = ev._load_and_normalize(ev.index_df.iloc[0])
    assert abs(data.mean()) < 1e-4  # z-scored over the whole volume, unclipped


def _checkpoint(directory):
    path = directory / "ckpt.pt"
    torch.save({"model": {}}, path)
    return path


def test_evaluate_entry_forwards_the_config_policy(files):
    from misfit.cli.evaluate_entrypoint import evaluate_entry

    index, config = files
    with patch("misfit.cli.evaluate_entrypoint.ReconstructionEvaluator") as ev:
        evaluate_entry(["--checkpoint", "c.pt", "--config", str(config),
                        "--index", str(index), "--output-csv", "o.csv"])
    assert ev.call_args.kwargs["normalization_policy"] == POLICY


def test_inspect_entry_forwards_the_config_policy(files):
    from misfit.cli.inspect_entrypoint import inspect_entry

    index, config = files
    with patch("misfit.cli.inspect_entrypoint.reconstruct") as rec:
        inspect_entry(["--checkpoint", "c.pt", "--config", str(config),
                       "--index", str(index), "--output-dir", "out"])
    assert rec.call_args.kwargs["normalization_policy"] == POLICY


@pytest.mark.parametrize("command", ["encode", "embed"])
def test_encode_and_embed_forward_the_config_policy(files, tmp_path, command):
    import misfit.inference.inference_utils as iutils

    index, config = files
    args = ["--encoder-checkpoint", "c.pt", "--config", str(config),
            "--index", str(index), "--output-dir", str(tmp_path / command)]
    with patch.object(iutils, "load_checkpoint", return_value={"model": {}}), \
         patch.object(iutils, "build_model_from_checkpoint", return_value=MagicMock()), \
         patch.object(iutils, "load_and_normalize", return_value=None) as load:
        if command == "encode":
            from misfit.cli.encode_entrypoint import encode_entry
            encode_entry(args)
        else:
            from misfit.cli.embed_entrypoint import embed_entry
            embed_entry(args)
    assert load.call_args.kwargs["normalization_policy"] == POLICY


def test_reconstruct_denormalizes_with_the_policy_scale(files, tmp_path):
    """Inspect inverts whatever affine map the policy applied."""
    import misfit.inference.inference_runners as runners

    index, _ = files
    volume = np.zeros((32, 32, 32), np.float32)
    with patch.object(runners.inference_utils, "load_checkpoint", return_value={}), \
         patch.object(runners.inference_utils, "build_model_from_checkpoint"), \
         patch.object(runners.inference_utils, "load_and_normalize",
                      return_value=(volume, IntensityScale(shift=100.0, scale=10.0))) as load, \
         patch.object(runners, "_tiled_reconstruct",
                      return_value=(np.full_like(volume, 2.0), np.zeros_like(volume))):
        runners.reconstruct(index_path=index, checkpoint_path="c.pt",
                            output_dir=tmp_path / "rec", model_config=MODEL_CONFIG,
                            device="cpu", normalization_policy=POLICY)

    assert load.call_args.kwargs == {
        "normalization_policy": POLICY, "target_spacing": None, "return_scale": True,
    }
    recon = nib.load(tmp_path / "rec" / "reconstructions" / "vol.nii.gz").get_fdata()
    np.testing.assert_allclose(recon, 2.0 * 10.0 + 100.0)
