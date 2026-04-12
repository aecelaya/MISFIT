"""Tests for misfit.cli.encode_entrypoint."""
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


PATCH_SIZE = 32
MODEL_CONFIG = {
    "architecture": "_tiny_test_model",
    "patch_size": [PATCH_SIZE, PATCH_SIZE, PATCH_SIZE],
    "mask_patch_size": 16,
    "mask_ratio": 0.75,
}


def _make_tiny_model():
    return SwinMAE(
        in_channels=1,
        feature_size=12,
        img_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
        mask_patch_size=16,
        mask_ratio=0.75,
    )


def _make_index(tmp_path: Path, volume_ids=("vol",), splits=None) -> Path:
    nifti_path = tmp_path / "vol.nii.gz"
    nib.save(
        nib.Nifti1Image(
            np.random.randn(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE).astype(np.float32),
            np.eye(4),
        ),
        str(nifti_path),
    )
    rows = []
    for i, vid in enumerate(volume_ids):
        row = {
            "volume_id": vid,
            "path": str(nifti_path),
            "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0,
        }
        if splits is not None:
            row["split"] = splits[i]
        rows.append(row)
    p = tmp_path / "index.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def _make_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model": MODEL_CONFIG}))
    return config_path


def _make_checkpoint(tmp_path: Path) -> Path:
    model = _make_tiny_model()
    p = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict()}, p)
    return p


# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------

def test_add_encode_args_defaults():
    from misfit.cli.args import ArgParser, add_encode_args
    p = ArgParser()
    add_encode_args(p)
    ns = p.parse_args([
        "--encoder-checkpoint", "best.pt",
        "--config", "config.json",
        "--index", "index.parquet",
        "--output-dir", "/out",
    ])
    assert ns.encoder_checkpoint == "best.pt"
    assert ns.config == "config.json"
    assert ns.split is None
    assert ns.device is None


# ---------------------------------------------------------------------------
# encode_entry
# ---------------------------------------------------------------------------

def test_encode_entry_saves_npz(tmp_path):
    """encode_entry produces a .npz file with feature_map and positions."""
    from misfit.cli.encode_entrypoint import encode_entry

    index_path = _make_index(tmp_path)
    config_path = _make_config(tmp_path)
    ckpt_path = _make_checkpoint(tmp_path)
    output_dir = tmp_path / "encodings"

    import misfit.inference.inference_utils as _iutils
    with patch.object(_iutils, "load_checkpoint", return_value={"model": {}}), \
         patch.object(_iutils, "build_model_from_checkpoint",
                      return_value=_make_tiny_model()), \
         patch.object(_iutils, "load_and_normalise",
                      return_value=np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
                                            dtype=np.float32)):
        encode_entry([
            "--encoder-checkpoint", str(ckpt_path),
            "--config", str(config_path),
            "--index", str(index_path),
            "--output-dir", str(output_dir),
        ])

    out = output_dir / "vol.npz"
    assert out.exists()
    data = np.load(str(out))
    assert "feature_map" in data
    assert "positions" in data
    assert data["feature_map"].ndim == 5   # (N_crops, C, D', H', W')
    assert data["positions"].shape[1] == 3


def test_encode_entry_missing_config_exits(tmp_path):
    from misfit.cli.encode_entrypoint import encode_entry
    index_path = _make_index(tmp_path)
    with pytest.raises(SystemExit):
        encode_entry([
            "--encoder-checkpoint", "best.pt",
            "--config", str(tmp_path / "nonexistent.json"),
            "--index", str(index_path),
            "--output-dir", str(tmp_path / "out"),
        ])


def test_encode_entry_skips_existing_output(tmp_path):
    """encode_entry skips volumes whose .npz already exists."""
    from misfit.cli.encode_entrypoint import encode_entry

    index_path = _make_index(tmp_path)
    config_path = _make_config(tmp_path)
    ckpt_path = _make_checkpoint(tmp_path)
    output_dir = tmp_path / "encodings"
    output_dir.mkdir()
    # Pre-create the output file.
    (output_dir / "vol.npz").touch()

    import misfit.inference.inference_utils as _iutils
    with patch.object(_iutils, "load_checkpoint", return_value={"model": {}}), \
         patch.object(_iutils, "build_model_from_checkpoint",
                      return_value=_make_tiny_model()), \
         patch.object(_iutils, "load_and_normalise",
                      return_value=np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
                                            dtype=np.float32)) as mock_load:
        encode_entry([
            "--encoder-checkpoint", str(ckpt_path),
            "--config", str(config_path),
            "--index", str(index_path),
            "--output-dir", str(output_dir),
        ])
    mock_load.assert_not_called()


def test_encode_entry_skips_on_load_failure(tmp_path):
    """encode_entry skips a volume when load_and_normalise returns None."""
    from misfit.cli.encode_entrypoint import encode_entry

    index_path = _make_index(tmp_path)
    config_path = _make_config(tmp_path)
    ckpt_path = _make_checkpoint(tmp_path)
    output_dir = tmp_path / "encodings"

    import misfit.inference.inference_utils as _iutils
    with patch.object(_iutils, "load_checkpoint", return_value={"model": {}}), \
         patch.object(_iutils, "build_model_from_checkpoint",
                      return_value=_make_tiny_model()), \
         patch.object(_iutils, "load_and_normalise", return_value=None):
        encode_entry([
            "--encoder-checkpoint", str(ckpt_path),
            "--config", str(config_path),
            "--index", str(index_path),
            "--output-dir", str(output_dir),
        ])

    assert not (output_dir / "vol.npz").exists()


def test_encode_entry_skips_on_exception(tmp_path):
    """encode_entry catches exceptions and does not propagate them."""
    from misfit.cli.encode_entrypoint import encode_entry, _encode_volume

    index_path = _make_index(tmp_path)
    config_path = _make_config(tmp_path)
    ckpt_path = _make_checkpoint(tmp_path)
    output_dir = tmp_path / "encodings"

    import misfit.inference.inference_utils as _iutils
    with patch.object(_iutils, "load_checkpoint", return_value={"model": {}}), \
         patch.object(_iutils, "build_model_from_checkpoint",
                      return_value=_make_tiny_model()), \
         patch.object(_iutils, "load_and_normalise",
                      return_value=np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
                                            dtype=np.float32)), \
         patch("misfit.cli.encode_entrypoint._encode_volume",
               side_effect=RuntimeError("boom")):
        encode_entry([
            "--encoder-checkpoint", str(ckpt_path),
            "--config", str(config_path),
            "--index", str(index_path),
            "--output-dir", str(output_dir),
        ])

    assert not (output_dir / "vol.npz").exists()


def test_encode_entry_split_filters_index(tmp_path):
    """encode_entry only processes rows matching --split."""
    from misfit.cli.encode_entrypoint import encode_entry

    index_path = _make_index(
        tmp_path,
        volume_ids=["train_vol", "val_vol"],
        splits=["train", "val"],
    )
    config_path = _make_config(tmp_path)
    ckpt_path = _make_checkpoint(tmp_path)
    output_dir = tmp_path / "encodings"

    import misfit.inference.inference_utils as _iutils
    with patch.object(_iutils, "load_checkpoint", return_value={"model": {}}), \
         patch.object(_iutils, "build_model_from_checkpoint",
                      return_value=_make_tiny_model()), \
         patch.object(_iutils, "load_and_normalise",
                      return_value=np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
                                            dtype=np.float32)):
        encode_entry([
            "--encoder-checkpoint", str(ckpt_path),
            "--config", str(config_path),
            "--index", str(index_path),
            "--output-dir", str(output_dir),
            "--split", "val",
        ])

    assert (output_dir / "val_vol.npz").exists()
    assert not (output_dir / "train_vol.npz").exists()


# ---------------------------------------------------------------------------
# _encode_volume
# ---------------------------------------------------------------------------

def test_encode_volume_output_shapes(tmp_path):
    """_encode_volume returns (N_crops, C, D', H', W') and (N_crops, 3)."""
    from misfit.cli.encode_entrypoint import _encode_volume

    model = _make_tiny_model()
    volume = np.zeros((PATCH_SIZE, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    feature_map, positions = _encode_volume(volume, model, PATCH_SIZE, torch.device("cpu"))

    assert feature_map.ndim == 5          # (N_crops, C, D', H', W')
    assert feature_map.shape[0] == 1      # single crop for exact-fit volume
    assert positions.shape == (1, 3)


def test_encode_volume_multi_crop(tmp_path):
    """A 2×-patch-size volume produces 8 crops."""
    from misfit.cli.encode_entrypoint import _encode_volume

    model = _make_tiny_model()
    volume = np.zeros((PATCH_SIZE * 2, PATCH_SIZE * 2, PATCH_SIZE * 2), dtype=np.float32)
    feature_map, positions = _encode_volume(volume, model, PATCH_SIZE, torch.device("cpu"))

    assert feature_map.shape[0] == 8
    assert positions.shape == (8, 3)


def test_encode_volume_positions_normalised(tmp_path):
    """Crop centre positions should be in (0, 1]."""
    from misfit.cli.encode_entrypoint import _encode_volume

    model = _make_tiny_model()
    volume = np.zeros((PATCH_SIZE * 2, PATCH_SIZE * 2, PATCH_SIZE * 2), dtype=np.float32)
    _, positions = _encode_volume(volume, model, PATCH_SIZE, torch.device("cpu"))

    assert positions.min() > 0.0
    assert positions.max() <= 1.0


def test_encode_volume_pads_non_multiple(tmp_path):
    """Volumes whose dimensions are not exact multiples of patch_size are padded."""
    from misfit.cli.encode_entrypoint import _encode_volume

    model = _make_tiny_model()
    # 40 is not a multiple of 32 — triggers padding branch
    volume = np.zeros((40, 40, 40), dtype=np.float32)
    feature_map, positions = _encode_volume(volume, model, PATCH_SIZE, torch.device("cpu"))

    # After padding to 64³, we get 2³ = 8 crops
    assert feature_map.shape[0] == 8
    assert positions.shape == (8, 3)
