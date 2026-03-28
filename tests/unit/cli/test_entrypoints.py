"""Tests for CLI entrypoints."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

import misfit.models  # noqa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_nifti_dir(tmp_path):
    data = np.random.randn(8, 8, 8).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, str(tmp_path / "vol.nii.gz"))


def _write_manifest(tmp_path, nifti_dir):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame({"path": [str(nifti_dir / "vol.nii.gz")]}).to_csv(manifest, index=False)
    return manifest


# ---------------------------------------------------------------------------
# index_entrypoint
# ---------------------------------------------------------------------------

class TestIndexEntry:
    def _write_csv(self, tmp_path):
        """Write a manifest CSV with one NIfTI path."""
        _write_nifti_dir(tmp_path)
        manifest = _write_manifest(tmp_path, tmp_path)
        return manifest

    def _write_parquet(self, tmp_path):
        """Write a manifest Parquet with one NIfTI path."""
        _write_nifti_dir(tmp_path)
        nifti_path = tmp_path / "vol.nii.gz"
        p = tmp_path / "paths.parquet"
        pd.DataFrame({"path": [str(nifti_path)]}).to_parquet(p, index=False)
        return p

    def test_with_csv_input(self, tmp_path):
        from misfit.cli.index_entrypoint import index_entry
        manifest = self._write_csv(tmp_path)
        output = tmp_path / "index.parquet"
        with patch("misfit.cli.index_entrypoint.build_index") as mock_build:
            mock_build.return_value = (None, [])
            with pytest.raises(SystemExit) as exc_info:
                index_entry([
                    "--input", str(manifest),
                    "--output", str(output),
                    "--num-workers-index", "1",
                ])
            assert exc_info.value.code == 0

    def test_with_parquet_input(self, tmp_path):
        from misfit.cli.index_entrypoint import index_entry
        manifest = self._write_parquet(tmp_path)
        output = tmp_path / "index.parquet"
        with patch("misfit.cli.index_entrypoint.build_index") as mock_build:
            mock_build.return_value = (None, [])
            with pytest.raises(SystemExit) as exc_info:
                index_entry([
                    "--input", str(manifest),
                    "--output", str(output),
                    "--num-workers-index", "1",
                ])
            assert exc_info.value.code == 0

    def test_nonexistent_input_exits(self, tmp_path):
        from misfit.cli.index_entrypoint import index_entry
        with pytest.raises(SystemExit):
            index_entry([
                "--input", str(tmp_path / "missing.csv"),
                "--output", str(tmp_path / "index.parquet"),
            ])

    def test_empty_input_exits(self, tmp_path):
        from misfit.cli.index_entrypoint import index_entry
        empty = tmp_path / "empty.csv"
        pd.DataFrame({"path": []}).to_csv(empty, index=False)
        with pytest.raises(SystemExit):
            index_entry([
                "--input", str(empty),
                "--output", str(tmp_path / "index.parquet"),
            ])

    def test_input_without_path_column_exits(self, tmp_path):
        from misfit.cli.index_entrypoint import index_entry
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"file": ["x.nii.gz"]}).to_csv(bad, index=False)
        with pytest.raises(SystemExit):
            index_entry([
                "--input", str(bad),
                "--output", str(tmp_path / "index.parquet"),
            ])

    def test_existing_split_config_is_reused(self, tmp_path):
        from misfit.cli.index_entrypoint import index_entry, _split_config_path
        import json
        manifest = self._write_csv(tmp_path)
        output = tmp_path / "index.parquet"
        # Pre-write a config so the "config exists" branch is taken.
        config_path = _split_config_path(output)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as fh:
            json.dump({"train": 0.7, "val": 0.15, "test": 0.15, "seed": 0}, fh)
        with patch("misfit.cli.index_entrypoint.build_index") as mock_build:
            mock_build.return_value = (None, [])
            with pytest.raises(SystemExit) as exc_info:
                index_entry([
                    "--input", str(manifest),
                    "--output", str(output),
                ])
            assert exc_info.value.code == 0
            _, kwargs = mock_build.call_args
            assert kwargs["split_ratios"] == {"train": 0.7, "val": 0.15, "test": 0.15}

    def test_unreadable_input_exits(self, tmp_path):
        from misfit.cli.index_entrypoint import index_entry
        # A file with .parquet extension that isn't actually Parquet triggers the
        # exception branch in _load_input_paths.
        bad = tmp_path / "garbage.parquet"
        bad.write_bytes(b"not a parquet file")
        with pytest.raises(SystemExit):
            index_entry([
                "--input", str(bad),
                "--output", str(tmp_path / "index.parquet"),
            ])

    def test_with_errors_exits_1(self, tmp_path):
        from misfit.cli.index_entrypoint import index_entry
        manifest = self._write_csv(tmp_path)
        output = tmp_path / "index.parquet"
        with patch("misfit.cli.index_entrypoint.build_index") as mock_build:
            mock_build.return_value = (None, ["error1"])
            with pytest.raises(SystemExit) as exc_info:
                index_entry([
                    "--input", str(manifest),
                    "--output", str(output),
                ])
            assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# train_entrypoint
# ---------------------------------------------------------------------------

def test_train_entry_calls_trainer(tmp_path):
    from misfit.cli.train_entrypoint import train_entry
    with patch("misfit.cli.train_entrypoint.MAETrainer") as MockTrainer:
        instance = MockTrainer.return_value
        instance.train.return_value = None
        train_entry([
            "--index", "index.parquet",
            "--results", str(tmp_path),
        ])
        MockTrainer.assert_called_once()
        instance.train.assert_called_once()


# ---------------------------------------------------------------------------
# evaluate_entrypoint
# ---------------------------------------------------------------------------

def test_evaluate_entry_calls_evaluator(tmp_path):
    from misfit.cli.evaluate_entrypoint import evaluate_entry
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model": {"name": "swinmae-small", "patch_size": [96, 96, 96], "mask_patch_size": 16, "mask_ratio": 0.75}, '
        '"evaluation": {"masked_mae": {}, "ssim": {}}}'
    )
    with patch("misfit.cli.evaluate_entrypoint.ReconstructionEvaluator") as MockEval:
        instance = MockEval.return_value
        instance.run.return_value = None
        evaluate_entry([
            "--checkpoint", "best.pt",
            "--config", str(config_path),
            "--index", "val.parquet",
            "--output-csv", str(tmp_path / "results.csv"),
        ])
        MockEval.assert_called_once()
        _, kwargs = MockEval.call_args
        assert kwargs["metrics"] == ["masked_mae", "ssim"]
        assert kwargs["model_config"]["name"] == "swinmae-small"
        instance.run.assert_called_once()


def test_evaluate_entry_missing_config_exits(tmp_path):
    from misfit.cli.evaluate_entrypoint import evaluate_entry
    with pytest.raises(SystemExit):
        evaluate_entry([
            "--checkpoint", "best.pt",
            "--config", str(tmp_path / "nonexistent.json"),
            "--index", "val.parquet",
            "--output-csv", str(tmp_path / "results.csv"),
        ])


def test_evaluate_entry_empty_evaluation_section_exits(tmp_path):
    from misfit.cli.evaluate_entrypoint import evaluate_entry
    config_path = tmp_path / "config.json"
    config_path.write_text('{"evaluation": {}}')
    with pytest.raises(SystemExit):
        evaluate_entry([
            "--checkpoint", "best.pt",
            "--config", str(config_path),
            "--index", "val.parquet",
            "--output-csv", str(tmp_path / "results.csv"),
        ])


# ---------------------------------------------------------------------------
# inspect_entrypoint
# ---------------------------------------------------------------------------

def test_inspect_entry_calls_reconstruct(tmp_path):
    from misfit.cli.inspect_entrypoint import inspect_entry
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model": {"name": "swinmae-small", "patch_size": [96, 96, 96], "mask_patch_size": 16, "mask_ratio": 0.75}}'
    )
    with patch("misfit.cli.inspect_entrypoint.reconstruct") as mock_fn:
        inspect_entry([
            "--checkpoint", "best.pt",
            "--config", str(config_path),
            "--index", "index.parquet",
            "--output-dir", str(tmp_path),
        ])
        mock_fn.assert_called_once()
        _, kwargs = mock_fn.call_args
        assert kwargs["model_config"]["name"] == "swinmae-small"


def test_inspect_entry_missing_config_exits(tmp_path):
    from misfit.cli.inspect_entrypoint import inspect_entry
    with pytest.raises(SystemExit):
        inspect_entry([
            "--checkpoint", "best.pt",
            "--config", str(tmp_path / "nonexistent.json"),
            "--index", "index.parquet",
            "--output-dir", str(tmp_path),
        ])


# ---------------------------------------------------------------------------
# embed_train_entrypoint
# ---------------------------------------------------------------------------

def test_embed_train_entry_calls_trainer(tmp_path):
    from misfit.cli.embed_train_entrypoint import embed_train_entry
    import misfit.embedding  # noqa
    with patch("misfit.embedding.embed_trainer.EmbedTrainer") as MockTrainer:
        instance = MockTrainer.return_value
        instance.run.return_value = None
        embed_train_entry([
            "--input", str(tmp_path / "input.csv"),
            "--output-dir", str(tmp_path),
            "--embed-dim", "64",
        ])
        MockTrainer.assert_called_once()
        instance.run.assert_called_once()


def test_embed_train_entry_value_error_exits(tmp_path):
    from misfit.cli.embed_train_entrypoint import embed_train_entry
    import misfit.embedding  # noqa
    with patch("misfit.embedding.embed_trainer.EmbedTrainer") as MockTrainer:
        MockTrainer.side_effect = ValueError("bad value")
        with pytest.raises(SystemExit):
            embed_train_entry([
                "--input", str(tmp_path / "input.csv"),
                "--output-dir", str(tmp_path),
                "--embed-dim", "64",
            ])


def test_embed_train_entry_runtime_error_exits(tmp_path):
    from misfit.cli.embed_train_entrypoint import embed_train_entry
    import misfit.embedding  # noqa
    with patch("misfit.embedding.embed_trainer.EmbedTrainer") as MockTrainer:
        MockTrainer.side_effect = RuntimeError("no files")
        with pytest.raises(SystemExit):
            embed_train_entry([
                "--input", str(tmp_path / "input.csv"),
                "--output-dir", str(tmp_path),
                "--embed-dim", "64",
            ])


# ---------------------------------------------------------------------------
# embed_entrypoint
# ---------------------------------------------------------------------------

def test_embed_entry_calls_extract_features(tmp_path):
    """embed_entry builds embedder and iterates the index."""
    from misfit.cli.embed_entrypoint import embed_entry

    # Write a tiny parquet index.
    nifti_path = tmp_path / "vol.nii.gz"
    data = np.random.randn(8, 8, 8).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, str(nifti_path))
    manifest = tmp_path / "index.parquet"
    pd.DataFrame({
        "volume_id": ["vol"],
        "path": [str(nifti_path)],
        "p1": [-2.0], "p99": [2.0], "fg_mean": [0.0], "fg_std": [1.0],
    }).to_parquet(manifest, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model": {"name": "swinmae-small", "patch_size": [32, 32, 32], "mask_patch_size": 16, "mask_ratio": 0.75}}'
    )

    output_dir = tmp_path / "embeddings"

    # embed_entry does lazy imports, so we patch the source modules.
    import misfit.inference.inference_utils as _iutils
    import misfit.embedding.aggregators.aggregator_registry as _areg
    import misfit.embedding.embedder as _emb_mod
    with patch.object(_iutils, "load_checkpoint", return_value={"model": {}}), \
    patch.object(_iutils, "build_model_from_checkpoint") as mock_model, \
    patch.object(_areg, "get_aggregator") as mock_agg_cls, \
    patch.object(_emb_mod, "Embedder") as mock_embedder_cls, \
    patch.object(_iutils, "load_and_normalise",
                 return_value=np.zeros((8, 8, 8), dtype=np.float32)):

        mock_model.return_value = MagicMock()
        mock_agg_instance = MagicMock()
        mock_agg_cls.return_value = MagicMock(return_value=mock_agg_instance)
        mock_embedder_instance = MagicMock()
        mock_embedder_instance.extract_crop_features.return_value = (
            np.zeros((2, 16), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        )
        mock_embedder_cls.return_value = mock_embedder_instance

        embed_entry([
            "--encoder-checkpoint", "best.pt",
            "--config", str(config_path),
            "--index", str(manifest),
            "--output-dir", str(output_dir),
        ])

    # output_dir should exist (created by embed_entry).
    assert output_dir.exists()


def test_embed_entry_skips_existing_output(tmp_path):
    """embed_entry skips volumes whose .npz already exists."""
    from misfit.cli.embed_entrypoint import embed_entry

    manifest = tmp_path / "index.parquet"
    pd.DataFrame({
        "volume_id": ["vol"],
        "path": ["doesnotmatter.nii.gz"],
        "p1": [-2.0], "p99": [2.0], "fg_mean": [0.0], "fg_std": [1.0],
    }).to_parquet(manifest, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model": {"name": "swinmae-small", "patch_size": [32, 32, 32], "mask_patch_size": 16, "mask_ratio": 0.75}}'
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Pre-create the output file so the embedder loop should skip it.
    (output_dir / "vol.npz").write_bytes(b"dummy")

    import misfit.inference.inference_utils as _iutils2
    import misfit.embedding.aggregators.aggregator_registry as _areg2
    import misfit.embedding.embedder as _emb_mod2
    with patch.object(_iutils2, "load_checkpoint", return_value={"model": {}}), \
    patch.object(_iutils2, "build_model_from_checkpoint", return_value=MagicMock()), \
    patch.object(_areg2, "get_aggregator",
                 return_value=MagicMock(return_value=MagicMock())), \
    patch.object(_emb_mod2, "Embedder") as mock_embedder_cls:
        mock_embedder_cls.return_value = MagicMock()
        embed_entry([
            "--encoder-checkpoint", "best.pt",
            "--config", str(config_path),
            "--index", str(manifest),
            "--output-dir", str(output_dir),
        ])
        # extract_crop_features should never be called since file exists.
        mock_embedder_cls.return_value.extract_crop_features.assert_not_called()


def test_embed_entry_with_aggregator_checkpoint(tmp_path):
    """embed_entry loads aggregator checkpoint if provided."""
    from misfit.cli.embed_entrypoint import embed_entry

    # Write dummy aggregator checkpoint.
    agg_ckpt = tmp_path / "agg.pt"
    torch.save({"aggregator_state": {}}, agg_ckpt)

    manifest = tmp_path / "index.parquet"
    pd.DataFrame({
        "volume_id": [],
        "path": [],
        "p1": [], "p99": [], "fg_mean": [], "fg_std": [],
    }).to_parquet(manifest, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model": {"name": "swinmae-small", "patch_size": [32, 32, 32], "mask_patch_size": 16, "mask_ratio": 0.75}}'
    )

    import misfit.inference.inference_utils as _iutils3
    import misfit.embedding.aggregators.aggregator_registry as _areg3
    import misfit.embedding.embedder as _emb_mod3
    with patch.object(_iutils3, "load_checkpoint", return_value={"model": {}}), \
    patch.object(_iutils3, "build_model_from_checkpoint", return_value=MagicMock()), \
    patch.object(_areg3, "get_aggregator",
                 return_value=MagicMock(return_value=MagicMock())), \
    patch.object(_emb_mod3, "Embedder", return_value=MagicMock()):
        embed_entry([
            "--encoder-checkpoint", "best.pt",
            "--config", str(config_path),
            "--index", str(manifest),
            "--output-dir", str(tmp_path / "out"),
            "--aggregator-checkpoint", str(agg_ckpt),
        ])


def test_embed_entry_missing_config_exits(tmp_path):
    """embed_entry exits 1 when --config file does not exist."""
    from misfit.cli.embed_entrypoint import embed_entry
    manifest = tmp_path / "index.parquet"
    pd.DataFrame({"volume_id": [], "path": []}).to_parquet(manifest, index=False)
    with pytest.raises(SystemExit):
        embed_entry([
            "--encoder-checkpoint", "best.pt",
            "--config", str(tmp_path / "nonexistent.json"),
            "--index", str(manifest),
            "--output-dir", str(tmp_path / "out"),
        ])


def test_embed_entry_infer_embed_dim(tmp_path):
    """_infer_embed_dim runs a dummy pass to discover output channels."""
    from misfit.cli.embed_entrypoint import _infer_embed_dim
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    dim = _infer_embed_dim(model, patch_size=32, device=torch.device("cpu"))
    assert isinstance(dim, int)
    assert dim > 0


def test_embed_entry_encoder_fn_is_called(tmp_path):
    """encoder_fn closure body is exercised when Embedder runs for real."""
    from misfit.cli.embed_entrypoint import embed_entry
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.embedding.aggregators.mean_pool import MeanPoolAggregator

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    tiny_model.eval()

    manifest = tmp_path / "index.parquet"
    pd.DataFrame({
        "volume_id": ["vol"],
        "path": ["unused"],
        "p1": [-2.0], "p99": [2.0], "fg_mean": [0.0], "fg_std": [1.0],
    }).to_parquet(manifest, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model": {"name": "swinmae-small", "patch_size": [32, 32, 32], "mask_patch_size": 16, "mask_ratio": 0.75}}'
    )
    output_dir = tmp_path / "out_encoder_fn"

    import misfit.inference.inference_utils as _iutils
    import misfit.embedding.aggregators.aggregator_registry as _areg

    with patch.object(_iutils, "load_checkpoint", return_value={"model": {}}), \
         patch.object(_iutils, "build_model_from_checkpoint",
                      return_value=tiny_model), \
         patch.object(_areg, "get_aggregator",
                      return_value=MeanPoolAggregator), \
         patch.object(_iutils, "load_and_normalise",
                      return_value=np.zeros((32, 32, 32), dtype=np.float32)):
        embed_entry([
            "--encoder-checkpoint", "best.pt",
            "--config", str(config_path),
            "--index", str(manifest),
            "--output-dir", str(output_dir),
        ])

    assert (output_dir / "vol.npz").exists()


def test_embed_entry_exception_during_processing(tmp_path):
    """Exception in extract_crop_features is caught and reported."""
    from misfit.cli.embed_entrypoint import embed_entry

    manifest = tmp_path / "index.parquet"
    pd.DataFrame({
        "volume_id": ["bad_vol"],
        "path": ["unused"],
        "p1": [-2.0], "p99": [2.0], "fg_mean": [0.0], "fg_std": [1.0],
    }).to_parquet(manifest, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model": {"name": "swinmae-small", "patch_size": [32, 32, 32], "mask_patch_size": 16, "mask_ratio": 0.75}}'
    )
    output_dir = tmp_path / "out_exc"

    import misfit.inference.inference_utils as _iutils
    import misfit.embedding.aggregators.aggregator_registry as _areg
    import misfit.embedding.embedder as _emb_mod

    mock_embedder = MagicMock()
    mock_embedder.extract_crop_features.side_effect = RuntimeError("explode")

    with patch.object(_iutils, "load_checkpoint", return_value={"model": {}}), \
         patch.object(_iutils, "build_model_from_checkpoint",
                      return_value=MagicMock()), \
         patch.object(_areg, "get_aggregator",
                      return_value=MagicMock(return_value=MagicMock())), \
         patch.object(_emb_mod, "Embedder", return_value=mock_embedder), \
         patch.object(_iutils, "load_and_normalise",
                      return_value=np.zeros((32, 32, 32), dtype=np.float32)):
        # must not raise — exception is caught internally
        embed_entry([
            "--encoder-checkpoint", "best.pt",
            "--config", str(config_path),
            "--index", str(manifest),
            "--output-dir", str(output_dir),
        ])

    # .npz not written because processing failed
    assert not (output_dir / "bad_vol.npz").exists()


def test_embed_entry_load_normalise_returns_none(tmp_path):
    """embed_entry skips a volume when load_and_normalise returns None."""
    from misfit.cli.embed_entrypoint import embed_entry

    manifest = tmp_path / "index.parquet"
    pd.DataFrame({
        "volume_id": ["bad"],
        "path": ["missing.nii.gz"],
        "p1": [-2.0], "p99": [2.0], "fg_mean": [0.0], "fg_std": [1.0],
    }).to_parquet(manifest, index=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model": {"name": "swinmae-small", "patch_size": [32, 32, 32], "mask_patch_size": 16, "mask_ratio": 0.75}}'
    )
    output_dir = tmp_path / "out_none"

    import misfit.inference.inference_utils as _iutils_n
    import misfit.embedding.aggregators.aggregator_registry as _areg_n
    import misfit.embedding.embedder as _emb_mod_n

    mock_embedder = MagicMock()
    with patch.object(_iutils_n, "load_checkpoint", return_value={"model": {}}), \
         patch.object(_iutils_n, "build_model_from_checkpoint", return_value=MagicMock()), \
         patch.object(_areg_n, "get_aggregator", return_value=MagicMock(return_value=MagicMock())), \
         patch.object(_emb_mod_n, "Embedder", return_value=mock_embedder), \
         patch.object(_iutils_n, "load_and_normalise", return_value=None):
        embed_entry([
            "--encoder-checkpoint", "best.pt",
            "--config", str(config_path),
            "--index", str(manifest),
            "--output-dir", str(output_dir),
        ])

    # No .npz written and extract_crop_features never called
    assert not (output_dir / "bad.npz").exists()
    mock_embedder.extract_crop_features.assert_not_called()


def test_embed_entry_split_filters_index(tmp_path):
    """embed_entry only processes rows matching --split."""
    from misfit.cli.embed_entrypoint import embed_entry

    config_path = tmp_path / "config.json"
    config_path.write_text(
        '{"model": {"name": "swinmae-small", "patch_size": [32, 32, 32], "mask_patch_size": 16, "mask_ratio": 0.75}}'
    )
    manifest = tmp_path / "index.parquet"
    pd.DataFrame({
        "volume_id": ["train_vol", "val_vol"],
        "path": ["unused", "unused"],
        "split": ["train", "val"],
        "p1": [0.0, 0.0], "p99": [1.0, 1.0],
        "fg_mean": [0.0, 0.0], "fg_std": [1.0, 1.0],
    }).to_parquet(manifest, index=False)

    output_dir = tmp_path / "embeddings"

    import misfit.inference.inference_utils as _iutils
    import misfit.embedding.aggregators.aggregator_registry as _areg
    import misfit.embedding.embedder as _emb_mod

    mock_embedder = MagicMock()
    mock_embedder.extract_crop_features.return_value = (
        np.zeros((2, 16), dtype=np.float32),
        np.zeros((2, 3), dtype=np.float32),
    )

    with patch.object(_iutils, "load_checkpoint", return_value={"model": {}}), \
         patch.object(_iutils, "build_model_from_checkpoint", return_value=MagicMock()), \
         patch.object(_areg, "get_aggregator", return_value=MagicMock(return_value=MagicMock())), \
         patch.object(_emb_mod, "Embedder", return_value=mock_embedder), \
         patch.object(_iutils, "load_and_normalise",
                      return_value=np.zeros((32, 32, 32), dtype=np.float32)):
        embed_entry([
            "--encoder-checkpoint", "best.pt",
            "--config", str(config_path),
            "--index", str(manifest),
            "--output-dir", str(output_dir),
            "--split", "val",
        ])

    # Only val_vol should have been processed (one call to extract_crop_features)
    assert mock_embedder.extract_crop_features.call_count == 1
