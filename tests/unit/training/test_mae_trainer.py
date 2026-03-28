"""Tests for misfit.training.trainers.mae_trainer.MAETrainer.

These tests mock heavy operations (distributed init, data loading) and use
CPU-only tiny models, following the same FakeDist / FakeScaler / DummyDDP
patterns used in MIST's test_base_trainer.py.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

import misfit.models  # noqa — trigger registrations


# ---------------------------------------------------------------------------
# Shared mock helpers (ported from MIST's test_base_trainer.py patterns)
# ---------------------------------------------------------------------------

class DummyDDP(nn.Module):
    """Passes through to the wrapped module; no NCCL needed."""

    def __init__(self, module, device_ids=None, **kwargs):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class FakeScaler:
    """Mock GradScaler that runs real backward but tracks AMP method calls."""

    def __init__(self):
        self._scaled = 0
        self._unscaled = 0
        self._stepped = 0
        self._updated = 0
        self._loss = None

    def scale(self, loss):
        self._scaled += 1
        self._loss = loss
        return self          # caller chains .backward()

    def backward(self):
        self._loss.backward()

    def unscale_(self, optimizer):
        self._unscaled += 1

    def step(self, optimizer):
        optimizer.step()
        self._stepped += 1

    def update(self):
        self._updated += 1

    def state_dict(self):
        return {"scale": 65536.0}

    def load_state_dict(self, sd):
        pass


@pytest.fixture()
def fake_dist(monkeypatch):
    """Replace torch.distributed in the mae_trainer module with FakeDist."""
    import misfit.training.trainers.mae_trainer as mt

    calls = {"init": 0, "destroy": 0, "barrier": 0, "all_reduce": 0}

    class FakeDist:
        _initialized = False

        @staticmethod
        def init_process_group(backend):
            calls["init"] += 1
            FakeDist._initialized = True

        @staticmethod
        def destroy_process_group():
            calls["destroy"] += 1
            FakeDist._initialized = False

        @staticmethod
        def barrier():
            calls["barrier"] += 1

        @staticmethod
        def all_reduce(t, op=None):
            calls["all_reduce"] += 1
            # Simulate sum across 2 ranks: double the value in-place
            t.mul_(2)

        ReduceOp = SimpleNamespace(SUM=0)

    monkeypatch.setattr(mt, "dist", FakeDist)
    return calls


@pytest.fixture(autouse=True)
def patch_cuda(monkeypatch):
    """Make CUDA calls no-ops so the test suite runs CPU-only."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None, raising=False)
    # .to(device) returns self — model stays on CPU
    monkeypatch.setattr(nn.Module, "to", lambda self, *a, **k: self, raising=False)


def _make_args(tmp_path: Path, patch_size=(32, 32, 32)) -> argparse.Namespace:
    """Build minimal MAETrainer args."""
    # Create dummy parquet index
    data = np.random.randn(*patch_size).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nifti_path = tmp_path / "vol.nii.gz"
    nib.save(img, str(nifti_path))

    df = pd.DataFrame([{
        "volume_id": "vol",
        "path": str(nifti_path),
        "split": "train",
        "shape_d": patch_size[0], "shape_h": patch_size[1], "shape_w": patch_size[2],
        "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
        "affine": json.dumps(np.eye(4).tolist()),
        "fg_x_start": 0, "fg_x_end": patch_size[0]-1,
        "fg_y_start": 0, "fg_y_end": patch_size[1]-1,
        "fg_z_start": 0, "fg_z_end": patch_size[2]-1,
        "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0,
    }])
    # Include a val row so the validation dataloader is non-empty too.
    val_row = df.iloc[0].to_dict()
    val_row["volume_id"] = "vol_val"
    val_row["split"] = "val"
    df = pd.concat([df, pd.DataFrame([val_row])], ignore_index=True)

    idx = tmp_path / "index.parquet"
    df.to_parquet(idx, index=False)

    return argparse.Namespace(
        model="swinmae-small",
        patch_size=list(patch_size),
        mask_patch_size=16,
        mask_ratio=0.75,
        loss="masked_mse",
        optimizer="adam",
        learning_rate=1e-4,
        weight_decay=0.0,
        lr_scheduler="cosine",
        warmup_epochs=0,
        epochs=1,
        batch_size=1,
        num_cpu_workers=0,
        seed=42,
        resume=False,
        overwrite=False,
        results=str(tmp_path / "results"),
        index=str(idx),
        amp_dtype="fp16",
    )


@pytest.fixture(autouse=True)
def set_env_vars(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")


def test_trainer_is_main_process(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    assert trainer.is_main is True
    assert trainer.rank == 0
    assert trainer.world_size == 1


def test_trainer_build_model(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)

    with patch("torch.cuda.set_device"):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        model = trainer._build_model()
    assert model is not None


def test_trainer_build_loss_masked_mse(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    criterion = trainer._build_loss()
    assert isinstance(criterion, MaskedMSELoss)


def test_trainer_build_loss_normalized_mse(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.loss_functions.reconstruction.normalized_mse import NormalizedMaskedMSELoss
    args = _make_args(tmp_path)
    args.loss = "normalized_masked_mse"
    trainer = MAETrainer(args)
    criterion = trainer._build_loss()
    assert isinstance(criterion, NormalizedMaskedMSELoss)


def test_trainer_build_optimizer(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = trainer._build_optimizer(model)
    assert isinstance(opt, torch.optim.Adam)


def test_trainer_build_scheduler(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = trainer._build_optimizer(model)
    sched = trainer._build_scheduler(opt)
    assert sched is not None


def test_trainer_save_and_load_checkpoint(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    ckpt_path = tmp_path / "test_ckpt.pt"

    trainer._save_checkpoint(model, opt, sched, None, 1, 10, 0.5, ckpt_path)
    assert ckpt_path.exists()

    start_epoch, global_step, best_val_loss = trainer._load_checkpoint(
        model, opt, sched, None, ckpt_path
    )
    assert start_epoch == 1
    assert global_step == 10
    assert best_val_loss == pytest.approx(0.5)


def test_trainer_load_checkpoint_nonexistent(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    start, step, best = trainer._load_checkpoint(
        model, opt, sched, None, tmp_path / "nonexistent.pt"
    )
    assert start == 0
    assert step == 0
    assert best == float("inf")


def test_trainer_aggregate_loss_non_distributed(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.is_distributed = False
    result = trainer._aggregate_loss(1.5)
    assert result == pytest.approx(1.5)


def test_trainer_training_step(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    model.to(torch.device("cpu"))
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch = torch.randn(1, 1, 32, 32, 32)
    loss_val = trainer._training_step(model, batch, criterion, optimizer, scaler=None)
    assert isinstance(loss_val, float)
    assert loss_val >= 0


def test_trainer_validation_step(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    model.eval()
    criterion = MaskedMSELoss()

    batch = torch.randn(1, 1, 32, 32, 32)
    loss_val = trainer._validation_step(model, batch, criterion)
    assert isinstance(loss_val, float)
    assert loss_val >= 0


def test_trainer_enable_cudnn_optimisations(tmp_path):
    """_enable_cudnn_optimisations sets cuda flags (no-op on CPU, just no error)."""
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer._enable_cudnn_optimisations()  # should not raise


def test_trainer_make_progress(tmp_path):
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    progress = trainer._make_progress()
    assert progress is not None


def test_trainer_load_checkpoint_with_scaler(tmp_path):
    """_load_checkpoint loads scaler state when scaler is provided and key exists."""
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    # Use a real GradScaler (even on CPU it serialises)
    scaler = torch.amp.GradScaler("cpu")
    ckpt_path = tmp_path / "scaler_ckpt.pt"

    trainer._save_checkpoint(model, opt, sched, scaler, 2, 20, 0.3, ckpt_path)
    start, step, best = trainer._load_checkpoint(model, opt, sched, scaler, ckpt_path)
    assert start == 2
    assert step == 20


def test_trainer_train_runs_single_epoch(tmp_path):
    """train() completes one epoch without error using mocked cuda calls."""
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss

    args = _make_args(tmp_path)
    args.epochs = 1
    args.resume = False

    # Tiny model on CPU
    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    tiny_model.to(torch.device("cpu"))

    dummy_batch = torch.randn(1, 1, 32, 32, 32)
    mock_loader = [dummy_batch]

    with patch("torch.cuda.set_device"), \
         patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               return_value=mock_loader):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()

    results_dir = Path(args.results)
    assert (results_dir / "checkpoints" / "checkpoint.pt").exists()


def test_trainer_train_with_resume(tmp_path):
    """train() with resume=True loads checkpoint and continues."""
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

    args = _make_args(tmp_path)
    args.epochs = 1
    args.resume = True

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    tiny_model.to(torch.device("cpu"))
    dummy_batch = torch.randn(1, 1, 32, 32, 32)
    mock_loader = [dummy_batch]

    with patch("torch.cuda.set_device"), \
         patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               return_value=mock_loader):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        # Checkpoint doesn't exist → _load_checkpoint returns (0, 0, inf) → no error
        trainer.train()

    assert True  # just confirm no exception


# ---------------------------------------------------------------------------
# Tests for previously-uncovered CUDA / DDP / AMP paths
# ---------------------------------------------------------------------------

def test_setup_distributed_calls_init_process_group(tmp_path, monkeypatch, fake_dist):
    """_setup_distributed calls dist.init_process_group when is_distributed=True (line 78)."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    assert trainer.is_distributed is True

    trainer._setup_distributed()
    assert fake_dist["init"] == 1


def test_build_model_distributed_wraps_with_ddp(tmp_path, monkeypatch, fake_dist):
    """_build_model wraps with SyncBatchNorm + DDP when is_distributed=True (lines 99-100)."""
    import misfit.training.trainers.mae_trainer as mt

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    # Replace DDP with our no-NCCL DummyDDP
    monkeypatch.setattr(mt, "DDP", DummyDDP)

    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = trainer._build_model()
    # Model should now be wrapped in DummyDDP
    assert isinstance(model, DummyDDP)


def test_training_step_with_scaler_calls_amp_methods(tmp_path):
    """_training_step uses scaler.scale/unscale_/step/update when scaler is provided (lines 166-170)."""
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = FakeScaler()

    batch = torch.randn(1, 1, 32, 32, 32)
    loss_val = trainer._training_step(model, batch, criterion, optimizer, scaler=scaler)

    assert isinstance(loss_val, float)
    assert scaler._scaled == 1
    assert scaler._unscaled == 1
    assert scaler._stepped == 1
    assert scaler._updated == 1


def test_aggregate_loss_distributed_all_reduces(tmp_path, monkeypatch, fake_dist):
    """_aggregate_loss calls dist.all_reduce and returns mean when distributed (lines 216-218)."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")
    assert trainer.is_distributed is True

    result = trainer._aggregate_loss(1.0)

    assert fake_dist["all_reduce"] == 1
    # FakeDist doubles the tensor then we divide by world_size=2 → back to 1.0
    assert result == pytest.approx(1.0)


def test_train_distributed_barriers_and_cleanup(tmp_path, monkeypatch, fake_dist):
    """train() calls dist.barrier and dist.destroy_process_group when distributed
    (lines 347, 384, 412-413, 460, 466)."""
    import misfit.training.trainers.mae_trainer as mt

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(mt, "DDP", DummyDDP)

    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

    args = _make_args(tmp_path)
    args.epochs = 1

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)

    dummy_batch = torch.randn(1, 1, 32, 32, 32)

    # Mock loader needs a sampler with set_epoch (DistributedSampler contract)
    mock_sampler = MagicMock()
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([dummy_batch]))
    mock_loader.__len__ = MagicMock(return_value=1)
    mock_loader.sampler = mock_sampler

    with patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               return_value=mock_loader):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()

    # sampler.set_epoch must have been called for the distributed epoch shuffle (line 384)
    mock_sampler.set_epoch.assert_called()

    # barrier called: once after mkdir (line 347), once before val (line 413),
    # once end-of-epoch (line 460) = at least 3 times
    assert fake_dist["barrier"] >= 3

    # destroy_process_group called at cleanup (line 466)
    assert fake_dist["destroy"] == 1


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------

def test_train_writes_config_json(tmp_path):
    """A fresh training run writes config.json to the results dir."""
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

    args = _make_args(tmp_path)
    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    dummy_batch = torch.zeros(1, 1, 32, 32, 32)
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([dummy_batch]))
    mock_loader.__len__ = MagicMock(return_value=1)
    mock_loader.sampler = MagicMock()

    with patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               return_value=mock_loader):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()

    config_path = Path(args.results) / "config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert config["model"]["name"] == "swinmae-small"
    assert config["training"]["epochs"] == 1
    assert "misfit_version" in config


def test_train_raises_if_config_exists_without_flags(tmp_path):
    """Existing config.json without --resume or --overwrite raises RuntimeError."""
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text("{}")

    trainer = MAETrainer(args)
    with pytest.raises(RuntimeError, match="config.json"):
        trainer.train()


def test_train_overwrite_ignores_existing_config(tmp_path):
    """--overwrite allows training to proceed even with an existing config.json."""
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

    args = _make_args(tmp_path)
    args.overwrite = True
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    old_config = results_dir / "config.json"
    old_config.write_text('{"model": {"name": "old"}}')

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    dummy_batch = torch.zeros(1, 1, 32, 32, 32)
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([dummy_batch]))
    mock_loader.__len__ = MagicMock(return_value=1)
    mock_loader.sampler = MagicMock()

    with patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               return_value=mock_loader):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()  # must not raise

    # Config should be overwritten with current args
    config = json.loads(old_config.read_text())
    assert config["model"]["name"] == "swinmae-small"


def test_validate_resume_raises_on_model_change(tmp_path):
    """_validate_resume raises ValueError when model name changes."""
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)

    saved_config = {
        "model": {"name": "swinmae-base", "patch_size": [32, 32, 32],
                  "mask_patch_size": 16},
        "training": {},
    }
    with pytest.raises(ValueError, match="model.name"):
        trainer._validate_resume(saved_config)


def test_validate_resume_raises_on_patch_size_change(tmp_path):
    """_validate_resume raises ValueError when patch_size changes."""
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)

    saved_config = {
        "model": {"name": "swinmae-small", "patch_size": [96, 96, 96],
                  "mask_patch_size": 16},
        "training": {},
    }
    with pytest.raises(ValueError, match="model.patch_size"):
        trainer._validate_resume(saved_config)


def test_validate_resume_warns_on_lr_change(tmp_path):
    """_validate_resume warns (not raises) when learning_rate changes."""
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)

    saved_config = {
        "model": {"name": "swinmae-small", "patch_size": [32, 32, 32],
                  "mask_patch_size": 16},
        "training": {"learning_rate": 9e-4},  # different from args (1e-4)
    }
    with patch("misfit.training.trainers.mae_trainer.print_warning") as mock_warn:
        trainer._validate_resume(saved_config)  # must not raise

    mock_warn.assert_called_once()
    assert "learning_rate" in mock_warn.call_args[0][0]


def test_validate_resume_passes_on_identical_config(tmp_path):
    """_validate_resume does not raise or warn when config matches args exactly."""
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)

    saved_config = trainer._build_config()

    with patch("misfit.training.trainers.mae_trainer.print_warning") as mock_warn:
        trainer._validate_resume(saved_config)  # must not raise or warn

    mock_warn.assert_not_called()


def test_build_config_structure(tmp_path):
    """_build_config returns a dict with the expected top-level keys."""
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    config = trainer._build_config()

    assert set(config.keys()) == {"misfit_version", "data", "model", "training", "evaluation"}
    assert "index" in config["data"]
    assert config["model"]["name"] == "swinmae-small"
    assert config["model"]["patch_size"] == [32, 32, 32]
    assert config["training"]["seed"] == 42
    assert config["training"]["amp"] is True
    assert config["training"]["amp_dtype"] == "fp16"
    assert isinstance(config["evaluation"], dict)
    assert all(isinstance(v, dict) for v in config["evaluation"].values())


def test_bf16_no_grad_scaler(tmp_path):
    """BF16 dtype sets amp_dtype correctly and skips GradScaler."""
    import torch
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    args.amp_dtype = "bf16"
    trainer = MAETrainer(args)
    assert trainer.amp_dtype == "bf16"

    # BF16 config should record the dtype.
    config = trainer._build_config()
    assert config["training"]["amp_dtype"] == "bf16"

    # _build_optimizer should use standard epsilon for BF16.
    from misfit.training.trainer_constants import tc
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = trainer._build_optimizer(model)
    for pg in opt.param_groups:
        assert pg["eps"] == tc.NO_AMP_EPS, "BF16 should use standard epsilon"


def test_fp16_uses_amp_epsilon(tmp_path):
    """FP16 dtype uses inflated optimizer epsilon."""
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.training.trainer_constants import tc
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

    args = _make_args(tmp_path)
    args.amp_dtype = "fp16"
    trainer = MAETrainer(args)
    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = trainer._build_optimizer(model)
    for pg in opt.param_groups:
        assert pg["eps"] == tc.AMP_FP16_EPS, "FP16 should use inflated epsilon"


def test_train_resume_reads_and_validates_config(tmp_path):
    """--resume with a compatible config.json calls _validate_resume (line 404)."""
    from misfit.training.trainers.mae_trainer import MAETrainer
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

    args = _make_args(tmp_path)
    args.resume = True

    # Write a compatible config.json into the results dir first.
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    dummy_batch = torch.zeros(1, 1, 32, 32, 32)
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([dummy_batch]))
    mock_loader.__len__ = MagicMock(return_value=1)
    mock_loader.sampler = MagicMock()

    # Build the config from args so it's guaranteed to be compatible, then
    # write it before train() runs.
    trainer_for_config = MAETrainer(args)
    compatible_config = trainer_for_config._build_config()
    import misfit.utils as _utils
    _utils.write_json_file(results_dir / "config.json", compatible_config)

    with patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               return_value=mock_loader):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()  # must not raise; config is compatible
