"""Tests for misfit.training.trainers.mae_trainer.MAETrainer.

These tests mock heavy operations (distributed init, data loading) and use
CPU-only tiny models, following the same FakeDist / FakeScaler / DummyDDP
patterns used in MIST's test_base_trainer.py.
"""
import argparse
import json
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

    def no_sync(self):
        from contextlib import nullcontext
        return nullcontext()



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
    # BF16 autocast checks is_bf16_supported() on __init__; return True so no CUDA init.
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True, raising=False)
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
        model="swinunetr-small",
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
        gradient_accumulation_steps=1,
        bucket_cap_mb=200,
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
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    criterion = trainer._build_loss()
    assert isinstance(criterion, MaskedMSELoss)


def test_trainer_build_loss_normalized_mse(tmp_path):
    from misfit.loss_functions.reconstruction.normalized_mse import NormalizedMaskedMSELoss
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    args.loss = "normalized_masked_mse"
    trainer = MAETrainer(args)
    criterion = trainer._build_loss()
    assert isinstance(criterion, NormalizedMaskedMSELoss)


def test_trainer_build_optimizer(tmp_path):
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = trainer._build_optimizer(model)
    assert isinstance(opt, torch.optim.Adam)


def test_trainer_build_scheduler(tmp_path):
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = trainer._build_optimizer(model)
    sched = trainer._build_scheduler(opt)
    assert sched is not None


def test_trainer_save_and_load_checkpoint(tmp_path):
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    ckpt_path = tmp_path / "test_ckpt.pt"

    trainer._save_checkpoint(model, opt, sched, 1, 10, 0.5, ckpt_path)
    assert ckpt_path.exists()

    start_epoch, global_step, best_val_loss = trainer._load_checkpoint(
        model, opt, sched, ckpt_path
    )
    assert start_epoch == 1
    assert global_step == 10
    assert best_val_loss == pytest.approx(0.5)


def test_trainer_load_checkpoint_nonexistent(tmp_path):
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    start, step, best = trainer._load_checkpoint(
        model, opt, sched, tmp_path / "nonexistent.pt"
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
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    model.to(torch.device("cpu"))
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch = {"image": torch.randn(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
    loss_val = trainer._training_step(model, batch, criterion, optimizer)
    assert isinstance(loss_val, float)
    assert loss_val >= 0


def test_trainer_validation_step(tmp_path):
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    model.eval()
    criterion = MaskedMSELoss()

    batch = {"image": torch.randn(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
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



def test_trainer_train_runs_single_epoch(tmp_path):
    """train() completes one epoch without error using mocked cuda calls."""
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    args.epochs = 1
    args.resume = False

    # Tiny model on CPU
    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    tiny_model.to(torch.device("cpu"))

    dummy_batch = {"image": torch.randn(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
    mock_loader = [dummy_batch]

    with patch("torch.cuda.set_device"), \
         patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               autospec=True, return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               autospec=True, return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               autospec=True, return_value=mock_loader):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()

    results_dir = Path(args.results)
    assert (results_dir / "checkpoints" / "checkpoint.pt").exists()


def test_trainer_train_with_resume(tmp_path):
    """train() with resume=True loads checkpoint and continues."""
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    args.epochs = 1
    args.resume = True

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    tiny_model.to(torch.device("cpu"))
    dummy_batch = {"image": torch.randn(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
    mock_loader = [dummy_batch]

    with patch("torch.cuda.set_device"), \
         patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               autospec=True, return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               autospec=True, return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               autospec=True, return_value=mock_loader):
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

    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    args.epochs = 1

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)

    dummy_batch = {"image": torch.randn(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}

    # Mock loader needs a sampler with set_epoch (DistributedSampler contract)
    mock_sampler = MagicMock()
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([dummy_batch]))
    mock_loader.__len__ = MagicMock(return_value=1)
    mock_loader.sampler = mock_sampler

    with patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               autospec=True, return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               autospec=True, return_value=mock_loader):
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
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    dummy_batch = {"image": torch.zeros(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([dummy_batch]))
    mock_loader.__len__ = MagicMock(return_value=1)
    mock_loader.sampler = MagicMock()

    with patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               autospec=True, return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               autospec=True, return_value=mock_loader):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()

    config_path = Path(args.results) / "config.json"
    assert config_path.exists()
    config = json.loads(config_path.read_text())
    assert config["model"]["architecture"] == "swinunetr-small"
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


def test_train_config_guard_raises_on_non_main_rank(tmp_path):
    """The precondition guard runs on every rank, not just rank 0.

    A non-main rank must also raise (before distributed init) so a stale output
    dir fails the whole job fast instead of leaving rank 0 aborted while the
    other ranks hang on NCCL rendezvous.
    """
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text("{}")

    trainer = MAETrainer(args)
    # Simulate a worker rank in a multi-GPU job.
    trainer.rank = 2
    trainer.is_main = False
    trainer.is_distributed = True

    with pytest.raises(RuntimeError, match="config.json"):
        trainer.train()


def test_train_overwrite_ignores_existing_config(tmp_path):
    """--overwrite allows training to proceed even with an existing config.json."""
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    args.overwrite = True
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    old_config = results_dir / "config.json"
    old_config.write_text('{"model": {"architecture": "old"}}')

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    dummy_batch = {"image": torch.zeros(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
    mock_loader = MagicMock()
    mock_loader.__iter__ = MagicMock(return_value=iter([dummy_batch]))
    mock_loader.__len__ = MagicMock(return_value=1)
    mock_loader.sampler = MagicMock()

    with patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               autospec=True, return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               autospec=True, return_value=mock_loader):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()  # must not raise

    # Config should be overwritten with current args
    config = json.loads(old_config.read_text())
    assert config["model"]["architecture"] == "swinunetr-small"


def test_validate_resume_raises_on_model_change(tmp_path):
    """_validate_resume raises ValueError when model name changes."""
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)

    saved_config = {
        "model": {"architecture": "swinunetr-base", "patch_size": [32, 32, 32],
                  "mask_patch_size": 16},
        "training": {},
    }
    with pytest.raises(ValueError, match="model.architecture"):
        trainer._validate_resume(saved_config)


def test_validate_resume_raises_on_patch_size_change(tmp_path):
    """_validate_resume raises ValueError when patch_size changes."""
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)

    saved_config = {
        "model": {"architecture": "swinunetr-small", "patch_size": [96, 96, 96],
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
        "model": {"architecture": "swinunetr-small", "patch_size": [32, 32, 32],
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
    assert config["model"]["architecture"] == "swinunetr-small"
    assert config["model"]["patch_size"] == [32, 32, 32]
    assert config["training"]["seed"] == 42
    assert config["training"]["amp"] is True
    assert "amp_dtype" not in config["training"]
    assert isinstance(config["evaluation"], dict)
    assert all(isinstance(v, dict) for v in config["evaluation"].values())


def test_bf16_optimizer_uses_standard_epsilon(tmp_path):
    """_build_optimizer always uses standard epsilon (BF16 has float32's dynamic range)."""
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainer_constants import tc
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    opt = trainer._build_optimizer(model)
    for pg in opt.param_groups:
        assert pg["eps"] == tc.NO_AMP_EPS


def test_train_resume_reads_and_validates_config(tmp_path):
    """--resume with a compatible config.json calls _validate_resume (line 404)."""
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    args.resume = True

    # Write a compatible config.json into the results dir first.
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)
    dummy_batch = {"image": torch.zeros(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
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
               autospec=True, return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               autospec=True, return_value=mock_loader):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()  # must not raise; config is compatible


# ---------------------------------------------------------------------------
# Gradient accumulation and DDP bucket tuning
# ---------------------------------------------------------------------------

def test_build_model_passes_bucket_cap_mb(tmp_path, monkeypatch):
    """_build_model forwards bucket_cap_mb to DDP when distributed."""
    import misfit.training.trainers.mae_trainer as mt

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")

    captured = {}

    class CapturingDDP(DummyDDP):
        def __init__(self, module, device_ids=None, bucket_cap_mb=None, **kwargs):
            super().__init__(module, device_ids=device_ids)
            captured["bucket_cap_mb"] = bucket_cap_mb

    monkeypatch.setattr(mt, "DDP", CapturingDDP)

    from misfit.training.trainers.mae_trainer import MAETrainer
    args = _make_args(tmp_path)
    args.bucket_cap_mb = 512
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")
    trainer._build_model()

    assert captured["bucket_cap_mb"] == 512


def test_training_step_skips_optimizer_on_non_last_accum(tmp_path):
    """_training_step does not call optimizer.step() when is_last_accum=False."""
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                    mask_patch_size=16, mask_ratio=0.75)
    criterion = MaskedMSELoss()
    optimizer = MagicMock(wraps=torch.optim.Adam(model.parameters(), lr=1e-4))

    batch = {"image": torch.randn(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
    optimizer.zero_grad()  # caller's responsibility
    trainer._training_step(model, batch, criterion, optimizer,
                           window_size=2, is_last_accum=False)

    optimizer.step.assert_not_called()


def test_training_step_scales_loss_by_window_size(tmp_path):
    """Gradient norm halves when window_size doubles (loss is divided by it)."""
    from misfit.loss_functions.reconstruction.masked_mse import MaskedMSELoss
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    trainer = MAETrainer(args)
    trainer.device = torch.device("cpu")

    def _grad_norm_after_step(window_size):
        torch.manual_seed(0)
        model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                        mask_patch_size=16, mask_ratio=0.75)
        criterion = MaskedMSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        batch = {"image": torch.randn(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
        optimizer.zero_grad()
        trainer._training_step(model, batch, criterion, optimizer,
                               window_size=window_size, is_last_accum=False)
        return sum(
            p.grad.norm().item() ** 2
            for p in model.parameters()
            if p.grad is not None
        ) ** 0.5

    norm_1 = _grad_norm_after_step(1)
    norm_2 = _grad_norm_after_step(2)
    assert abs(norm_1 / norm_2 - 2.0) < 0.01


def test_build_config_includes_accumulation_fields(tmp_path):
    """_build_config records gradient_accumulation_steps and bucket_cap_mb."""
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    args.gradient_accumulation_steps = 4
    args.bucket_cap_mb = 400
    trainer = MAETrainer(args)
    config = trainer._build_config()

    assert config["training"]["gradient_accumulation_steps"] == 4
    assert config["training"]["bucket_cap_mb"] == 400


def test_train_partial_trailing_window_steps_once(tmp_path):
    """A trailing partial window (3 batches, accum=2) still produces a final
    synced optimizer step — never left dangling under no_sync()."""
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    args.gradient_accumulation_steps = 2
    args.epochs = 1

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)

    dummy_batch = {"image": torch.randn(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
    # 3 micro-batches, accum=2 → one full window (batches 0-1) + a trailing
    # partial window (batch 2). Both must produce an optimizer step.
    mock_loader = [dummy_batch, dummy_batch, dummy_batch]

    step_calls = {"n": 0}
    original_step = torch.optim.Adam.step

    def counting_step(self_opt, *a, **kw):
        step_calls["n"] += 1
        return original_step(self_opt, *a, **kw)

    with patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               autospec=True, return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               autospec=True, return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               autospec=True, return_value=mock_loader), \
         patch.object(torch.optim.Adam, "step", counting_step):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()

    # Full window (batches 0-1) + trailing window (batch 2) = 2 optimizer steps.
    assert step_calls["n"] == 2


def test_train_gradient_accumulation_two_steps(tmp_path):
    """With accum_steps=2 and a 2-batch loader, exactly 1 optimizer step fires."""
    from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE
    from misfit.training.trainers.mae_trainer import MAETrainer

    args = _make_args(tmp_path)
    args.gradient_accumulation_steps = 2
    args.epochs = 1

    tiny_model = SwinMAE(in_channels=1, feature_size=12, img_size=(32, 32, 32),
                         mask_patch_size=16, mask_ratio=0.75)

    dummy_batch = {"image": torch.randn(1, 1, 32, 32, 32), "spacing": torch.ones(1, 3)}
    mock_loader = [dummy_batch, dummy_batch]  # exactly 2 micro-batches

    step_count = {"n": 0}
    original_step = torch.optim.Adam.step

    def counting_step(self_opt, *a, **kw):
        step_count["n"] += 1
        return original_step(self_opt, *a, **kw)

    with patch("misfit.training.trainers.mae_trainer.get_model_from_registry",
               autospec=True, return_value=tiny_model), \
         patch("misfit.training.trainers.mae_trainer.get_training_dataloader",
               autospec=True, return_value=mock_loader), \
         patch("misfit.training.trainers.mae_trainer.get_validation_dataloader",
               autospec=True, return_value=mock_loader), \
         patch.object(torch.optim.Adam, "step", counting_step):
        trainer = MAETrainer(args)
        trainer.device = torch.device("cpu")
        trainer.train()

    # 2 micro-batches / accum_steps=2 → exactly 1 optimizer step during training
    assert step_count["n"] == 1


def test_warn_if_underutilising_gpus_fires_for_multi_gpu_no_torchrun(tmp_path, monkeypatch):
    """Warns when >1 GPU is visible but WORLD_SIZE==1 (no torchrun)."""
    import misfit.training.trainers.mae_trainer as mt

    trainer = mt.MAETrainer(_make_args(tmp_path))
    trainer.is_distributed = False  # WORLD_SIZE == 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    warn = MagicMock()
    monkeypatch.setattr(mt, "print_warning", warn)
    trainer._warn_if_underutilising_gpus()

    warn.assert_called_once()
    msg = warn.call_args.args[0]
    assert "torchrun" in msg and "--nproc_per_node=4" in msg


def test_warn_if_underutilising_gpus_silent_when_distributed(tmp_path, monkeypatch):
    """No warning under torchrun (is_distributed) or with a single visible GPU."""
    import misfit.training.trainers.mae_trainer as mt

    trainer = mt.MAETrainer(_make_args(tmp_path))
    warn = MagicMock()
    monkeypatch.setattr(mt, "print_warning", warn)

    # Distributed run: never warn even with many GPUs.
    trainer.is_distributed = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    trainer._warn_if_underutilising_gpus()

    # Single visible GPU: nothing to warn about.
    trainer.is_distributed = False
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    trainer._warn_if_underutilising_gpus()

    warn.assert_not_called()
