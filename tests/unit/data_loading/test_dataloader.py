"""Tests for misfit.data_loading.dataloader."""
import json
import os
from multiprocessing import resource_tracker

import nibabel as nib
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler

from misfit.data_loading.dataloader import (
    get_training_dataloader,
    get_validation_dataloader,
    seal_resource_tracker_fd,
)
from misfit.data_loading.samplers import EpochSampler


def _write_index_and_nifti(tmp_path, split="train"):
    data = np.random.randn(32, 32, 32).astype(np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    nifti_path = tmp_path / "vol.nii.gz"
    nib.save(img, str(nifti_path))

    df = pd.DataFrame([{
        "volume_id": "vol",
        "path": str(nifti_path),
        "split": split,
        "shape_d": 32, "shape_h": 32, "shape_w": 32,
        "spacing_d": 1.0, "spacing_h": 1.0, "spacing_w": 1.0,
        "affine": json.dumps(np.eye(4).tolist()),
        "fg_x_start": 0, "fg_x_end": 31,
        "fg_y_start": 0, "fg_y_end": 31,
        "fg_z_start": 0, "fg_z_end": 31,
        "p1": -2.0, "p99": 2.0, "fg_mean": 0.0, "fg_std": 1.0,
    }])
    index_path = tmp_path / "index.parquet"
    df.to_parquet(index_path, index=False)
    return index_path


def test_training_dataloader_returns_dataloader(tmp_path):
    index_path = _write_index_and_nifti(tmp_path)
    loader = get_training_dataloader(
        index_path, patch_size=(32, 32, 32), batch_size=1, num_workers=0
    )
    assert isinstance(loader, DataLoader)


def test_validation_dataloader_returns_dataloader(tmp_path):
    index_path = _write_index_and_nifti(tmp_path)
    loader = get_validation_dataloader(
        index_path, patch_size=(32, 32, 32), batch_size=1, num_workers=0
    )
    assert isinstance(loader, DataLoader)


def test_training_dataloader_uses_epoch_sampler(tmp_path):
    index_path = _write_index_and_nifti(tmp_path)
    loader = get_training_dataloader(
        index_path, patch_size=(32, 32, 32), batch_size=2, num_workers=0,
        samples_per_epoch=8, rank=1, world_size=2, seed=3,
    )
    assert isinstance(loader.sampler, EpochSampler)
    assert (loader.sampler.rank, loader.sampler.world_size) == (1, 2)
    assert loader.sampler.seed == 3
    # The epoch length comes from samples_per_epoch, not the dataset size.
    assert len(loader) == 4


def test_validation_dataloader_no_shuffle(tmp_path):
    index_path = _write_index_and_nifti(tmp_path)
    loader = get_validation_dataloader(
        index_path, patch_size=(32, 32, 32), batch_size=1, num_workers=0,
    )
    assert loader.drop_last is False


def test_validation_dataloader_distributed_attaches_sampler(tmp_path):
    """distributed=True creates a DistributedSampler (line 122)."""
    from unittest.mock import MagicMock, patch

    index_path = _write_index_and_nifti(tmp_path, split="val")
    mock_sampler = MagicMock(spec=DistributedSampler)

    with patch(
        "misfit.data_loading.dataloader.DistributedSampler",
        return_value=mock_sampler,
    ) as MockDS:
        loader = get_validation_dataloader(
            index_path, patch_size=(32, 32, 32), batch_size=1,
            num_workers=0, distributed=True,
        )

    MockDS.assert_called_once()
    assert loader.sampler is mock_sampler


def test_training_dataloader_filters_train_split(tmp_path):
    """Training dataloader only sees rows with split='train'."""
    index_path = _write_index_and_nifti(tmp_path, split="train")
    loader = get_training_dataloader(
        index_path, patch_size=(32, 32, 32), batch_size=1, num_workers=0,
    )
    assert len(loader.dataset) == 1


def test_training_dataset_excludes_val_split(tmp_path):
    """MISFITDataset with split='train' excludes rows labelled 'val'."""
    from misfit.data_loading.dataset import MISFITDataset
    index_path = _write_index_and_nifti(tmp_path, split="val")
    ds = MISFITDataset(index_path, patch_size=(32, 32, 32), split="train")
    assert len(ds) == 0


def test_validation_dataloader_filters_val_split(tmp_path):
    """Validation dataloader only sees rows with split='val'."""
    index_path = _write_index_and_nifti(tmp_path, split="val")
    loader = get_validation_dataloader(
        index_path, patch_size=(32, 32, 32), batch_size=1, num_workers=0,
    )
    assert len(loader.dataset) == 1


def test_dataloaders_seal_resource_tracker_fd_in_workers(tmp_path):
    """Both loaders run seal_resource_tracker_fd as their worker_init_fn."""
    index_path = _write_index_and_nifti(tmp_path)
    for factory in (get_training_dataloader, get_validation_dataloader):
        loader = factory(index_path, patch_size=(32, 32, 32), num_workers=0)
        assert loader.worker_init_fn is seal_resource_tracker_fd


def test_seal_resource_tracker_fd_marks_fd_close_on_exec(monkeypatch):
    """The worker's (inheritable) tracker fd is made non-inheritable."""
    read_fd, write_fd = os.pipe()
    try:
        os.set_inheritable(write_fd, True)
        monkeypatch.setattr(resource_tracker._resource_tracker, "_fd", write_fd)
        seal_resource_tracker_fd(0)
        assert not os.get_inheritable(write_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_seal_resource_tracker_fd_noop_without_tracker(monkeypatch):
    """No tracker fd (e.g. fork start method) → nothing to do, no error."""
    monkeypatch.setattr(resource_tracker._resource_tracker, "_fd", None)
    seal_resource_tracker_fd(0)


def _write_val_index(tmp_path, n):
    """Index with n val rows pointing at one NIfTI."""
    index_path = _write_index_and_nifti(tmp_path, split="val")
    df = pd.read_parquet(index_path)
    df = pd.concat([df.assign(volume_id=f"v{i}") for i in range(n)], ignore_index=True)
    df.to_parquet(index_path, index=False)
    return index_path


def test_validation_dataloader_caps_at_max_volumes(tmp_path):
    """max_volumes keeps the first N val volumes, in index order."""
    index_path = _write_val_index(tmp_path, 5)
    loader = get_validation_dataloader(
        index_path, patch_size=(32, 32, 32), num_workers=0, max_volumes=3,
    )
    assert list(loader.dataset.indices) == [0, 1, 2]


def test_validation_dataloader_max_volumes_above_split_size_keeps_all(tmp_path):
    index_path = _write_val_index(tmp_path, 2)
    loader = get_validation_dataloader(
        index_path, patch_size=(32, 32, 32), num_workers=0, max_volumes=10,
    )
    assert len(loader.dataset) == 2
