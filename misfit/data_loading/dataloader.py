"""DataLoader factory functions for MISFIT training.

Provides get_training_dataloader and get_validation_dataloader. Training
draws a fixed number of samples per epoch from an :class:`EpochSampler`, with
each rank taking a disjoint share; validation uses a DistributedSampler when
distributed=True.

Typical single-node usage::

    from misfit.data_loading.dataloader import get_training_dataloader

    loader = get_training_dataloader("index.parquet", batch_size=4)
    for batch in loader:
        # batch: (B, 1, X, Y, Z) float32 tensor
        ...

Typical multi-node usage (inside a torchrun worker)::

    loader = get_training_dataloader(
        "index.parquet",
        batch_size=4,
        num_workers=8,
        samples_per_epoch=1000,
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )
"""
import os
from multiprocessing import resource_tracker
from pathlib import Path

from torch.utils.data import DataLoader, DistributedSampler, Subset

from misfit.data_loading.dataset import MISFITDataset
from misfit.data_loading.samplers import EpochSampler
from misfit.normalization.normalization_registry import DEFAULT_NORMALIZATION_POLICY


def seal_resource_tracker_fd(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that keeps the resource tracker exitable.

    Under the ``spawn`` start method (the default on macOS), each worker
    inherits the write end of the multiprocessing resource tracker's pipe,
    marked inheritable. With the ``file_system`` sharing strategy (the only
    one PyTorch supports on macOS), the worker's first shared-memory batch
    launches a ``torch_shm_manager`` daemon via fork + exec, which inherits
    that fd and outlives the worker for as long as the main process stays
    connected to it. At interpreter exit, ``ResourceTracker.__del__``
    (added in recent 3.12 / 3.13 patch releases) closes the main process's end and blocks in
    ``waitpid`` for the tracker, which never sees EOF because the daemon
    still holds a write end. The daemon in turn waits on the main process,
    so ``misfit_train`` hangs after training. Marking the fd close-on-exec
    keeps it out of exec'd daemons; the worker itself still uses it normally.

    A no-op when the worker has no tracker fd (e.g. ``fork`` on Linux, where
    the default ``file_descriptor`` strategy never starts a tracker).
    """
    fd = getattr(resource_tracker._resource_tracker, "_fd", None)
    if fd is not None:
        os.set_inheritable(fd, False)


def get_training_dataloader(
    index_path: str | Path,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    batch_size: int = 2,
    num_workers: int = 8,
    samples_per_epoch: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    target_spacing: tuple[float, float, float] | None = None,
    normalization_policy: str = DEFAULT_NORMALIZATION_POLICY,
) -> DataLoader:
    """Build a training DataLoader with augmentation enabled.

    Each epoch draws ``samples_per_epoch`` samples for this rank from an
    :class:`EpochSampler`, independent of how many volumes the index holds;
    ranks draw disjoint samples. Call ``loader.sampler.set_epoch(epoch)`` at
    the start of each epoch (with ``start=`` to resume mid-epoch).

    Args:
        index_path: Path to the Parquet metadata index from misfit_index.
        patch_size: Spatial crop size (X, Y, Z) fed to MISFITDataset and
            the model. All dims must be divisible by 32. Defaults to (96, 96, 96).
        batch_size: Number of volumes per batch per GPU. Defaults to 2.
        num_workers: Subprocesses for data loading. 4 - 16 is typical for NFS
            or local NVMe; 16 - 32 for Lustre/GPFS HPC filesystems.
            Defaults to 8.
        samples_per_epoch: Samples this rank draws per epoch (a multiple of
            batch_size). None draws one pass over the train split, shared
            across ranks. Defaults to None.
        rank: This process's rank under torchrun. Defaults to 0.
        world_size: Number of torchrun processes. Defaults to 1.
        seed: Random seed for the sampler. Defaults to 42.
        target_spacing: (x, y, z) spacing to resample volumes to before
            cropping, or None for native resolution. Defaults to None.
        normalization_policy: Registered normalization policy name.
            Defaults to ``clip_and_z_score``.

    Returns:
        Configured PyTorch DataLoader yielding (B, 1, X, Y, Z) tensors.
    """
    dataset = MISFITDataset(
        index_path=index_path,
        patch_size=patch_size,
        augment=True,
        split="train",
        target_spacing=target_spacing,
        normalization_policy=normalization_policy,
    )
    sampler = EpochSampler(
        len(dataset),
        samples_per_epoch=samples_per_epoch,
        rank=rank,
        world_size=world_size,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        # Keep worker processes alive across epochs to amortize NIfTI import
        # overhead — nibabel has non-trivial module-load cost per worker.
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seal_resource_tracker_fd,
        drop_last=True,
    )


def get_validation_dataloader(
    index_path: str | Path,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    batch_size: int = 1,
    num_workers: int = 4,
    distributed: bool = False,
    seed: int = 42,
    target_spacing: tuple[float, float, float] | None = None,
    max_volumes: int | None = None,
    normalization_policy: str = DEFAULT_NORMALIZATION_POLICY,
) -> DataLoader:
    """Build a validation DataLoader with deterministic center crop.

    No augmentation is applied. Volumes are center-cropped to patch_size
    for consistent evaluation across epochs. With ``max_volumes``, only the
    first ``max_volumes`` val volumes (in index order) are used, so the val
    loss is computed on the same fixed subset every time and stays
    comparable across runs on corpora of different sizes.

    Args:
        index_path: Path to the Parquet metadata index from misfit_index.
        patch_size: Spatial crop size (X, Y, Z). Defaults to (96, 96, 96).
        batch_size: Number of volumes per batch. Defaults to 1.
        num_workers: Subprocesses for data loading. Defaults to 4.
        distributed: If True, attach a DistributedSampler. Defaults to False.
        seed: Random seed for DistributedSampler. Defaults to 42.
        target_spacing: (x, y, z) spacing to resample volumes to before
            cropping, or None for native resolution. Defaults to None.
        max_volumes: Use at most this many val volumes, or None for all.
            Defaults to None.
        normalization_policy: Registered normalization policy name.
            Defaults to ``clip_and_z_score``.

    Returns:
        Configured PyTorch DataLoader yielding (B, 1, X, Y, Z) tensors.
    """
    dataset = MISFITDataset(
        index_path=index_path,
        patch_size=patch_size,
        augment=False,
        split="val",
        target_spacing=target_spacing,
        normalization_policy=normalization_policy,
    )
    if max_volumes is not None and max_volumes < len(dataset):
        dataset = Subset(dataset, range(max_volumes))

    sampler: DistributedSampler | None = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=False, seed=seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seal_resource_tracker_fd,
        drop_last=False,
    )
