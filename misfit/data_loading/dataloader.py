"""DataLoader factory functions for MISFIT training.

Provides get_training_dataloader and get_validation_dataloader. Both support
distributed training via PyTorch's DistributedSampler — pass distributed=True
when running under torchrun for multi-node jobs.

Typical single-node usage::

    from misfit.data_loading.dataloader import get_training_dataloader

    loader = get_training_dataloader("index.parquet", batch_size=4)
    for batch in loader:
        # batch: (B, 1, D, H, W) float32 tensor
        ...

Typical multi-node usage (inside a torchrun worker)::

    loader = get_training_dataloader(
        "index.parquet",
        batch_size=4,
        num_workers=8,
        distributed=True,
    )
"""
from pathlib import Path

from torch.utils.data import DataLoader, DistributedSampler

from misfit.data_loading.dataset import MISFITDataset


def get_training_dataloader(
    index_path: str | Path,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    batch_size: int = 2,
    num_workers: int = 8,
    distributed: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Build a training DataLoader with augmentation enabled.

    When distributed=True, a DistributedSampler is attached so that each
    rank in a torchrun job sees a non-overlapping shard of the dataset. Call
    ``loader.sampler.set_epoch(epoch)`` at the start of each epoch to ensure
    proper shuffling across ranks.

    Args:
        index_path: Path to the Parquet metadata index from misfit_index.
        patch_size: Spatial crop size (D, H, W) fed to MISFITDataset and
            SwinMAE. All dims must be divisible by 32. Defaults to (96, 96, 96).
        batch_size: Number of volumes per batch per GPU. Defaults to 2.
        num_workers: Subprocesses for data loading. 4 - 16 is typical for NFS
            or local NVMe; 16 - 32 for Lustre/GPFS HPC filesystems.
            Defaults to 8.
        distributed: If True, attach a DistributedSampler for multi-node /
            multi-GPU training under torchrun. Defaults to False.
        seed: Random seed passed to DistributedSampler for reproducibility.
            Defaults to 42.

    Returns:
        Configured PyTorch DataLoader yielding (B, 1, D, H, W) tensors.
    """
    dataset = MISFITDataset(
        index_path=index_path,
        patch_size=patch_size,
        augment=True,
        split="train",
    )

    sampler: DistributedSampler | None = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True, seed=seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        # Shuffle only when not using DistributedSampler (which owns shuffling).
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        # Keep worker processes alive across epochs to amortize NIfTI import
        # overhead — nibabel has non-trivial module-load cost per worker.
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True,
    )


def get_validation_dataloader(
    index_path: str | Path,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    batch_size: int = 1,
    num_workers: int = 4,
    distributed: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Build a validation DataLoader with deterministic center crop.

    No augmentation is applied. Volumes are center-cropped to patch_size
    for consistent evaluation across epochs.

    Args:
        index_path: Path to the Parquet metadata index from misfit_index.
        patch_size: Spatial crop size (D, H, W). Defaults to (96, 96, 96).
        batch_size: Number of volumes per batch. Defaults to 1.
        num_workers: Subprocesses for data loading. Defaults to 4.
        distributed: If True, attach a DistributedSampler. Defaults to False.
        seed: Random seed for DistributedSampler. Defaults to 42.

    Returns:
        Configured PyTorch DataLoader yielding (B, 1, D, H, W) tensors.
    """
    dataset = MISFITDataset(
        index_path=index_path,
        patch_size=patch_size,
        augment=False,
        split="val",
    )

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
        drop_last=False,
    )
