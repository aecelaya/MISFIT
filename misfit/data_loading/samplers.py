"""Samplers for step-based MISFIT training."""
import math
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler


class EpochSampler(Sampler[int]):
    """Draws a fixed number of samples per epoch, independent of dataset size.

    MISFIT trains in fixed-length epochs (``--steps-per-epoch``) so the
    learning-rate schedule, validation, and checkpoint cadence don't depend
    on how many volumes are in the corpus. Each epoch draws indices from a
    seeded stream of back-to-back random permutations of the dataset, so
    every volume is drawn about equally often over training, and a volume is
    never repeated within an epoch unless the epoch is longer than the
    dataset.

    The stream depends only on ``seed`` and the epoch, never on the rank:
    every rank builds the same stream and takes a disjoint, strided slice of
    it, so no two ranks draw the same sample. Because it is deterministic,
    :meth:`set_epoch` can also skip samples an interrupted epoch already
    consumed, letting a mid-epoch resume continue the same sample sequence.

    Args:
        dataset_size: Number of samples in the dataset (> 0).
        samples_per_epoch: Samples this rank draws per epoch. None draws one
            pass over the dataset, split across ranks. Defaults to None.
        rank: This process's rank. Defaults to 0.
        world_size: Number of processes. Defaults to 1.
        seed: Base random seed. Defaults to 42.

    Raises:
        ValueError: If ``dataset_size`` is not positive.
    """

    def __init__(
        self,
        dataset_size: int,
        samples_per_epoch: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError(
                "Cannot sample from an empty dataset: the index has no "
                "volumes in this split."
            )
        self.dataset_size = dataset_size
        self.samples_per_epoch = (
            math.ceil(dataset_size / world_size)
            if samples_per_epoch is None else samples_per_epoch
        )
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int, start: int = 0) -> None:
        """Select the epoch to draw, skipping its first ``start`` samples.

        Args:
            epoch: Epoch index; seeds that epoch's permutation stream.
            start: Number of this rank's samples to skip (already consumed
                before a mid-epoch checkpoint). Defaults to 0.
        """
        self.epoch = epoch
        self.start = start

    def __len__(self) -> int:
        return self.samples_per_epoch - self.start

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        total = self.samples_per_epoch * self.world_size
        num_permutations = math.ceil(total / self.dataset_size)
        stream = torch.cat([
            torch.randperm(self.dataset_size, generator=generator)
            for _ in range(num_permutations)
        ])[:total]
        mine = stream[self.rank::self.world_size]
        return iter(mine[self.start:].tolist())
