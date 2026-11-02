"""Tests for misfit.data_loading.samplers.EpochSampler."""
from collections import Counter

import pytest

from misfit.data_loading.samplers import EpochSampler


def test_draws_samples_per_epoch_regardless_of_dataset_size():
    assert len(list(EpochSampler(3, samples_per_epoch=10))) == 10
    assert len(list(EpochSampler(1000, samples_per_epoch=10))) == 10


def test_default_is_one_pass_split_across_ranks():
    assert len(EpochSampler(10)) == 10
    assert len(EpochSampler(10, world_size=4)) == 3


def test_no_repeats_within_an_epoch_shorter_than_the_dataset():
    indices = list(EpochSampler(100, samples_per_epoch=50))
    assert len(set(indices)) == 50


def test_long_epoch_covers_the_dataset_evenly():
    """An epoch of k full passes draws every sample exactly k times."""
    counts = Counter(EpochSampler(7, samples_per_epoch=21))
    assert set(counts.values()) == {3}
    assert set(counts) == set(range(7))


def test_ranks_draw_disjoint_samples_from_one_stream():
    ranks = [
        list(EpochSampler(100, samples_per_epoch=20, rank=r, world_size=4, seed=1))
        for r in range(4)
    ]
    flat = [i for r in ranks for i in r]
    assert len(flat) == 80 and len(set(flat)) == 80


def test_epochs_differ_and_are_reproducible():
    sampler = EpochSampler(50, samples_per_epoch=10, seed=0)
    sampler.set_epoch(0)
    epoch0 = list(sampler)
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    sampler.set_epoch(0)
    assert list(sampler) == epoch0
    assert epoch1 != epoch0


def test_set_epoch_start_skips_consumed_samples():
    """Resuming mid-epoch continues the same sample order."""
    sampler = EpochSampler(50, samples_per_epoch=10, seed=0)
    sampler.set_epoch(2)
    full = list(sampler)
    sampler.set_epoch(2, start=4)
    assert len(sampler) == 6
    assert list(sampler) == full[4:]


def test_rejects_empty_dataset():
    with pytest.raises(ValueError, match="empty"):
        EpochSampler(0)
