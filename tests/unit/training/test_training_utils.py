"""Tests for misfit.training.training_utils."""
import random

import numpy as np
import pytest
import torch

from misfit.training.training_utils import (
    RunningMean,
    build_accumulation_plan,
    set_seed,
)

# ---------------------------------------------------------------------------
# RunningMean
# ---------------------------------------------------------------------------


def test_running_mean_initial_value():
    m = RunningMean()
    assert m.value == 0.0


def test_running_mean_single_update():
    m = RunningMean()
    m.update(4.0)
    assert m.value == 4.0


def test_running_mean_multiple_updates():
    m = RunningMean()
    m.update(2.0)
    m.update(4.0)
    assert m.value == pytest.approx(3.0)


def test_running_mean_weighted_update():
    m = RunningMean()
    m.update(10.0, n=2)  # sum=20, count=2
    m.update(0.0, n=2)   # sum=20, count=4
    assert m.value == pytest.approx(5.0)


def test_running_mean_reset():
    m = RunningMean()
    m.update(5.0)
    m.reset()
    assert m.value == 0.0


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

def test_set_seed_reproducible():
    set_seed(42, rank=0)
    a = random.random()
    set_seed(42, rank=0)
    b = random.random()
    assert a == b


def test_set_seed_rank_offset():
    set_seed(42, rank=0)
    a = np.random.randn()
    set_seed(42, rank=1)
    b = np.random.randn()
    # Different seeds → different values
    assert a != b


def test_set_seed_sets_torch_seed():
    set_seed(99, rank=0)
    t1 = torch.randn(1).item()
    set_seed(99, rank=0)
    t2 = torch.randn(1).item()
    assert t1 == t2


# ---------------------------------------------------------------------------
# build_accumulation_plan
# ---------------------------------------------------------------------------

def test_accumulation_plan_evenly_divisible():
    # 4 batches, accum 2 → two full windows; only window ends step.
    assert build_accumulation_plan(4, 2) == [
        (2, False), (2, True), (2, False), (2, True)
    ]


def test_accumulation_plan_trailing_partial_window():
    # 5 batches, accum 2 → two full windows + a trailing window of size 1.
    plan = build_accumulation_plan(5, 2)
    assert plan == [(2, False), (2, True), (2, False), (2, True), (1, True)]
    # The trailing batch is scaled by its actual size (1), not accum_steps.
    assert plan[-1] == (1, True)


def test_accumulation_plan_final_batch_always_closes_window():
    # This is the core invariant that prevents the no_sync() divergence bug:
    # whatever the batch count, the last micro-step must close its window.
    for num_batches in range(1, 20):
        for accum in range(1, 6):
            plan = build_accumulation_plan(num_batches, accum)
            assert len(plan) == num_batches
            assert plan[-1][1] is True
            # One optimizer step per window end; equals ceil(num/accum).
            n_steps = sum(end for _, end in plan)
            assert n_steps == -(-num_batches // accum)


def test_accumulation_plan_accum_one_steps_every_batch():
    assert build_accumulation_plan(3, 1) == [(1, True), (1, True), (1, True)]


def test_accumulation_plan_single_batch_smaller_than_accum():
    # 1 batch, accum 3 → a single window of size 1 that steps (with sync).
    assert build_accumulation_plan(1, 3) == [(1, True)]


def test_accumulation_plan_empty_loader():
    assert build_accumulation_plan(0, 2) == []


def test_accumulation_plan_rejects_bad_accum():
    with pytest.raises(ValueError, match="accum_steps must be >= 1"):
        build_accumulation_plan(4, 0)
