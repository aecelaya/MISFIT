"""Tests for misfit.training.training_utils."""
import random

import numpy as np
import pytest
import torch

from misfit.training.training_utils import RunningMean, set_seed


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
    assert m._count == 0


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
