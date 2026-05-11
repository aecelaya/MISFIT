"""Tests for misfit.training.lr_schedulers.lr_scheduler_registry."""
import pytest
import torch
import torch.nn as nn

from misfit.training.lr_schedulers.lr_scheduler_registry import (
    get_lr_scheduler,
    list_lr_schedulers,
)


def _make_optimizer():
    model = nn.Linear(4, 4)
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def test_list_lr_schedulers_sorted():
    schedulers = list_lr_schedulers()
    assert schedulers == sorted(schedulers)


def test_list_lr_schedulers_contains_builtins():
    schedulers = list_lr_schedulers()
    assert "cosine" in schedulers
    assert "polynomial" in schedulers
    assert "constant" in schedulers


@pytest.mark.parametrize("name", ["cosine", "polynomial", "constant"])
def test_get_lr_scheduler_success(name):
    opt = _make_optimizer()
    sched = get_lr_scheduler(name, opt, epochs=10, warmup_epochs=0)
    assert sched is not None


def test_get_lr_scheduler_unknown_raises():
    opt = _make_optimizer()
    with pytest.raises(ValueError, match="not registered"):
        get_lr_scheduler("totally_unknown_sched", opt, epochs=10)


def test_get_lr_scheduler_with_warmup():
    opt = _make_optimizer()
    sched = get_lr_scheduler("cosine", opt, epochs=20, warmup_epochs=5)
    # With warmup, should be a SequentialLR
    from torch.optim.lr_scheduler import SequentialLR
    assert isinstance(sched, SequentialLR)


def test_get_lr_scheduler_no_warmup():
    opt = _make_optimizer()
    sched = get_lr_scheduler("cosine", opt, epochs=10, warmup_epochs=0)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    assert isinstance(sched, CosineAnnealingLR)


def test_get_lr_scheduler_case_insensitive():
    opt = _make_optimizer()
    sched = get_lr_scheduler("COSINE", opt, epochs=10)
    assert sched is not None
