"""Tests for misfit.training.optimizers.optimizer_registry."""
import pytest
import torch
import torch.nn as nn

from misfit.training.optimizers.optimizer_registry import (
    OPTIMIZER_REGISTRY,
    get_optimizer,
    list_optimizers,
    register_optimizer,
)


def _params():
    return nn.Linear(4, 4).parameters()


def test_list_optimizers_sorted():
    opts = list_optimizers()
    assert opts == sorted(opts)


def test_list_optimizers_contains_builtins():
    opts = list_optimizers()
    assert "adam" in opts
    assert "adamw" in opts
    assert "sgd" in opts


@pytest.mark.parametrize("name", ["adam", "adamw", "sgd"])
def test_get_optimizer_success(name):
    opt = get_optimizer(name, _params(), 1e-3, 1e-4, 1e-8)
    assert isinstance(opt, torch.optim.Optimizer)


def test_get_optimizer_unknown_raises():
    with pytest.raises(ValueError, match="not registered"):
        get_optimizer("totally_unknown_optimizer", _params(), 1e-3, 0.0, 1e-8)


def test_get_optimizer_case_insensitive():
    opt = get_optimizer("ADAM", _params(), 1e-3, 0.0, 1e-8)
    assert isinstance(opt, torch.optim.Adam)


def test_register_optimizer_custom():
    name = "_test_custom_opt_xyz"
    OPTIMIZER_REGISTRY.pop(name, None)

    @register_optimizer(name)
    def _build(params, lr, wd, eps):
        return torch.optim.SGD(params, lr=lr)

    opt = get_optimizer(name, _params(), 1e-2, 0.0, 1e-8)
    assert isinstance(opt, torch.optim.SGD)
    OPTIMIZER_REGISTRY.pop(name, None)
