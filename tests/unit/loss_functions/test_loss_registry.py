"""Tests for misfit.loss_functions.loss_registry."""
import pytest

import misfit.loss_functions  # noqa — trigger registrations
from misfit.loss_functions.loss_registry import (
    LOSS_REGISTRY,
    get_loss,
    list_registered_losses,
    register_loss,
)
from misfit.loss_functions.base import ReconstructionLoss


def test_list_registered_losses_sorted():
    losses = list_registered_losses()
    assert losses == sorted(losses)


def test_list_contains_builtin_losses():
    losses = list_registered_losses()
    assert "masked_mse" in losses
    assert "masked_l1" in losses
    assert "normalized_masked_mse" in losses


def test_get_loss_success():
    cls = get_loss("masked_mse")
    assert issubclass(cls, ReconstructionLoss)


def test_get_loss_case_insensitive():
    cls = get_loss("MASKED_MSE")
    assert issubclass(cls, ReconstructionLoss)


def test_get_loss_unknown_raises():
    with pytest.raises(ValueError, match="not registered"):
        get_loss("totally_unknown_loss_xyz")


def test_register_loss_duplicate_raises():
    name = "_test_dup_loss_xyz"
    LOSS_REGISTRY.pop(name.lower(), None)

    @register_loss(name)
    class MyLoss(ReconstructionLoss):
        def forward(self, r, t, m):
            return r.sum()

    with pytest.raises(ValueError, match="already registered"):
        @register_loss(name)
        class MyLoss2(ReconstructionLoss):
            def forward(self, r, t, m):
                return r.sum()

    LOSS_REGISTRY.pop(name.lower(), None)
