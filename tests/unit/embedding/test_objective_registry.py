"""Tests for misfit.embedding.objectives.objective_registry."""
import pytest

import misfit.embedding  # noqa — trigger registrations
from misfit.embedding.objectives.objective_registry import (
    OBJECTIVE_REGISTRY,
    get_objective,
    list_objectives,
    register_objective,
)
from misfit.embedding.objectives.base import TrainingObjective


def test_list_objectives_sorted():
    objs = list_objectives()
    assert objs == sorted(objs)


def test_list_objectives_contains_builtins():
    objs = list_objectives()
    assert "classification" in objs
    assert "contrastive" in objs


def test_get_objective_success():
    cls = get_objective("classification")
    assert issubclass(cls, TrainingObjective)


def test_get_objective_unknown_raises():
    with pytest.raises(KeyError, match="not registered"):
        get_objective("totally_unknown_obj_xyz")


def test_register_objective_duplicate_raises():
    name = "_test_dup_obj_xyz"
    OBJECTIVE_REGISTRY.pop(name, None)

    @register_objective(name)
    class MyObj(TrainingObjective):
        name = "_test_dup_obj_xyz"
        def validate_labels(self, df, col): return df
        def build_dataset(self, fd, df, col): return None
        def build_batch_sampler(self, ds, bs): return None
        def build_head(self, dim): return None
        def compute_loss(self, out, labels): return out.sum()

    with pytest.raises(KeyError, match="already registered"):
        @register_objective(name)
        class MyObj2(TrainingObjective):
            name = "_test_dup_obj_xyz2"
            def validate_labels(self, df, col): return df
            def build_dataset(self, fd, df, col): return None
            def build_batch_sampler(self, ds, bs): return None
            def build_head(self, dim): return None
            def compute_loss(self, out, labels): return out.sum()

    OBJECTIVE_REGISTRY.pop(name, None)


def test_register_objective_wrong_base_raises():
    with pytest.raises(TypeError, match="TrainingObjective"):
        @register_objective("_test_bad_obj")
        class NotAnObjective:
            pass
