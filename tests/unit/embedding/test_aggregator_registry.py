"""Tests for misfit.embedding.aggregators.aggregator_registry."""
import pytest

import misfit.embedding  # noqa — trigger registrations
from misfit.embedding.aggregators.aggregator_registry import (
    AGGREGATOR_REGISTRY,
    get_aggregator,
    list_aggregators,
    register_aggregator,
)
from misfit.embedding.aggregators.base import AbstractAggregator


def test_list_aggregators_sorted():
    aggs = list_aggregators()
    assert aggs == sorted(aggs)


def test_list_aggregators_contains_builtins():
    aggs = list_aggregators()
    assert "mean_pool" in aggs
    assert "attention_pool" in aggs


def test_get_aggregator_success():
    cls = get_aggregator("mean_pool")
    assert issubclass(cls, AbstractAggregator)


def test_get_aggregator_unknown_raises():
    with pytest.raises(KeyError, match="not registered"):
        get_aggregator("totally_unknown_agg_xyz")


def test_register_aggregator_duplicate_raises():
    name = "_test_dup_agg_xyz"
    AGGREGATOR_REGISTRY.pop(name, None)

    import torch
    from typing import Optional

    @register_aggregator(name)
    class MyAgg(AbstractAggregator):
        def forward(self, crop_features, positions=None, padding_mask=None):
            return crop_features.mean(dim=0)

    with pytest.raises(KeyError, match="already registered"):
        @register_aggregator(name)
        class MyAgg2(AbstractAggregator):
            def forward(self, crop_features, positions=None, padding_mask=None):
                return crop_features.mean(dim=0)

    AGGREGATOR_REGISTRY.pop(name, None)


def test_register_aggregator_wrong_base_raises():
    with pytest.raises(TypeError, match="AbstractAggregator"):
        @register_aggregator("_test_bad_agg")
        class NotAnAgg:
            pass
