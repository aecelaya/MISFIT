"""MISFIT crop aggregators."""
from misfit.embedding.aggregators.aggregator_registry import (
    get_aggregator,
    list_aggregators,
    register_aggregator,
)
from misfit.embedding.aggregators.attention_pool import AttentionPoolAggregator  # noqa: F401
from misfit.embedding.aggregators.mean_pool import MeanPoolAggregator  # noqa: F401

__all__ = [
    "AttentionPoolAggregator",
    "MeanPoolAggregator",
    "get_aggregator",
    "list_aggregators",
    "register_aggregator",
]
