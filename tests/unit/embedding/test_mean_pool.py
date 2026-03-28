"""Tests for misfit.embedding.aggregators.mean_pool.MeanPoolAggregator."""
import torch
import pytest

from misfit.embedding.aggregators.mean_pool import MeanPoolAggregator


@pytest.fixture
def agg():
    return MeanPoolAggregator(embed_dim=16)


def test_mean_pool_forward_shape(agg):
    features = torch.randn(5, 16)
    out = agg(features)
    assert out.shape == (16,)


def test_mean_pool_ignores_positions(agg):
    features = torch.randn(5, 16)
    pos = torch.randn(5, 3)
    out_with_pos = agg(features, positions=pos)
    out_no_pos = agg(features, positions=None)
    assert torch.allclose(out_with_pos, out_no_pos)


def test_mean_pool_respects_padding_mask(agg):
    features = torch.zeros(4, 16)
    features[0] = torch.ones(16)  # only valid crop
    mask = torch.tensor([False, True, True, True])  # only first valid
    out = agg(features, padding_mask=mask)
    assert torch.allclose(out, torch.ones(16))


def test_mean_pool_no_padding_mask(agg):
    features = torch.ones(3, 16)
    out = agg(features)
    assert torch.allclose(out, torch.ones(16))
