"""Tests for misfit.embedding.aggregators.attention_pool.AttentionPoolAggregator."""
import pytest
import torch

from misfit.embedding.aggregators.attention_pool import AttentionPoolAggregator


@pytest.fixture
def agg():
    return AttentionPoolAggregator(embed_dim=16, use_position_encoding=True)


@pytest.fixture
def agg_no_pos():
    return AttentionPoolAggregator(embed_dim=16, use_position_encoding=False)


def test_attention_pool_output_shape(agg):
    features = torch.randn(5, 16)
    positions = torch.rand(5, 3)
    out = agg(features, positions=positions)
    assert out.shape == (16,)


def test_attention_pool_no_position_encoding(agg_no_pos):
    features = torch.randn(5, 16)
    out = agg_no_pos(features, positions=None)
    assert out.shape == (16,)


def test_attention_pool_with_padding_mask(agg):
    features = torch.randn(5, 16)
    positions = torch.rand(5, 3)
    mask = torch.tensor([False, False, False, True, True])
    out = agg(features, positions=positions, padding_mask=mask)
    assert out.shape == (16,)


def test_attention_pool_no_positions_with_encoding(agg):
    """Positions=None should still work (pos_proj not applied)."""
    features = torch.randn(5, 16)
    out = agg(features, positions=None)
    assert out.shape == (16,)


def test_attention_pool_output_not_nan(agg):
    features = torch.randn(5, 16)
    positions = torch.rand(5, 3)
    out = agg(features, positions=positions)
    assert not torch.isnan(out).any()
