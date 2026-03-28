"""Tests for misfit.embedding.embedder.Embedder."""
import numpy as np
import pytest
import torch

from misfit.embedding.aggregators.mean_pool import MeanPoolAggregator
from misfit.embedding.embedder import Embedder


def _dummy_encoder(x):
    """Returns a (B, 16, 2, 2, 2) feature map for any input size."""
    B = x.shape[0]
    return torch.ones(B, 16, 2, 2, 2)


@pytest.fixture
def embedder():
    aggregator = MeanPoolAggregator(embed_dim=16)
    return Embedder(
        encoder_fn=_dummy_encoder,
        aggregator=aggregator,
        patch_size=16,
        device=torch.device("cpu"),
    )


def test_embedder_embed_shape(embedder):
    volume = torch.randn(1, 32, 32, 32)
    embedding = embedder.embed(volume)
    assert embedding.shape == (16,)


def test_embedder_extract_crop_features_shapes(embedder):
    volume = torch.randn(1, 32, 32, 32)
    features, positions = embedder.extract_crop_features(volume)
    assert isinstance(features, np.ndarray)
    assert isinstance(positions, np.ndarray)
    assert features.ndim == 2
    assert features.shape[1] == 16
    assert positions.ndim == 2
    assert positions.shape[1] == 3


def test_embedder_pads_volume_if_needed(embedder):
    """Volume of (1, 17, 17, 17) should be padded to (1, 32, 32, 32)."""
    volume = torch.randn(1, 17, 17, 17)
    embedding = embedder.embed(volume)
    assert embedding.shape == (16,)


def test_embedder_correct_n_crops(embedder):
    """32x32x32 with patch_size=16 → 2^3=8 crops."""
    volume = torch.randn(1, 32, 32, 32)
    features, positions = embedder.extract_crop_features(volume)
    assert features.shape[0] == 8  # 2*2*2 crops


def test_embedder_positions_in_unit_cube(embedder):
    volume = torch.randn(1, 32, 32, 32)
    _, positions = embedder.extract_crop_features(volume)
    assert (positions >= 0).all()
    assert (positions <= 1).all()
