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


def test_extract_crop_features_crops_already_on_device():
    """Crops passed to encoder_fn must already be on self.device.

    The fix removed a redundant .to(self.device) inside the loop; this test
    confirms that crops are on the correct device before encoder_fn is called.
    """
    received_devices = []

    def tracking_encoder(x):
        received_devices.append(str(x.device))
        B = x.shape[0]
        return torch.ones(B, 16, 2, 2, 2, device=x.device)

    aggregator = MeanPoolAggregator(embed_dim=16)
    emb = Embedder(
        encoder_fn=tracking_encoder,
        aggregator=aggregator,
        patch_size=16,
        device=torch.device("cpu"),
    )
    volume = torch.randn(1, 32, 32, 32)
    emb._extract_crop_features(volume)

    # Every crop should have been on "cpu" when encoder_fn saw it.
    assert all(d == "cpu" for d in received_devices), (
        f"Some crops were not on cpu before encoder_fn: {received_devices}"
    )


def test_extract_crop_features_positions_on_device():
    """positions returned by _extract_crop_features must be on self.device."""
    aggregator = MeanPoolAggregator(embed_dim=16)
    emb = Embedder(
        encoder_fn=_dummy_encoder,
        aggregator=aggregator,
        patch_size=16,
        device=torch.device("cpu"),
    )
    volume = torch.randn(1, 32, 32, 32)
    _, positions = emb._extract_crop_features(volume)
    assert positions.device == torch.device("cpu")


def test_embedder_int_patch_size_normalised_to_triple():
    """An int patch_size is stored internally as a (D, H, W) triple."""
    aggregator = MeanPoolAggregator(embed_dim=16)
    emb = Embedder(
        encoder_fn=_dummy_encoder,
        aggregator=aggregator,
        patch_size=16,
        device=torch.device("cpu"),
    )
    assert emb.patch_size == (16, 16, 16)


def test_embedder_non_cubic_patch_size_tiles_per_axis():
    """Anisotropic patch_size tiles each axis independently."""
    aggregator = MeanPoolAggregator(embed_dim=16)
    emb = Embedder(
        encoder_fn=_dummy_encoder,
        aggregator=aggregator,
        patch_size=(16, 32, 16),
        device=torch.device("cpu"),
    )
    assert emb.patch_size == (16, 32, 16)

    # Volume 32×64×16 with patch 16×32×16 → 2 × 2 × 1 = 4 crops.
    volume = torch.randn(1, 32, 64, 16)
    features, positions = emb.extract_crop_features(volume)
    assert features.shape == (4, 16)
    assert positions.shape == (4, 3)


def test_embedder_non_cubic_pads_per_axis():
    """Anisotropic patch_size pads each axis to its own multiple."""
    aggregator = MeanPoolAggregator(embed_dim=16)
    emb = Embedder(
        encoder_fn=_dummy_encoder,
        aggregator=aggregator,
        patch_size=(16, 32, 16),
        device=torch.device("cpu"),
    )
    # 17×33×15 → padded to 32×64×16 → 2 × 2 × 1 = 4 crops.
    volume = torch.randn(1, 17, 33, 15)
    features, positions = emb.extract_crop_features(volume)
    assert features.shape[0] == 4
    assert positions.shape == (4, 3)
