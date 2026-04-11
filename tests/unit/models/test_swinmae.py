"""Tests for misfit.models.swinunetr.misfit_swinunetr_mae (SwinMAE)
and swinunetr_registry builder functions."""
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch

import misfit.models  # noqa — trigger registrations
from misfit.models.swinunetr.misfit_swinunetr_mae import MAEDecoder, SwinMAE


# Use tiny feature_size=12 for speed (not in registry, direct construction)
IMG_SIZE = (32, 32, 32)
MASK_PATCH_SIZE = 16


@pytest.fixture(scope="module")
def tiny_model():
    """Smallest possible SwinMAE for fast CPU tests."""
    return SwinMAE(
        in_channels=1,
        feature_size=12,
        img_size=IMG_SIZE,
        mask_patch_size=MASK_PATCH_SIZE,
        mask_ratio=0.75,
    )


# ---------------------------------------------------------------------------
# MAEDecoder
# ---------------------------------------------------------------------------

def test_mae_decoder_forward():
    decoder = MAEDecoder(in_channels=16, out_channels=1, num_upsample=2, base_channels=8)
    x = torch.randn(1, 16, 4, 4, 4)
    out = decoder(x)
    assert out.shape == (1, 1, 16, 16, 16)


def test_mae_decoder_base_channels_floor():
    """base_channels prevents channels collapsing to 0."""
    decoder = MAEDecoder(in_channels=4, out_channels=1, num_upsample=3, base_channels=4)
    x = torch.randn(1, 4, 2, 2, 2)
    out = decoder(x)
    assert out.shape[0] == 1


# ---------------------------------------------------------------------------
# SwinMAE construction validation
# ---------------------------------------------------------------------------

def test_swinmae_invalid_img_size_not_div32():
    with pytest.raises(ValueError, match="divisible by 32"):
        SwinMAE(feature_size=12, img_size=(33, 32, 32))


def test_swinmae_invalid_img_size_not_div_mask_patch():
    with pytest.raises(ValueError, match="mask_patch_size"):
        SwinMAE(feature_size=12, img_size=(32, 32, 32), mask_patch_size=24)


# ---------------------------------------------------------------------------
# SwinMAE forward
# ---------------------------------------------------------------------------

def test_swinmae_forward_output_keys(tiny_model):
    x = torch.randn(1, 1, *IMG_SIZE)
    out = tiny_model(x)
    assert "reconstruction" in out
    assert "mask" in out


def test_swinmae_forward_reconstruction_shape(tiny_model):
    x = torch.randn(1, 1, *IMG_SIZE)
    out = tiny_model(x)
    assert out["reconstruction"].shape == x.shape


def test_swinmae_forward_mask_shape(tiny_model):
    x = torch.randn(1, 1, *IMG_SIZE)
    out = tiny_model(x)
    assert out["mask"].shape == (1, 1, *IMG_SIZE)


def test_swinmae_forward_mask_binary(tiny_model):
    x = torch.randn(1, 1, *IMG_SIZE)
    out = tiny_model(x)
    mask = out["mask"]
    assert ((mask == 0.0) | (mask == 1.0)).all()


def test_swinmae_forward_custom_mask_ratio():
    """A model constructed with mask_ratio=0.5 should mask ~50% of voxels."""
    model = SwinMAE(
        in_channels=1,
        feature_size=12,
        img_size=IMG_SIZE,
        mask_patch_size=MASK_PATCH_SIZE,
        mask_ratio=0.5,
    )
    x = torch.randn(1, 1, *IMG_SIZE)
    out = model(x)
    mask = out["mask"]
    ratio = mask.mean().item()
    assert 0.3 <= ratio <= 0.7


# ---------------------------------------------------------------------------
# generate_mask
# ---------------------------------------------------------------------------

def test_generate_mask_shape(tiny_model):
    x = torch.randn(2, 1, *IMG_SIZE)
    mask = tiny_model.generate_mask(x)
    assert mask.shape == (2, 1, *IMG_SIZE)


def test_generate_mask_ratio(tiny_model):
    x = torch.randn(1, 1, *IMG_SIZE)
    mask = tiny_model.generate_mask(x)
    ratio = mask.mean().item()
    # With mask_ratio=0.75, expect ~75% masked
    assert 0.6 <= ratio <= 0.9


# ---------------------------------------------------------------------------
# get_encoder_state_dict
# ---------------------------------------------------------------------------

def test_get_encoder_state_dict_returns_ordered_dict(tiny_model):
    sd = tiny_model.get_encoder_state_dict()
    assert isinstance(sd, OrderedDict)


def test_get_encoder_state_dict_keys_have_mist_prefix(tiny_model):
    sd = tiny_model.get_encoder_state_dict()
    for key in sd:
        assert key.startswith("model.swinViT."), (
            f"Expected 'model.swinViT.' prefix for MIST compatibility, got: {key!r}"
        )


def test_get_encoder_state_dict_not_empty(tiny_model):
    sd = tiny_model.get_encoder_state_dict()
    assert len(sd) > 0


# ---------------------------------------------------------------------------
# MONAI version compatibility — SwinUNETR img_size fallback
# ---------------------------------------------------------------------------

def test_swinmae_init_without_img_size_succeeds():
    """MONAI >= 1.5: SwinUNETR constructed without img_size (normal path)."""
    # The real SwinUNETR in the test environment should work without img_size.
    model = SwinMAE(feature_size=12, img_size=IMG_SIZE, mask_patch_size=MASK_PATCH_SIZE)
    assert model is not None


def test_swinmae_init_falls_back_to_img_size_on_type_error():
    """MONAI < 1.5: TypeError on first call triggers fallback with img_size."""
    calls = []

    # Build a realistic mock encoder: returns 5 hidden states with the right
    # channel shapes so the bottleneck probe and forward pass don't crash.
    mock_encoder = MagicMock()
    # _hidden[4] must have .shape[1] accessible (bottleneck channel count)
    hidden = [MagicMock() for _ in range(5)]
    hidden[4].shape = (1, 192, 1, 1, 1)
    mock_encoder.return_value = hidden

    mock_swinunetr = MagicMock()
    mock_swinunetr.swinViT = mock_encoder

    def patched_swinunetr(*args, **kwargs):
        calls.append(kwargs.copy())
        if len(calls) == 1:
            # Simulate MONAI < 1.5: first call without img_size fails.
            raise TypeError("missing a required argument: 'img_size'")
        # Second call (fallback with img_size) succeeds.
        return mock_swinunetr

    with patch(
        "misfit.models.swinunetr.misfit_swinunetr_mae.SwinUNETR",
        side_effect=patched_swinunetr,
    ):
        model = SwinMAE(feature_size=12, img_size=IMG_SIZE, mask_patch_size=MASK_PATCH_SIZE)

    assert len(calls) == 2
    assert "img_size" not in calls[0]
    assert calls[1]["img_size"] == IMG_SIZE


# ---------------------------------------------------------------------------
# swinunetr_registry builder functions (lines 25 and 31)
# Patch SwinMAE so no weights are allocated — we only care that the right
# feature_size is forwarded by each builder.
# ---------------------------------------------------------------------------

def test_swinmae_small_registry_real_instantiation():
    """swinmae-small is actually instantiated via the registry (feature_size=24)."""
    from misfit.models.model_registry import get_model_from_registry

    model = get_model_from_registry(
        "swinmae-small",
        in_channels=1,
        img_size=(32, 32, 32),
        mask_patch_size=16,
        mask_ratio=0.75,
    )
    assert isinstance(model, SwinMAE)
    # feature_size=24 means the encoder bottleneck has 16 * 24 = 384 channels
    x = torch.randn(1, 1, 32, 32, 32)
    out = model(x)
    assert "reconstruction" in out
    assert out["reconstruction"].shape == x.shape


@pytest.mark.parametrize("model_name,expected_feature_size", [
    ("swinmae-base",  48),
    ("swinmae-large", 96),
])
def test_registry_builder_forwards_feature_size(model_name, expected_feature_size):
    """swinmae-base and swinmae-large builders pass the correct feature_size."""
    from misfit.models.model_registry import get_model_from_registry

    mock_instance = MagicMock()
    with patch(
        "misfit.models.swinunetr.swinunetr_registry.SwinMAE",
        return_value=mock_instance,
    ) as MockSwinMAE:
        result = get_model_from_registry(
            model_name,
            in_channels=1,
            img_size=(32, 32, 32),
            mask_patch_size=16,
            mask_ratio=0.75,
        )

    MockSwinMAE.assert_called_once_with(
        feature_size=expected_feature_size,
        in_channels=1,
        img_size=(32, 32, 32),
        mask_patch_size=16,
        mask_ratio=0.75,
    )
    assert result is mock_instance


# ---------------------------------------------------------------------------
# Transfer learning integration — MIST swinunetr-small accepts MISFIT encoder
# ---------------------------------------------------------------------------

def test_mist_swinunetr_small_accepts_misfit_encoder(tmp_path):
    """MISFIT SwinMAE (swinmae-small) encoder weights load into MIST swinunetr-small.

    Requires MIST to be installed in the test environment; skipped otherwise.
    Both models use feature_size=24 so encoder shapes are identical.
    """
    model_loader = pytest.importorskip("mist.models.model_loader")
    pytest.importorskip("mist.models")  # trigger MIST model registrations
    from mist.models.model_registry import get_model_from_registry

    # Build MISFIT swinmae-small and export encoder in MIST-compatible format.
    misfit_model = SwinMAE(
        in_channels=1,
        feature_size=24,
        img_size=(32, 32, 32),
        mask_patch_size=16,
        mask_ratio=0.75,
    )
    encoder_sd = misfit_model.get_encoder_state_dict()
    weights_path = tmp_path / "misfit_encoder.pt"
    torch.save(encoder_sd, weights_path)

    # Build MIST swinunetr-small and load the MISFIT encoder weights.
    mist_model = get_model_from_registry(
        "swinunetr-small",
        in_channels=1,
        out_channels=2,
    )
    mist_model, summary = model_loader.load_pretrained_encoder(
        mist_model, str(weights_path)
    )

    assert len(summary["loaded"]) > 0, "No encoder weights were transferred."
    assert len(summary["skipped"]) == 0, (
        f"Encoder weights were skipped (key mismatch): {summary['skipped'][:5]}"
    )
