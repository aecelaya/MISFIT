"""Tests for the resenc-pocket MAE model (ResEncMAE with use_pocket_model).

resenc-pocket needs MIST 3.0's ``ResEncUNet(use_pocket_model=...)``. Tests that
build the real network skip on older MIST; the version check and the argument
pass-through are tested against stand-ins, so they run on every MIST.
"""
import inspect
from unittest.mock import patch

import pytest
import torch
from mist.models.resenc.mist_resenc import ResEncUNet

import misfit.models  # noqa — trigger registrations
from misfit.models.model_registry import get_model_from_registry, list_registered_models
from misfit.models.pretrained import load_model_weights
from misfit.models.resenc import misfit_resenc_mae
from misfit.models.resenc.misfit_resenc_mae import ISOTROPIC_PLANNING_SPACING, ResEncMAE

IMG_SIZE = (32, 32, 32)
MASK_PATCH_SIZE = 16

MIST_HAS_POCKET = "use_pocket_model" in inspect.signature(ResEncUNet).parameters
requires_pocket = pytest.mark.skipif(
    not MIST_HAS_POCKET, reason="resenc-pocket needs mist-medical>=3.0.0rc0"
)


def _pocket(**kwargs):
    return get_model_from_registry(
        "resenc-pocket", in_channels=1, img_size=IMG_SIZE,
        mask_patch_size=MASK_PATCH_SIZE, mask_ratio=0.75, **kwargs,
    )


def _resenc_unet_with_pocket_arg(calls):
    """A ResEncUNet stand-in that accepts use_pocket_model, as MIST 3.0 does.

    It records the argument and builds MIST's full-width network, so the
    pass-through is testable on any MIST.
    """
    def build(*args, use_pocket_model=False, **kwargs):
        calls.append(use_pocket_model)
        return ResEncUNet(*args, **kwargs)
    return build


# ---------------------------------------------------------------------------
# Registry, pass-through, and the MIST version check (any MIST)
# ---------------------------------------------------------------------------

def test_resenc_pocket_is_registered():
    assert "resenc-pocket" in list_registered_models()


def test_resenc_pocket_passes_use_pocket_model_to_mist():
    calls = []
    with patch.object(
        misfit_resenc_mae, "ResEncUNet", _resenc_unet_with_pocket_arg(calls)
    ):
        model = _pocket()
    assert calls == [True]
    assert model.use_pocket_model is True


def test_resenc_does_not_pass_use_pocket_model():
    """Plain resenc never passes the argument, so it still builds on MIST 2.4."""
    with patch.object(
        misfit_resenc_mae, "ResEncUNet", wraps=misfit_resenc_mae.ResEncUNet
    ) as spy:
        model = get_model_from_registry(
            "resenc", in_channels=1, img_size=IMG_SIZE,
            mask_patch_size=MASK_PATCH_SIZE,
        )
    assert "use_pocket_model" not in spy.call_args.kwargs
    assert model.use_pocket_model is False


def test_resenc_pocket_on_old_mist_raises_a_clear_error():
    def old_resenc_unet(in_channels, out_channels, patch_size, target_spacing,
                        use_deep_supervision):  # MIST 2.4's signature
        raise AssertionError("must not be built")

    with patch.object(misfit_resenc_mae, "ResEncUNet", old_resenc_unet):
        with pytest.raises(ImportError, match=r"mist-medical>=3\.0\.0rc0"):
            _pocket()


def test_pocket_checkpoint_mismatch_names_resenc_pocket():
    """A pocket model rejects mismatched weights with a pocket-specific hint."""
    model = ResEncMAE(img_size=IMG_SIZE, mask_patch_size=MASK_PATCH_SIZE)
    model.use_pocket_model = True  # stand-in for a pocket model on any MIST
    sd = model.state_dict()
    del sd["head.weight"]
    with pytest.raises(ValueError, match="nnFoundationCNN .* don't load into it"):
        load_model_weights(model, sd)


# ---------------------------------------------------------------------------
# The real pocket network (MIST 3.0+)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pocket_model():
    if not MIST_HAS_POCKET:
        pytest.skip("resenc-pocket needs mist-medical>=3.0.0rc0")
    return _pocket()


@requires_pocket
def test_resenc_pocket_keeps_32_filters_per_stage(pocket_model):
    assert list(pocket_model.encoder.output_channels) == [32] * 4
    full = sum(p.numel() for p in ResEncMAE(
        img_size=IMG_SIZE, mask_patch_size=MASK_PATCH_SIZE).parameters())
    assert sum(p.numel() for p in pocket_model.parameters()) < full / 5


@requires_pocket
def test_resenc_pocket_forward_and_loss(pocket_model):
    x = torch.randn(2, 1, *IMG_SIZE)
    out = pocket_model(x)
    assert out["reconstruction"].shape == x.shape
    assert out["mask"].shape == (2, 1, *IMG_SIZE)
    loss = ((out["reconstruction"] - x).square() * out["mask"]).sum() / out["mask"].sum()
    assert torch.isfinite(loss)


@requires_pocket
def test_resenc_pocket_every_parameter_receives_gradient():
    model = _pocket()
    model(torch.randn(1, 1, *IMG_SIZE))["reconstruction"].square().mean().backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


@requires_pocket
def test_resenc_pocket_encoder_returns_bottleneck_last(pocket_model):
    feats = pocket_model.encoder(torch.randn(1, 1, *IMG_SIZE))
    assert feats[-1].shape == (1, 32, 4, 4, 4)


@requires_pocket
def test_resenc_pocket_encoder_state_dict_is_encoder_only(pocket_model):
    sd = pocket_model.get_encoder_state_dict()
    assert sd and all(k.startswith("encoder.") for k in sd)


@requires_pocket
def test_resenc_pocket_encoder_loads_into_mist_resenc_pocket(tmp_path):
    """The encoder is a drop-in for MIST's resenc-pocket, key by key."""
    import mist.models  # noqa: F401 — trigger MIST model registrations
    from mist.models import model_loader
    from mist.models.model_registry import get_model_from_registry as get_mist_model

    misfit_sd = _pocket().get_encoder_state_dict()
    mist_model = get_mist_model(
        "resenc-pocket", in_channels=1, out_channels=3,
        patch_size=list(IMG_SIZE), target_spacing=list(ISOTROPIC_PLANNING_SPACING),
    )
    mist_sd = mist_model.get_encoder_state_dict()
    assert list(misfit_sd) == list(mist_sd)
    for key, tensor in misfit_sd.items():
        assert tensor.shape == mist_sd[key].shape, key

    weights_path = tmp_path / "misfit_encoder.pt"
    torch.save(misfit_sd, weights_path)
    mist_model, summary = model_loader.load_pretrained_encoder(
        mist_model, str(weights_path)
    )
    assert len(summary["loaded"]) == len(mist_sd)
    assert not summary["skipped"]
    for key, tensor in misfit_sd.items():
        assert torch.equal(mist_model.state_dict()[key], tensor), key


@requires_pocket
def test_mist_accepts_resenc_pocket_config_and_rejects_resenc():
    """MISFIT saves model.architecture = resenc-pocket; MIST must see it as such."""
    from mist.models.model_loader import validate_encoder_compatibility

    spatial = {"patch_size": list(IMG_SIZE), "target_spacing": [1.0, 1.0, 1.0]}
    source = {"model": {"architecture": "resenc-pocket"}, "spatial_config": spatial}
    validate_encoder_compatibility(
        source, {"model": {"architecture": "resenc-pocket"}, "spatial_config": spatial}
    )
    with pytest.raises(ValueError, match="Architecture mismatch"):
        validate_encoder_compatibility(
            source, {"model": {"architecture": "resenc"}, "spatial_config": spatial}
        )


@requires_pocket
def test_resenc_pocket_rejects_full_width_nnssl_weights():
    """nnFoundationCNN-style (full-width) ResEnc weights don't fit the pocket model."""
    full = ResEncUNet(1, 1, list(IMG_SIZE), [1.0, 1.0, 1.0], use_deep_supervision=False)
    with pytest.raises(ValueError, match="resenc-pocket keeps 32 filters"):
        load_model_weights(_pocket(), {"network_weights": full.state_dict()})
