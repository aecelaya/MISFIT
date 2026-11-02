"""Tests for misfit.models.resenc (ResEncMAE) and its registry entry."""
from unittest.mock import patch

import pytest
import torch

import misfit.models  # noqa — trigger registrations
from misfit.models.masking import generate_patch_mask
from misfit.models.model_registry import get_model_from_registry
from misfit.models.resenc import misfit_resenc_mae
from misfit.models.resenc.misfit_resenc_mae import PRETRAINING_SPACING, ResEncMAE

IMG_SIZE = (32, 32, 32)
MASK_PATCH_SIZE = 16


@pytest.fixture(scope="module")
def tiny_model():
    """Smallest ResEncMAE (4 stages at 32³) for fast CPU tests."""
    return ResEncMAE(
        in_channels=1,
        img_size=IMG_SIZE,
        mask_patch_size=MASK_PATCH_SIZE,
        mask_ratio=0.75,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_resencmae_invalid_img_size_not_div32():
    with pytest.raises(ValueError, match="divisible by 32"):
        ResEncMAE(img_size=(48, 32, 32), mask_patch_size=16)


def test_resencmae_invalid_img_size_not_div_mask_patch():
    with pytest.raises(ValueError, match="mask_patch_size"):
        ResEncMAE(img_size=IMG_SIZE, mask_patch_size=24)


def test_resencmae_built_from_mist_resenc_with_isotropic_spacing():
    """The network is MIST's ResEncUNet, planned at isotropic spacing."""
    with patch.object(
        misfit_resenc_mae, "ResEncUNet", wraps=misfit_resenc_mae.ResEncUNet
    ) as spy:
        ResEncMAE(in_channels=1, img_size=IMG_SIZE, mask_patch_size=MASK_PATCH_SIZE)

    spy.assert_called_once_with(
        in_channels=1,
        out_channels=1,
        patch_size=IMG_SIZE,
        target_spacing=PRETRAINING_SPACING,
        use_deep_supervision=False,
    )
    assert PRETRAINING_SPACING == (1.0, 1.0, 1.0)


def test_resencmae_topology_follows_img_size():
    """Isotropic planning: one stride-2 stage per halving down to < 8 voxels."""
    model = ResEncMAE(img_size=(64, 64, 32), mask_patch_size=16)
    assert model.encoder.strides == [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 1]]


def test_resencmae_drops_deep_supervision_layers(tiny_model):
    """Only the full-resolution head remains; no lower-res seg layers."""
    assert not hasattr(tiny_model.decoder, "seg_layers")
    assert not any(".seg_layers." in k for k in tiny_model.state_dict())
    assert tiny_model.head.out_channels == 1


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def test_resencmae_forward_shapes(tiny_model):
    x = torch.randn(2, 1, *IMG_SIZE)
    out = tiny_model(x)
    assert set(out) == {"reconstruction", "mask"}
    assert out["reconstruction"].shape == x.shape
    assert out["mask"].shape == (2, 1, *IMG_SIZE)
    assert ((out["mask"] == 0.0) | (out["mask"] == 1.0)).all()


def test_resencmae_forward_same_output_in_eval_mode(tiny_model):
    """forward returns the reconstruction dict in eval mode too (evaluate/inspect)."""
    tiny_model.eval()
    try:
        out = tiny_model(torch.randn(1, 1, *IMG_SIZE))
    finally:
        tiny_model.train()
    assert out["reconstruction"].shape == (1, 1, *IMG_SIZE)


def test_resencmae_every_parameter_receives_gradient():
    """No dead parameters — DDP (find_unused_parameters=False) would reject them."""
    model = ResEncMAE(img_size=IMG_SIZE, mask_patch_size=MASK_PATCH_SIZE)
    out = model(torch.randn(1, 1, *IMG_SIZE))
    out["reconstruction"].square().mean().backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


def test_resencmae_mask_ratio_override(tiny_model):
    x = torch.randn(1, 1, *IMG_SIZE)
    dense = tiny_model(x, mask_ratio=0.75)["mask"].mean().item()
    sparse = tiny_model(x, mask_ratio=0.125)["mask"].mean().item()
    assert dense == pytest.approx(0.75)
    assert sparse == pytest.approx(0.125)


def test_resencmae_generate_mask_delegates_to_shared_masking(tiny_model):
    x = torch.randn(2, 1, *IMG_SIZE)
    torch.manual_seed(7)
    from_model = tiny_model.generate_mask(x)
    torch.manual_seed(7)
    from_shared = generate_patch_mask(x, MASK_PATCH_SIZE, 0.75)
    assert torch.equal(from_model, from_shared)


def test_resencmae_encoder_returns_bottleneck_last(tiny_model):
    """misfit_encode / misfit_embed read model.encoder(x)[-1] as the bottleneck."""
    feats = tiny_model.encoder(torch.randn(1, 1, *IMG_SIZE))
    assert feats[-1].shape == (1, 256, 4, 4, 4)


# ---------------------------------------------------------------------------
# Registry and MIST transfer
# ---------------------------------------------------------------------------

def test_resenc_registry_builds_resencmae():
    model = get_model_from_registry(
        "resenc", in_channels=1, img_size=IMG_SIZE,
        mask_patch_size=MASK_PATCH_SIZE, mask_ratio=0.5,
    )
    assert isinstance(model, ResEncMAE)
    assert model.mask_ratio == 0.5


def test_get_encoder_state_dict_is_encoder_only(tiny_model):
    sd = tiny_model.get_encoder_state_dict()
    assert sd and all(k.startswith("encoder.") for k in sd)
    assert set(sd) == {k for k in tiny_model.state_dict() if k.startswith("encoder.")}


def test_misfit_resenc_encoder_matches_mist_resenc_exactly(tmp_path):
    """The encoder is a drop-in for MIST's resenc planned at the same patch/spacing.

    Keys and shapes must equal MIST's own encoder state dict exactly, and MIST's
    pretrained-encoder loader must transfer every tensor with nothing skipped.
    """
    import mist.models  # noqa: F401 — trigger MIST model registrations
    from mist.models import model_loader
    from mist.models.model_registry import get_model_from_registry as get_mist_model

    misfit_model = get_model_from_registry(
        "resenc", in_channels=1, img_size=IMG_SIZE,
        mask_patch_size=MASK_PATCH_SIZE, mask_ratio=0.75,
    )
    mist_model = get_mist_model(
        "resenc", in_channels=1, out_channels=3,
        patch_size=list(IMG_SIZE), target_spacing=list(PRETRAINING_SPACING),
    )

    misfit_sd = misfit_model.get_encoder_state_dict()
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
