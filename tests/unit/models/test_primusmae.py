"""Tests for misfit.models.primus (PrimusMAE) and its registry entries."""
from unittest.mock import patch

import pytest
import torch
from mist.models.primus.primus_constants import PrimusConstants

import misfit.models  # noqa — trigger registrations
from misfit.models.masking import generate_patch_mask
from misfit.models.model_registry import get_model_from_registry
from misfit.models.pretrained import load_model_weights
from misfit.models.primus import primus_registry
from misfit.models.primus.misfit_primus_mae import PrimusMAE, resize_position_embedding

IMG_SIZE = (32, 32, 32)
TOKEN_GRID = (4, 4, 4)
EMBED_DIM = 48


def _tiny(**overrides):
    """A one-block PrimusMAE, far smaller than any registered variant."""
    kwargs = {
        "embed_dim": EMBED_DIM, "depth": 1, "num_heads": 4, "decoder_depth": 1,
        "decoder_num_heads": 4, "img_size": IMG_SIZE, "mask_patch_size": 16,
        "mask_ratio": 0.75,
    }
    return PrimusMAE(**{**kwargs, **overrides})


@pytest.fixture(scope="module")
def tiny_model():
    return _tiny().eval()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_primusmae_rejects_mask_patch_size_splitting_a_token():
    with pytest.raises(ValueError, match="multiple of the 8-voxel"):
        _tiny(mask_patch_size=12)


def test_primusmae_rejects_img_size_not_divisible_by_mask_patch_size():
    with pytest.raises(ValueError, match="mask_patch_size=16"):
        _tiny(img_size=(40, 32, 32))


def test_primusmae_rejects_8_voxel_cubes_on_a_size_they_do_not_divide():
    with pytest.raises(ValueError, match="mask_patch_size=8"):
        _tiny(img_size=(36, 32, 32), mask_patch_size=8)


def test_primusmae_uses_mist_primus_modules():
    """The encoder and patch decoder are MIST's own modules."""
    from mist.models.primus.eva_blocks import Eva
    from mist.models.primus.primus_blocks import PatchDecode, PatchEmbed

    model = _tiny()
    assert isinstance(model.down_projection, PatchEmbed)
    assert isinstance(model.eva, Eva)
    assert isinstance(model.up_projection, PatchDecode)
    assert isinstance(model.decoder, Eva)


def test_primusmae_mask_token_is_a_learned_token():
    assert _tiny().mask_token.shape == (1, 1, EMBED_DIM)
    assert "mask_token" in dict(_tiny().named_parameters())


@pytest.mark.parametrize(
    ("name", "depth", "heads", "dim", "decoder_depth", "decoder_heads"),
    [
        ("primus-medium", 16, 12, 864, 2, 12),
        ("primus-large", 24, 16, 1056, 4, 16),
        ("primus-x", 40, 16, 1056, 8, 16),
    ],
)
def test_primus_registry_entries(name, depth, heads, dim, decoder_depth, decoder_heads):
    """Each entry uses MIST's encoder size, and forwards the caller's arguments."""
    with patch.object(primus_registry, "PrimusMAE") as model_cls:
        model = get_model_from_registry(
            name, in_channels=1, img_size=IMG_SIZE, mask_patch_size=8, mask_ratio=0.8,
            target_spacing=None,
        )
    assert model is model_cls.return_value
    model_cls.assert_called_once_with(
        embed_dim=dim, depth=depth, num_heads=heads, decoder_depth=decoder_depth,
        decoder_num_heads=decoder_heads, in_channels=1, img_size=IMG_SIZE,
        mask_patch_size=8, mask_ratio=0.8, target_spacing=None,
    )
    assert PrimusConstants.VARIANTS[name.removeprefix("primus-")] == (depth, heads, dim)


def test_primus_registry_builds_primusmae_and_ignores_target_spacing():
    """--target-spacing is passed to every model; Primus's topology doesn't use it."""
    with patch.dict(PrimusConstants.VARIANTS, {"medium": (1, 4, EMBED_DIM)}):
        model = primus_registry._create_primus_mae(
            "medium", decoder_depth=1, decoder_num_heads=4, img_size=IMG_SIZE,
            target_spacing=(1.0, 1.0, 2.5),
        )
    assert isinstance(model, PrimusMAE)


# ---------------------------------------------------------------------------
# Forward contract
# ---------------------------------------------------------------------------


def test_primusmae_forward_shapes(tiny_model):
    x = torch.randn(2, 1, *IMG_SIZE)
    with torch.no_grad():
        out = tiny_model(x)
    assert set(out) == {"reconstruction", "mask"}
    assert out["reconstruction"].shape == (2, 1, *IMG_SIZE)
    assert out["mask"].shape == (2, 1, *IMG_SIZE)


def test_primusmae_masks_in_train_and_eval_mode():
    """Unlike timm's patch dropout, masking isn't disabled in eval mode."""
    model = _tiny()
    x = torch.randn(1, 1, *IMG_SIZE)
    for mode in (model.train, model.eval):
        mode()
        assert model(x)["mask"].mean().item() == pytest.approx(0.75)


def test_primusmae_mask_ratio_override(tiny_model):
    x = torch.randn(1, 1, *IMG_SIZE)
    with torch.no_grad():
        assert tiny_model(x, mask_ratio=0.5)["mask"].mean().item() == pytest.approx(0.5)


def test_primusmae_generate_mask_delegates_to_shared_masking(tiny_model):
    x = torch.randn(2, 1, *IMG_SIZE)
    torch.manual_seed(0)
    actual = tiny_model.generate_mask(x)
    torch.manual_seed(0)
    expected = generate_patch_mask(x, 16, 0.75)
    assert torch.equal(actual, expected)


def test_primusmae_rejects_masking_every_token(tiny_model):
    with pytest.raises(ValueError, match="masks every token"):
        tiny_model(torch.randn(1, 1, *IMG_SIZE), mask_ratio=1.0)


def test_primusmae_every_parameter_receives_gradient():
    """No dead parameters — DDP (find_unused_parameters=False) would reject them."""
    model = _tiny()
    model(torch.randn(2, 1, *IMG_SIZE))["reconstruction"].square().mean().backward()
    assert [n for n, p in model.named_parameters() if p.grad is None] == []


# ---------------------------------------------------------------------------
# Token masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mask_patch_size", [8, 16])
def test_visible_token_indices_match_the_voxel_mask(mask_patch_size):
    """Each sample's visible tokens are exactly the voxel mask's visible cubes."""
    model = _tiny(mask_patch_size=mask_patch_size)
    x = torch.randn(3, 1, *IMG_SIZE)
    mask = model.generate_mask(x)
    visible = model._visible_token_indices(mask)

    for sample in range(3):
        expected = torch.ones(TOKEN_GRID)
        for flat in visible[sample].tolist():
            d, h, w = torch.unravel_index(torch.tensor(flat), TOKEN_GRID)
            expected[d, h, w] = 0
        tokens_as_voxels = (
            expected.repeat_interleave(8, 0).repeat_interleave(8, 1).repeat_interleave(8, 2)
        )
        assert torch.equal(mask[sample, 0], tokens_as_voxels)


def test_masked_tokens_do_not_reach_the_encoder():
    """Changing only masked voxels leaves the encoded visible tokens unchanged."""
    model = _tiny().eval()
    x = torch.randn(1, 1, *IMG_SIZE)
    mask = model.generate_mask(x)
    visible = model._visible_token_indices(mask)
    changed = x + mask * torch.randn_like(x) * 10

    def encode(image):
        tokens = model.down_projection(image).flatten(2).transpose(1, 2)
        return model._encode_visible(tokens, visible)

    with torch.no_grad():
        assert torch.equal(encode(x), encode(changed))


def test_encode_visible_with_every_token_matches_mist_eva():
    """With nothing dropped, the manual encoder path is MIST's Eva.forward."""
    model = _tiny().eval()
    tokens = torch.randn(2, 64, EMBED_DIM)
    every_token = torch.arange(64).expand(2, -1)
    with torch.no_grad():
        assert torch.allclose(
            model._encode_visible(tokens, every_token), model.eva(tokens), atol=1e-6
        )


def test_encoded_tokens_land_in_their_own_positions():
    """Visible tokens return to their original slots; the rest are mask tokens."""
    model = _tiny().eval()
    x = torch.randn(1, 1, *IMG_SIZE)

    torch.manual_seed(1)
    with patch.object(model.decoder, "forward", side_effect=lambda t: t) as decoder:
        with torch.no_grad():
            model(x)
    restored = decoder.call_args.args[0][0]
    torch.manual_seed(1)
    visible = model._visible_token_indices(model.generate_mask(x))

    masked = sorted(set(range(64)) - set(visible[0].tolist()))
    assert torch.all(restored[masked] == model.mask_token[0, 0])
    tokens = model.down_projection(x).flatten(2).transpose(1, 2)
    with torch.no_grad():
        encoded = model._encode_visible(tokens, visible)
    assert torch.allclose(restored[visible[0]], encoded[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


def test_primusmae_encoder_returns_bottleneck_last(tiny_model):
    """model.encoder(x)[-1] is the spatial feature map encode and embed use."""
    with torch.no_grad():
        features = tiny_model.encoder(torch.randn(2, 1, *IMG_SIZE))
    assert features[-1].shape == (2, EMBED_DIM, *TOKEN_GRID)


def test_primusmae_encoder_is_not_a_module_in_the_state_dict(tiny_model):
    """Parameter names stay MIST's and nnssl's; there is no encoder.* prefix."""
    assert not [k for k in tiny_model.state_dict() if k.startswith("encoder.")]


def test_primusmae_encoder_keeps_tokens_in_spatial_order():
    """Token i of the sequence lands at its own (D, H, W) grid position."""
    model = _tiny().eval()
    tokens = torch.arange(64.0).reshape(1, 1, 4, 4, 4).expand(1, EMBED_DIM, -1, -1, -1)
    with patch.object(model.eva, "forward", side_effect=lambda t: t), patch.object(
        model.down_projection, "forward", side_effect=lambda _: tokens
    ):
        grid = model.encoder(torch.zeros(1, 1, *IMG_SIZE))[-1]
    assert torch.equal(grid, tokens)


def test_get_encoder_state_dict_is_encoder_only(tiny_model):
    keys = list(tiny_model.get_encoder_state_dict())
    assert keys
    assert all(k.startswith(("down_projection.", "eva.")) for k in keys)


def test_misfit_primus_encoder_matches_mist_primus_exactly(tmp_path):
    """The encoder is a drop-in for MIST's Primus of the same size.

    Keys and shapes must equal MIST's own encoder state dict exactly, and MIST's
    pretrained-encoder loader must transfer every tensor with nothing skipped.
    """
    from mist.models import model_loader
    from mist.models.primus.mist_primus import Primus

    misfit_model = _tiny()
    mist_model = Primus(
        in_channels=1, out_channels=3, patch_size=IMG_SIZE,
        embed_dim=EMBED_DIM, depth=1, num_heads=4,
    )
    misfit_sd = misfit_model.get_encoder_state_dict()
    mist_sd = mist_model.get_encoder_state_dict()
    assert list(misfit_sd) == list(mist_sd)
    for key, tensor in misfit_sd.items():
        assert tensor.shape == mist_sd[key].shape, key

    weights_path = tmp_path / "misfit_encoder.pt"
    torch.save(misfit_sd, weights_path)
    mist_model, summary = model_loader.load_pretrained_encoder(mist_model, str(weights_path))
    assert len(summary["loaded"]) == len(mist_sd)
    assert summary["skipped"] == []
    for key, tensor in misfit_sd.items():
        assert torch.equal(mist_model.state_dict()[key], tensor), key


# ---------------------------------------------------------------------------
# Loading pretrained weights
# ---------------------------------------------------------------------------


def test_resize_position_embedding_to_a_new_grid():
    resized = resize_position_embedding(torch.randn(1, 216, 12), (4, 4, 2))
    assert resized.shape == (1, 32, 12)


def test_resize_position_embedding_preserves_a_constant():
    source = torch.full((1, 216, 12), 3.0, dtype=torch.float64)
    resized = resize_position_embedding(source, (4, 4, 4))
    assert resized.dtype == torch.float64
    assert torch.allclose(resized, torch.full_like(resized, 3.0))


def test_resize_position_embedding_leaves_a_matching_size_alone():
    source = torch.randn(1, 64, 12)
    assert resize_position_embedding(source, (4, 4, 4)) is source


def test_resize_position_embedding_leaves_a_non_cubic_source_alone():
    source = torch.randn(1, 32, 12)  # 4 x 4 x 2 can't be recovered from 32.
    assert resize_position_embedding(source, (4, 4, 4)) is source


def test_adapt_pretrained_state_dict_resizes_both_position_embeddings():
    model = _tiny()
    source = _tiny(img_size=(48, 48, 48)).state_dict()
    adapted = model.adapt_pretrained_state_dict(source)
    assert adapted["eva.pos_embed"].shape == model.eva.pos_embed.shape
    assert adapted["decoder.pos_embed"].shape == model.decoder.pos_embed.shape
    assert source["eva.pos_embed"].shape[1] == 216  # The input isn't modified.


def test_adapt_pretrained_state_dict_tolerates_absent_position_embeddings(tiny_model):
    assert tiny_model.adapt_pretrained_state_dict({"other": torch.zeros(1)}).keys() == {"other"}


def test_loads_a_checkpoint_from_another_patch_size():
    """Everything but the position embeddings transfers unchanged."""
    source = _tiny(img_size=(48, 48, 48))
    model = _tiny()
    load_model_weights(model, {"network_weights": source.state_dict()})
    for key, tensor in source.state_dict().items():
        if not key.endswith("pos_embed"):
            assert torch.equal(model.state_dict()[key], tensor), key


def test_loading_a_different_depth_fails_strictly():
    """nnFoundationViT in a smaller variant must not silently truncate."""
    with pytest.raises(ValueError, match="unexpected"):
        load_model_weights(_tiny(), _tiny(depth=2).state_dict())
