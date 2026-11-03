"""Tests for misfit.models.pretrained (MISFIT and nnssl checkpoint loading)."""
from collections import OrderedDict

import pytest
import torch
from mist.models.resenc.mist_resenc import ResEncUNet

from misfit.models.pretrained import (
    extract_state_dict,
    load_model_weights,
    load_pretrained_weights,
)
from misfit.models.resenc.misfit_resenc_mae import ResEncMAE
from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE

PATCH = (32, 32, 32)


def _resenc(patch=PATCH):
    return ResEncMAE(in_channels=1, img_size=patch, mask_patch_size=16)


def _nnssl_state_dict(patch=PATCH):
    """A state dict named as nnssl/nnU-Net saves a ResEnc MAE network.

    MIST's full ResEncUNet uses nnU-Net's names, including every
    deep-supervision segmentation layer; nnU-Net's decoder also registers the
    encoder a second time under ``decoder.encoder.*``.
    """
    torch.manual_seed(0)
    net = ResEncUNet(1, 1, list(patch), [1.0, 1.0, 1.0], use_deep_supervision=False)
    state_dict = OrderedDict(net.state_dict())
    for k, v in net.encoder.state_dict().items():
        state_dict[f"decoder.encoder.{k}"] = v
    return net, state_dict


# --- extract_state_dict -----------------------------------------------------

@pytest.mark.parametrize("container", ["model", "network_weights", "state_dict"])
def test_extract_state_dict_unwraps_containers(container):
    sd = {"w": torch.ones(1)}
    assert extract_state_dict({container: sd, "epoch": 3}) == sd


def test_extract_state_dict_plain_and_ddp_prefix():
    sd = {"module.encoder.w": torch.ones(1), "head.b": torch.zeros(1)}
    assert list(extract_state_dict(sd)) == ["encoder.w", "head.b"]


def test_extract_state_dict_raises_without_weights():
    with pytest.raises(ValueError, match="No model weights"):
        extract_state_dict({"epoch": 3, "citations": []})


# --- load_model_weights -------------------------------------------------------

def test_loads_misfit_checkpoint_exactly():
    source = _resenc()
    target = _resenc()
    load_model_weights(target, {"model": source.state_dict(), "epoch": 1})
    for k, v in source.state_dict().items():
        assert torch.equal(target.state_dict()[k], v), k


def test_loads_nnssl_checkpoint_and_maps_the_head():
    """nnssl names load in full: the last seg layer becomes the head."""
    net, nnssl_sd = _nnssl_state_dict()
    model = _resenc()
    load_model_weights(model, {"network_weights": nnssl_sd, "citations": []})

    x = torch.randn(1, 1, *PATCH)
    net.eval()
    model.eval()
    with torch.no_grad():
        expected = net(x)
        actual = model.head(model.decoder(model.encoder(x))[0])
    assert torch.equal(actual, expected)
    # mask_token isn't in nnssl checkpoints; it keeps its zero init.
    assert torch.all(model.mask_token == 0)


def test_rejects_a_different_topology_with_a_hint():
    _, nnssl_sd = _nnssl_state_dict(patch=(64, 64, 64))
    with pytest.raises(ValueError, match="no --target-spacing"):
        load_model_weights(_resenc(), {"network_weights": nnssl_sd})


def test_rejects_missing_and_unexpected_weights():
    sd = _resenc().state_dict()
    del sd["head.weight"]
    with pytest.raises(ValueError, match="missing"):
        load_model_weights(_resenc(), sd)

    sd = _resenc().state_dict()
    sd["extra.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        load_model_weights(_resenc(), sd)


def test_swinmae_drops_legacy_spacing_embed():
    kwargs = dict(in_channels=1, feature_size=12, img_size=PATCH, mask_patch_size=16)
    source = SwinMAE(**kwargs)
    sd = OrderedDict(source.state_dict())
    sd["spacing_embed.0.weight"] = torch.zeros(2, 3)
    target = SwinMAE(**kwargs)
    load_model_weights(target, {"model": sd})
    assert torch.equal(
        target.state_dict()["mask_token"], source.state_dict()["mask_token"]
    )


def test_load_pretrained_weights_reads_a_file(tmp_path):
    source = _resenc()
    path = tmp_path / "checkpoint_final.pth"
    torch.save({"network_weights": source.state_dict()}, path)
    target = _resenc()
    load_pretrained_weights(target, path)
    assert torch.equal(
        target.state_dict()["head.weight"], source.state_dict()["head.weight"]
    )
