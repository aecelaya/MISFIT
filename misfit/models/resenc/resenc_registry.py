"""Registry entry for the ResEnc MAE model.

The name matches MIST's ``resenc`` model, so an encoder pretrained here
fine-tunes in MIST under the same architecture name. nnU-Net's ResEnc M/L/XL
presets share one network and differ only in planned patch and batch size, so
there is a single entry; the pretraining patch size is set by --patch-size.

``resenc-pocket`` matches MIST's model of that name: the same stages and blocks
with 32 filters in every stage (1.8M parameters at six stages, against 102M).
It needs MIST 3.0 or later.
"""

from misfit.models.model_registry import register_model
from misfit.models.resenc.misfit_resenc_mae import ResEncMAE


@register_model("resenc")
def _resencmae(**kwargs) -> ResEncMAE:
    """ResEnc UNet with an MAE objective (MIST/nnU-Net ResEnc network)."""
    return ResEncMAE(**kwargs)


@register_model("resenc-pocket")
def _resencmae_pocket(**kwargs) -> ResEncMAE:
    """ResEnc UNet with 32 filters per stage and an MAE objective."""
    return ResEncMAE(use_pocket_model=True, **kwargs)
