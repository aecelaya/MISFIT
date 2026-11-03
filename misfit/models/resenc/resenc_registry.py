"""Registry entry for the ResEnc MAE model.

The name matches MIST's ``resenc`` model, so an encoder pretrained here
fine-tunes in MIST under the same architecture name. nnU-Net's ResEnc M/L/XL
presets share one network and differ only in planned patch and batch size, so
there is a single entry; the pretraining patch size is set by --patch-size.
"""

from misfit.models.model_registry import register_model
from misfit.models.resenc.misfit_resenc_mae import ResEncMAE


@register_model("resenc")
def _resencmae(**kwargs) -> ResEncMAE:
    """ResEnc UNet with an MAE objective (MIST/nnU-Net ResEnc network)."""
    return ResEncMAE(**kwargs)
