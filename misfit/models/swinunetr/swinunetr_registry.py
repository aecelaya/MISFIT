"""Registry entries for SwinUNETR-V2 MAE variants.

Three sizes mirror the MIST SwinUNETR variants, making it straightforward to
pretrain a MISFIT encoder at a given capacity and fine-tune the corresponding
MIST SwinUNETR at the same feature_size.

    feature_size=24  →  swinmae-small   (~28M encoder parameters)
    feature_size=48  →  swinmae-base    (~62M encoder parameters)
    feature_size=96  →  swinmae-large   (~197M encoder parameters)
"""

from misfit.models.model_registry import register_model
from misfit.models.swinunetr.misfit_swinunetr_mae import SwinMAE


@register_model("swinmae-small")
def _swinmae_small(**kwargs) -> SwinMAE:
    """SwinMAE with feature_size=24 (~28M encoder parameters)."""
    return SwinMAE(feature_size=24, **kwargs)


@register_model("swinmae-base")
def _swinmae_base(**kwargs) -> SwinMAE:
    """SwinMAE with feature_size=48 (~62M encoder parameters)."""
    return SwinMAE(feature_size=48, **kwargs)


@register_model("swinmae-large")
def _swinmae_large(**kwargs) -> SwinMAE:
    """SwinMAE with feature_size=96 (~197M encoder parameters)."""
    return SwinMAE(feature_size=96, **kwargs)
