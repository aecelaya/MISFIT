"""Registry entries for the Primus MAE variants.

The names match MIST's models, so an encoder pretrained here fine-tunes in MIST
under the same architecture name. Encoder sizes come from MIST. Only Primus (V1)
is registered; see :mod:`misfit.models.primus.misfit_primus_mae`.

Decoder sizes: nnssl's defaults for medium and nnFoundationViT's for x. nnssl has
no preset for large, so its 4 blocks are this module's choice, in proportion to
the other two (about a fifth of the encoder's depth or less).
"""

from mist.models.primus.primus_constants import PrimusConstants

from misfit.models.model_registry import register_model
from misfit.models.primus.misfit_primus_mae import PrimusMAE


def _create_primus_mae(
    variant: str, decoder_depth: int, decoder_num_heads: int, **kwargs
) -> PrimusMAE:
    """Build a PrimusMAE with one of MIST's Primus encoder sizes."""
    depth, num_heads, embed_dim = PrimusConstants.VARIANTS[variant]
    return PrimusMAE(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
        **kwargs,
    )


@register_model("primus-medium")
def _primusmae_medium(**kwargs) -> PrimusMAE:
    """Primus-M with an MAE objective (16 encoder blocks, dim 864)."""
    return _create_primus_mae("medium", decoder_depth=2, decoder_num_heads=12, **kwargs)


@register_model("primus-large")
def _primusmae_large(**kwargs) -> PrimusMAE:
    """Primus-L with an MAE objective (24 encoder blocks, dim 1056)."""
    return _create_primus_mae("large", decoder_depth=4, decoder_num_heads=16, **kwargs)


@register_model("primus-x")
def _primusmae_x(**kwargs) -> PrimusMAE:
    """PrimusX with an MAE objective (40 encoder blocks, dim 1056): nnFoundationViT."""
    return _create_primus_mae("x", decoder_depth=8, decoder_num_heads=16, **kwargs)
