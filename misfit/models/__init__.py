"""MISFIT model registry — importing this module triggers all registrations."""

# Import registry modules to trigger @register_model decorators.
import misfit.models.primus.primus_registry  # noqa: F401
import misfit.models.resenc.resenc_registry  # noqa: F401
import misfit.models.swinunetr.swinunetr_registry  # noqa: F401
