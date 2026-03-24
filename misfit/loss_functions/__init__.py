"""MISFIT loss function registry — importing this module triggers all registrations."""

# Import reconstruction losses to trigger @register_loss decorators.
import misfit.loss_functions.reconstruction.masked_mse      # noqa: F401
import misfit.loss_functions.reconstruction.masked_l1       # noqa: F401
import misfit.loss_functions.reconstruction.normalized_mse  # noqa: F401
