"""Loss function registry for MISFIT."""
from collections.abc import Callable

from misfit.loss_functions.base import ReconstructionLoss

# Maps loss name strings to ReconstructionLoss subclasses.
LOSS_REGISTRY: dict[str, type[ReconstructionLoss]] = {}


def register_loss(name: str) -> Callable:
    """Decorator to register a ReconstructionLoss subclass.

    Args:
        name: Unique string identifier for the loss (case-insensitive lookup).

    Returns:
        The original class, unmodified.

    Raises:
        ValueError: If a loss with this name is already registered.
    """
    def decorator(cls: type[ReconstructionLoss]) -> type[ReconstructionLoss]:
        key = name.lower()
        if key in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered.")
        LOSS_REGISTRY[key] = cls
        return cls
    return decorator


def get_loss(name: str) -> type[ReconstructionLoss]:
    """Retrieve a loss class from the registry.

    Args:
        name: Registered loss name (case-insensitive).

    Returns:
        The ReconstructionLoss subclass. Caller is responsible for
        instantiation (e.g., ``get_loss("masked_mse")()``).

    Raises:
        ValueError: If the name is not registered.
    """
    key = name.lower()
    if key not in LOSS_REGISTRY:
        raise ValueError(
            f"Loss '{name}' is not registered.\n"
            f"Available losses: {sorted(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[key]


def list_registered_losses() -> list[str]:
    """List all registered loss names.

    Returns:
        Sorted list of registered loss name strings.
    """
    return sorted(LOSS_REGISTRY.keys())
