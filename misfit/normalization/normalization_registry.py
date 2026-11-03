"""Normalization policy registry for MISFIT."""
from collections.abc import Callable

from misfit.normalization.base import NormalizationPolicy

# Maps policy name strings to NormalizationPolicy subclasses.
NORMALIZATION_POLICY_REGISTRY: dict[str, type[NormalizationPolicy]] = {}

# The policy MISFIT used before policies were selectable; also what configs
# without a normalization_policy entry were trained with.
DEFAULT_NORMALIZATION_POLICY = "clip_and_z_score"


def register_normalization_policy(name: str) -> Callable:
    """Decorator to register a NormalizationPolicy subclass.

    Args:
        name: Unique string identifier for the policy (case-insensitive lookup).

    Returns:
        The original class, unmodified.

    Raises:
        ValueError: If a policy with this name is already registered.
    """
    def decorator(cls: type[NormalizationPolicy]) -> type[NormalizationPolicy]:
        key = name.lower()
        if key in NORMALIZATION_POLICY_REGISTRY:
            raise ValueError(f"Normalization policy '{name}' is already registered.")
        NORMALIZATION_POLICY_REGISTRY[key] = cls
        return cls
    return decorator


def get_normalization_policy(name: str) -> NormalizationPolicy:
    """Instantiate a registered normalization policy.

    Args:
        name: Registered policy name (case-insensitive).

    Returns:
        A NormalizationPolicy instance.

    Raises:
        ValueError: If the name is not registered.
    """
    key = name.lower()
    if key not in NORMALIZATION_POLICY_REGISTRY:
        raise ValueError(
            f"Normalization policy '{name}' is not registered.\n"
            f"Available policies: {sorted(NORMALIZATION_POLICY_REGISTRY.keys())}"
        )
    return NORMALIZATION_POLICY_REGISTRY[key]()


def list_normalization_policies() -> list[str]:
    """List all registered normalization policy names.

    Returns:
        Sorted list of registered policy name strings.
    """
    return sorted(NORMALIZATION_POLICY_REGISTRY.keys())


def get_normalization_policy_name(config: dict) -> str:
    """Return the normalization policy saved in a ``config.json`` dict.

    Configs written before policies were selectable have no
    ``data.normalization_policy``; they used ``clip_and_z_score``.

    Args:
        config: Full config dict loaded from ``config.json``.

    Returns:
        The policy name.
    """
    return config.get("data", {}).get(
        "normalization_policy", DEFAULT_NORMALIZATION_POLICY
    )
