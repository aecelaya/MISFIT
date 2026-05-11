"""Registry for MISFIT embedding training objectives."""
from collections.abc import Callable
from typing import TypeVar

from misfit.embedding.objectives.base import TrainingObjective

T = TypeVar("T", bound=TrainingObjective)
OBJECTIVE_REGISTRY: dict[str, type[TrainingObjective]] = {}


def register_objective(name: str) -> Callable[[type[T]], type[T]]:
    """Class decorator — registers a :class:`TrainingObjective` subclass."""
    def decorator(cls: type[T]) -> type[T]:
        if not issubclass(cls, TrainingObjective):
            raise TypeError(f"{cls.__name__} must inherit from TrainingObjective.")
        if name in OBJECTIVE_REGISTRY:
            raise KeyError(f"Objective '{name}' is already registered.")
        OBJECTIVE_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def list_objectives() -> list[str]:
    """Return a sorted list of all registered objective names."""
    return sorted(OBJECTIVE_REGISTRY.keys())


def get_objective(name: str) -> type[TrainingObjective]:
    """Return the objective *class* for *name*.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in OBJECTIVE_REGISTRY:
        raise KeyError(
            f"Objective '{name}' is not registered. "
            f"Available: {list_objectives()}"
        )
    return OBJECTIVE_REGISTRY[name]
