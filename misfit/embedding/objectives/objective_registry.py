"""Registry for MISFIT embedding training objectives."""
from typing import Callable, Dict, List, Type, TypeVar

from misfit.embedding.objectives.base import TrainingObjective

T = TypeVar("T", bound=TrainingObjective)
OBJECTIVE_REGISTRY: Dict[str, Type[TrainingObjective]] = {}


def register_objective(name: str) -> Callable[[Type[T]], Type[T]]:
    """Class decorator — registers a :class:`TrainingObjective` subclass."""
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, TrainingObjective):
            raise TypeError(f"{cls.__name__} must inherit from TrainingObjective.")
        if name in OBJECTIVE_REGISTRY:
            raise KeyError(f"Objective '{name}' is already registered.")
        OBJECTIVE_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def list_objectives() -> List[str]:
    """Return a sorted list of all registered objective names."""
    return sorted(OBJECTIVE_REGISTRY.keys())


def get_objective(name: str) -> Type[TrainingObjective]:
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
