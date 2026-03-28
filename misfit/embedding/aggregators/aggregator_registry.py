"""Registry for MISFIT crop aggregators."""
from typing import Callable, Dict, List, Type, TypeVar

from misfit.embedding.aggregators.base import AbstractAggregator

T = TypeVar("T", bound=AbstractAggregator)
AGGREGATOR_REGISTRY: Dict[str, Type[AbstractAggregator]] = {}


def register_aggregator(name: str) -> Callable[[Type[T]], Type[T]]:
    """Class decorator — registers an :class:`AbstractAggregator` subclass."""
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, AbstractAggregator):
            raise TypeError(f"{cls.__name__} must inherit from AbstractAggregator.")
        if name in AGGREGATOR_REGISTRY:
            raise KeyError(f"Aggregator '{name}' is already registered.")
        AGGREGATOR_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def list_aggregators() -> List[str]:
    """Return a sorted list of all registered aggregator names."""
    return sorted(AGGREGATOR_REGISTRY.keys())


def get_aggregator(name: str) -> Type[AbstractAggregator]:
    """Return the aggregator *class* for *name* (caller supplies embed_dim).

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in AGGREGATOR_REGISTRY:
        raise KeyError(
            f"Aggregator '{name}' is not registered. "
            f"Available: {list_aggregators()}"
        )
    return AGGREGATOR_REGISTRY[name]
