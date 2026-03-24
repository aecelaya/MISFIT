"""Registry for MISFIT prediction ensemblers."""
from typing import Callable, Dict, List, Type, TypeVar

from misfit.inference.ensemblers.base import AbstractEnsembler

T = TypeVar("T", bound=AbstractEnsembler)
ENSEMBLER_REGISTRY: Dict[str, Type[AbstractEnsembler]] = {}


def register_ensembler(name: str) -> Callable[[Type[T]], Type[T]]:
    """Class decorator — registers an :class:`AbstractEnsembler` subclass."""
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, AbstractEnsembler):
            raise TypeError(f"{cls.__name__} must inherit from AbstractEnsembler.")
        if name in ENSEMBLER_REGISTRY:
            raise KeyError(f"Ensembler '{name}' is already registered.")
        ENSEMBLER_REGISTRY[name] = cls
        return cls
    return decorator


def list_ensemblers() -> List[str]:
    """Return a list of all registered ensembler names."""
    return list(ENSEMBLER_REGISTRY.keys())


def get_ensembler(name: str) -> AbstractEnsembler:
    """Return a fresh instance of the named ensembler.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in ENSEMBLER_REGISTRY:
        raise KeyError(
            f"Ensembler '{name}' is not registered. "
            f"Available: {list_ensemblers()}"
        )
    return ENSEMBLER_REGISTRY[name]()
