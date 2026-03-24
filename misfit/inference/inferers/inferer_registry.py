"""Registry for MISFIT inference strategies."""
from typing import Callable, Dict, List, Type, TypeVar

from misfit.inference.inferers.base import AbstractInferer

T = TypeVar("T", bound=AbstractInferer)
INFERER_REGISTRY: Dict[str, Type[AbstractInferer]] = {}


def register_inferer(name: str) -> Callable[[Type[T]], Type[T]]:
    """Class decorator — registers an :class:`AbstractInferer` subclass."""
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, AbstractInferer):
            raise TypeError(f"{cls.__name__} must inherit from AbstractInferer.")
        if name in INFERER_REGISTRY:
            raise KeyError(f"Inferer '{name}' is already registered.")
        INFERER_REGISTRY[name] = cls
        return cls
    return decorator


def list_inferers() -> List[str]:
    """Return a list of all registered inferer names."""
    return list(INFERER_REGISTRY.keys())


def get_inferer(name: str) -> Type[AbstractInferer]:
    """Retrieve a registered inferer *class* by name.

    Returns the class (not an instance) so the caller can pass
    constructor arguments.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in INFERER_REGISTRY:
        raise KeyError(
            f"Inferer '{name}' is not registered. "
            f"Available: {list_inferers()}"
        )
    return INFERER_REGISTRY[name]
