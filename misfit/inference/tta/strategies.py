"""Test-time augmentation (TTA) strategies for MISFIT.

A strategy is a named, callable object that returns a list of
:class:`~misfit.inference.tta.transforms.AbstractTransform` instances.
The :class:`~misfit.inference.predictor.Predictor` iterates over this list,
running the model once per transform and ensembling the un-augmented outputs.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type, TypeVar

from misfit.inference.tta.transforms import AbstractTransform, get_transform

T = TypeVar("T", bound="TTAStrategy")
TTA_STRATEGY_REGISTRY: Dict[str, Type["TTAStrategy"]] = {}


def register_strategy(name: str) -> Callable[[Type[T]], Type[T]]:
    """Class decorator — registers a TTA strategy by name."""
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, TTAStrategy):
            raise TypeError(f"{cls.__name__} must inherit from TTAStrategy.")
        if name in TTA_STRATEGY_REGISTRY:
            raise KeyError(f"Strategy '{name}' is already registered.")
        TTA_STRATEGY_REGISTRY[name] = cls
        return cls
    return decorator


def list_strategies() -> List[str]:
    """Return a list of all registered TTA strategy names."""
    return list(TTA_STRATEGY_REGISTRY.keys())


def get_strategy(name: str) -> "TTAStrategy":
    """Return a fresh instance of the named TTA strategy.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in TTA_STRATEGY_REGISTRY:
        raise KeyError(
            f"TTA strategy '{name}' is not registered. "
            f"Available: {list_strategies()}"
        )
    return TTA_STRATEGY_REGISTRY[name]()


class TTAStrategy(ABC):
    """Abstract base class for TTA strategies."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def get_transforms(self) -> List[AbstractTransform]:
        """Return the list of transforms to apply at inference time."""

    def __call__(self) -> List[AbstractTransform]:
        return self.get_transforms()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, TTAStrategy) and self.name == other.name


@register_strategy("none")
class NoTTAStrategy(TTAStrategy):
    """Single forward pass — no augmentation."""
    def get_transforms(self) -> List[AbstractTransform]:
        return [get_transform("identity")]


@register_strategy("all_flips")
class AllFlipsStrategy(TTAStrategy):
    """All 7 axis-flip combinations plus the identity (8 passes total)."""
    def get_transforms(self) -> List[AbstractTransform]:
        return [
            get_transform("identity"),
            get_transform("flip_x"),
            get_transform("flip_y"),
            get_transform("flip_z"),
            get_transform("flip_xy"),
            get_transform("flip_xz"),
            get_transform("flip_yz"),
            get_transform("flip_xyz"),
        ]
