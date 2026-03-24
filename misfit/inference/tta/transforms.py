"""Test-time augmentation (TTA) transforms for MISFIT.

Each transform has a ``forward`` method that augments the input and an
``inverse`` method that un-augments the model output, so the final
prediction is always in the original volume's coordinate system.

The flip transforms work on any 5-D tensor ``(B, C, D, H, W)`` and are
therefore compatible with both reconstruction outputs and feature maps.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Type, TypeVar

import torch

T = TypeVar("T", bound="AbstractTransform")
TTA_TRANSFORM_REGISTRY: Dict[str, Type["AbstractTransform"]] = {}


def register_transform(name: str) -> Callable[[Type[T]], Type[T]]:
    """Class decorator — registers a TTA transform by name."""
    def decorator(cls: Type[T]) -> Type[T]:
        if not issubclass(cls, AbstractTransform):
            raise TypeError(f"{cls.__name__} must subclass AbstractTransform.")
        if name in TTA_TRANSFORM_REGISTRY:
            raise KeyError(f"Transform '{name}' is already registered.")
        TTA_TRANSFORM_REGISTRY[name] = cls
        return cls
    return decorator


def list_transforms() -> List[str]:
    """Return a list of all registered TTA transform names."""
    return list(TTA_TRANSFORM_REGISTRY.keys())


def get_transform(name: str) -> "AbstractTransform":
    """Return a fresh instance of the named TTA transform.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in TTA_TRANSFORM_REGISTRY:
        raise KeyError(
            f"TTA transform '{name}' is not registered. "
            f"Available: {list_transforms()}"
        )
    return TTA_TRANSFORM_REGISTRY[name]()


class AbstractTransform(ABC):
    """Abstract base class for invertible TTA transforms."""

    def __init__(self) -> None:
        self.name = self.__class__.__name__.lower()

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the transform to an input image tensor."""

    @abstractmethod
    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        """Invert the transform on a prediction tensor."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.forward(image)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AbstractTransform) and self.name == other.name


@register_transform("identity")
class IdentityTransform(AbstractTransform):
    """No-op transform — passes through unchanged."""
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return prediction


@register_transform("flip_x")
class FlipXTransform(AbstractTransform):
    """Flip along the depth (D) axis."""
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(2,))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(2,))


@register_transform("flip_y")
class FlipYTransform(AbstractTransform):
    """Flip along the height (H) axis."""
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(3,))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(3,))


@register_transform("flip_z")
class FlipZTransform(AbstractTransform):
    """Flip along the width (W) axis."""
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(4,))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(4,))


@register_transform("flip_xy")
class FlipXYTransform(AbstractTransform):
    """Flip along the D and H axes."""
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(2, 3))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(2, 3))


@register_transform("flip_xz")
class FlipXZTransform(AbstractTransform):
    """Flip along the D and W axes."""
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(2, 4))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(2, 4))


@register_transform("flip_yz")
class FlipYZTransform(AbstractTransform):
    """Flip along the H and W axes."""
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(3, 4))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(3, 4))


@register_transform("flip_xyz")
class FlipXYZTransform(AbstractTransform):
    """Flip along all three spatial axes."""
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(2, 3, 4))

    def inverse(self, prediction: torch.Tensor) -> torch.Tensor:
        return torch.flip(prediction, dims=(2, 3, 4))
