"""Abstract base class for all MISFIT inferers."""
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch


class AbstractInferer(ABC):
    """Abstract base class for MISFIT inference strategies.

    An inferer encapsulates *how* a model is applied to a tensor — e.g.
    a single whole-volume forward pass vs. a sliding-window sweep.  It is
    decoupled from *what* the model returns so that the same inferer works
    for both reconstruction and feature-extraction callables.
    """

    def __init__(self) -> None:
        self.name = self.__class__.__name__.lower()

    def __call__(
        self,
        image: torch.Tensor,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Invoke :meth:`infer` like a function."""
        return self.infer(image, model_fn)

    @abstractmethod
    def infer(
        self,
        image: torch.Tensor,
        model_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Apply *model_fn* to *image* and return the output tensor.

        Args:
            image: Input tensor of shape ``(1, C, D, H, W)``.
            model_fn: Callable that maps a tensor to a tensor.  Wrap the
                model as needed before passing (e.g.
                ``lambda x: model(x)["reconstruction"]``).

        Returns:
            Output tensor, shape determined by *model_fn*.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, AbstractInferer) and self.name == other.name
