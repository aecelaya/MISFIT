"""Optimizer registry for MISFIT training."""
from collections.abc import Callable, Iterator

import torch
import torch.optim as optim

from misfit.training.optimizers.optimizer_constants import oc

OPTIMIZER_REGISTRY: dict[str, Callable] = {}


def register_optimizer(name: str) -> Callable:
    """Decorator to register an optimizer builder function."""
    def decorator(fn: Callable) -> Callable:
        OPTIMIZER_REGISTRY[name.lower()] = fn
        return fn
    return decorator


def get_optimizer(
    name: str,
    params: Iterator[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float,
    eps: float,
) -> optim.Optimizer:
    """Instantiate an optimizer from the registry.

    Args:
        name: Registered optimizer name (case-insensitive).
        params: Model parameters to optimise.
        learning_rate: Base learning rate.
        weight_decay: L2 regularisation coefficient.
        eps: Numerical stability epsilon. Use TrainerConstants.AMP_EPS
            (1e-4) when training with AMP, NO_AMP_EPS (1e-8) otherwise.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If the name is not registered.
    """
    key = name.lower()
    if key not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Optimizer '{name}' is not registered. "
            f"Available: {sorted(OPTIMIZER_REGISTRY.keys())}"
        )
    return OPTIMIZER_REGISTRY[key](params, learning_rate, weight_decay, eps)


def list_optimizers() -> list:
    """Return a sorted list of registered optimizer names."""
    return sorted(OPTIMIZER_REGISTRY.keys())


@register_optimizer("adam")
def _adam(params, lr, weight_decay, eps):
    return optim.Adam(params, lr=lr, weight_decay=weight_decay, eps=eps)


@register_optimizer("adamw")
def _adamw(params, lr, weight_decay, eps):
    return optim.AdamW(params, lr=lr, weight_decay=weight_decay, eps=eps)


@register_optimizer("sgd")
def _sgd(params, lr, weight_decay, eps):  # eps unused for SGD
    return optim.SGD(
        params,
        lr=lr,
        weight_decay=weight_decay,
        momentum=oc.SGD_MOMENTUM,
        nesterov=oc.SGD_NESTEROV,
    )
