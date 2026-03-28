"""MISFIT embedding training objectives."""
from misfit.embedding.objectives.classification import ClassificationObjective
from misfit.embedding.objectives.contrastive import ContrastiveObjective
from misfit.embedding.objectives.objective_registry import (
    get_objective,
    list_objectives,
    register_objective,
)

__all__ = [
    "ClassificationObjective",
    "ContrastiveObjective",
    "get_objective",
    "list_objectives",
    "register_objective",
]
