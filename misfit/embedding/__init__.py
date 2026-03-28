"""MISFIT global image embedding module."""
# Trigger aggregator and objective registrations.
import misfit.embedding.aggregators  # noqa: F401
import misfit.embedding.objectives   # noqa: F401

from misfit.embedding.embedder import Embedder
from misfit.embedding.embed_trainer import EmbedTrainer

__all__ = ["Embedder", "EmbedTrainer"]
