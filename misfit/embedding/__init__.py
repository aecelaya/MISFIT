"""MISFIT global image embedding module."""
# Trigger aggregator and objective registrations.
import misfit.embedding.aggregators  # noqa: F401
import misfit.embedding.objectives  # noqa: F401
from misfit.embedding.embed_trainer import EmbedTrainer
from misfit.embedding.embedder import Embedder

__all__ = ["Embedder", "EmbedTrainer"]
