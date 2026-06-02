"""Representation and baseline models."""

from typing import Any

from src.models.baselines import CharacterCNNBaseline, TfidfIsolationForest, TfidfOneClassSVM

__all__ = [
    "CharacterCNNBaseline",
    "ContrastiveDistilBertEncoder",
    "TfidfIsolationForest",
    "TfidfOneClassSVM",
]


def __getattr__(name: str) -> Any:
    """Load the transformer stack only when the encoder is requested."""
    if name == "ContrastiveDistilBertEncoder":
        from src.models.encoder import ContrastiveDistilBertEncoder

        return ContrastiveDistilBertEncoder
    raise AttributeError(name)
