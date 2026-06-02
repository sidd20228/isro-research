from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Ablation:
    """One controlled experiment variant."""

    normalize: bool = True
    contrastive: bool = True
    augmentation: bool = True
    embedding_dim: int = 128
    detector: str = "mahalanobis"


def default_ablations() -> list[Ablation]:
    """Generate the declared V1 ablation matrix."""
    controlled = [
        Ablation(),
        Ablation(normalize=False),
        Ablation(contrastive=False),
        Ablation(augmentation=False),
    ]
    grid = [
        Ablation(embedding_dim=dimension, detector=detector)
        for dimension, detector in product(
            (64, 128, 256),
            ("mahalanobis", "isolation_forest", "one_class_svm", "normalizing_flow"),
        )
    ]
    return controlled + grid
