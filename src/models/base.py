from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self

import numpy as np


class AnomalyModel(Protocol):
    """Common interface for anomaly-scoring models."""

    def fit(self, samples: list[str] | np.ndarray) -> Self: ...

    def score(self, samples: list[str] | np.ndarray) -> np.ndarray: ...

    def predict(self, samples: list[str] | np.ndarray) -> np.ndarray: ...

    def save(self, path: str | Path) -> None: ...

    @classmethod
    def load(cls, path: str | Path) -> Self: ...
