from __future__ import annotations

import logging
from pathlib import Path
from typing import Self

import joblib
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

LOGGER = logging.getLogger(__name__)


class _PersistedDetector:
    threshold: float

    def predict(self, samples: np.ndarray) -> np.ndarray:
        return (self.score(samples) >= self.threshold).astype(int)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return joblib.load(path)


class MahalanobisDetector(_PersistedDetector):
    """Shrinkage-covariance Mahalanobis distance detector."""

    def __init__(self, quantile: float = 0.99) -> None:
        self.quantile = quantile
        self.location = np.array([])
        self.precision = np.array([[]])
        self.threshold = float("inf")

    def fit(self, samples: np.ndarray) -> Self:
        estimator = LedoitWolf().fit(samples)
        self.location = estimator.location_
        self.precision = estimator.precision_
        self.threshold = float(np.quantile(self.score(samples), self.quantile))
        return self

    def score(self, samples: np.ndarray) -> np.ndarray:
        centered = samples - self.location
        return np.einsum("ij,jk,ik->i", centered, self.precision, centered)


class IsolationForestDetector(_PersistedDetector):
    """Isolation Forest over security embeddings."""

    def __init__(self, contamination: float = 0.01, random_state: int = 42) -> None:
        self.model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
        self.threshold = 0.0

    def fit(self, samples: np.ndarray) -> Self:
        self.model.fit(samples)
        return self

    def score(self, samples: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(samples)


class OneClassSVMDetector(_PersistedDetector):
    """One-class SVM over security embeddings."""

    def __init__(self, nu: float = 0.01) -> None:
        self.model = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        self.threshold = 0.0

    def fit(self, samples: np.ndarray) -> Self:
        self.model.fit(samples)
        return self

    def score(self, samples: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(samples).reshape(-1)


class NormalizingFlowDetector(_PersistedDetector):
    """Lightweight affine normalizing flow for embedding likelihood."""

    def __init__(self, quantile: float = 0.99, epochs: int = 100, learning_rate: float = 1e-3, device: str = "cpu") -> None:
        self.quantile = quantile
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.flow: object | None = None
        self.threshold = float("inf")

    def fit(self, samples: np.ndarray) -> Self:
        import torch

        from src.detectors.flow import AffineFlow

        tensor = torch.as_tensor(samples, dtype=torch.float32, device=self.device)
        self.flow = AffineFlow(samples.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.learning_rate)
        for _ in range(self.epochs):
            loss = -self.flow.log_prob(tensor).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.threshold = float(np.quantile(self.score(samples), self.quantile))
        return self

    def score(self, samples: np.ndarray) -> np.ndarray:
        import torch

        if self.flow is None:
            raise RuntimeError("Normalizing flow is not fitted")
        with torch.no_grad():
            tensor = torch.as_tensor(samples, dtype=torch.float32, device=self.device)
            return (-self.flow.log_prob(tensor)).cpu().numpy()

    def save(self, path: str | Path) -> None:
        import torch

        if self.flow is None:
            raise RuntimeError("Normalizing flow is not fitted")
        torch.save({"state_dict": self.flow.state_dict(), "dimension": self.flow.dimension, "config": self.__dict__}, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        import torch

        from src.detectors.flow import AffineFlow

        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = {key: value for key, value in payload["config"].items() if key != "flow"}
        instance = cls()
        instance.__dict__.update(config)
        instance.flow = AffineFlow(payload["dimension"])
        instance.flow.load_state_dict(payload["state_dict"])
        return instance
