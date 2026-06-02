from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import OneClassSVM

LOGGER = logging.getLogger(__name__)


class _TfidfBaseline:
    """Shared persistence and prediction behavior for TF-IDF baselines."""

    vectorizer: TfidfVectorizer
    detector: Any

    def fit(self, samples: list[str], threshold_quantile: float = 0.99) -> Self:
        features = self.vectorizer.fit_transform(samples)
        self.detector.fit(features)
        self.threshold = float(np.quantile(self._score_features(features), threshold_quantile))
        LOGGER.info("Fitted %s on %d requests", type(self).__name__, len(samples))
        return self

    def score(self, samples: list[str]) -> np.ndarray:
        features = self.vectorizer.transform(samples)
        return self._score_features(features)

    def _score_features(self, features: Any) -> np.ndarray:
        return -np.asarray(self.detector.decision_function(features)).reshape(-1)

    def predict(self, samples: list[str]) -> np.ndarray:
        return (self.score(samples) >= self.threshold).astype(int)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return joblib.load(path)


class TfidfIsolationForest(_TfidfBaseline):
    """Character n-gram TF-IDF with Isolation Forest."""

    def __init__(
        self,
        contamination: float = 0.01,
        max_features: int = 50_000,
        random_state: int = 42,
    ) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=max_features)
        self.detector = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
        self.threshold = 0.0


class TfidfOneClassSVM(_TfidfBaseline):
    """Character n-gram TF-IDF with one-class SVM."""

    def __init__(self, nu: float = 0.01, max_features: int = 50_000) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=max_features)
        self.detector = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        self.threshold = 0.0


@dataclass
class CharacterCNNConfig:
    """Character CNN hyperparameters."""

    max_length: int = 512
    embedding_dim: int = 32
    channels: int = 64
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 1e-3
    quantile: float = 0.99
    device: str = "cpu"


class CharacterCNNBaseline:
    """Autoencoding character CNN trained only on benign requests."""

    ALPHABET = "".join(chr(index) for index in range(32, 127))

    def __init__(self, config: CharacterCNNConfig | None = None) -> None:
        import torch

        from src.models.character_cnn import CharacterCNNAutoencoder

        self.config = config or CharacterCNNConfig()
        self.vocabulary = {character: index + 1 for index, character in enumerate(self.ALPHABET)}
        self.unknown_index = len(self.vocabulary) + 1
        self.model = CharacterCNNAutoencoder(len(self.vocabulary) + 2, self.config.embedding_dim, self.config.channels)
        self.model.to(torch.device(self.config.device))
        self.threshold = float("inf")

    def _encode(self, samples: list[str]) -> Any:
        import torch

        encoded = np.zeros((len(samples), self.config.max_length), dtype=np.int64)
        for row, sample in enumerate(samples):
            for column, character in enumerate(sample[: self.config.max_length]):
                encoded[row, column] = self.vocabulary.get(character, self.unknown_index)
        return torch.from_numpy(encoded)

    def fit(self, samples: list[str], threshold_quantile: float | None = None) -> Self:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self.config.device)
        loader = DataLoader(TensorDataset(self._encode(samples)), batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.model.train()
        for epoch in range(self.config.epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                logits = self.model(batch)
                mask = batch.ne(0)
                per_character = torch.nn.functional.cross_entropy(logits.transpose(1, 2), batch, reduction="none")
                loss = (per_character * mask).sum() / mask.sum().clamp(min=1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            LOGGER.info("Finished character CNN epoch %d/%d", epoch + 1, self.config.epochs)
        quantile = self.config.quantile if threshold_quantile is None else threshold_quantile
        self.threshold = float(np.quantile(self.score(samples), quantile))
        return self

    def score(self, samples: list[str]) -> np.ndarray:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device(self.config.device)
        loader = DataLoader(TensorDataset(self._encode(samples)), batch_size=self.config.batch_size)
        losses: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(device)
                logits = self.model(batch)
                mask = batch.ne(0)
                per_character = torch.nn.functional.cross_entropy(logits.transpose(1, 2), batch, reduction="none")
                losses.append(((per_character * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)).cpu().numpy())
        return np.concatenate(losses) if losses else np.array([], dtype=float)

    def predict(self, samples: list[str]) -> np.ndarray:
        return (self.score(samples) >= self.threshold).astype(int)

    def save(self, path: str | Path) -> None:
        import torch

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "config": self.config, "threshold": self.threshold}, output)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(payload["config"])
        instance.model.load_state_dict(payload["state_dict"])
        instance.threshold = float(payload["threshold"])
        return instance
