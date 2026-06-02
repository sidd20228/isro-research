from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

LOGGER = logging.getLogger(__name__)


def _save(path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    LOGGER.info("Saved figure to %s", output)


def plot_roc(labels: np.ndarray, scores: np.ndarray, path: str | Path) -> None:
    RocCurveDisplay.from_predictions(labels, scores)
    _save(path)


def plot_precision_recall(labels: np.ndarray, scores: np.ndarray, path: str | Path) -> None:
    PrecisionRecallDisplay.from_predictions(labels, scores)
    _save(path)


def plot_distance_distribution(labels: np.ndarray, scores: np.ndarray, path: str | Path) -> None:
    plt.figure()
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, label="benign")
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, label="attack")
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.legend()
    _save(path)


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, path: str | Path, random_state: int = 42) -> None:
    projection = TSNE(n_components=2, random_state=random_state).fit_transform(embeddings)
    _scatter(projection, labels, path, "t-SNE security embeddings")


def plot_umap(embeddings: np.ndarray, labels: np.ndarray, path: str | Path, random_state: int = 42) -> None:
    import umap

    projection = umap.UMAP(random_state=random_state).fit_transform(embeddings)
    _scatter(projection, labels, path, "UMAP security embeddings")


def _scatter(projection: np.ndarray, labels: np.ndarray, path: str | Path, title: str) -> None:
    plt.figure()
    for label, name in ((0, "benign"), (1, "attack")):
        subset = projection[labels == label]
        plt.scatter(subset[:, 0], subset[:, 1], s=8, alpha=0.6, label=name)
    plt.title(title)
    plt.legend()
    _save(path)
