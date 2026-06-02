from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def false_positive_rate_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """Return the lowest false-positive rate that reaches target TPR."""
    fpr, tpr, _ = roc_curve(labels, scores)
    matches = np.flatnonzero(tpr >= target_tpr)
    return float(fpr[matches[0]]) if len(matches) else 1.0


def evaluate_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute binary open-set detection metrics."""
    predictions = (scores >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "fpr_at_95_tpr": false_positive_rate_at_tpr(labels, scores),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "threshold": float(threshold),
        "sample_count": int(len(labels)),
    }
