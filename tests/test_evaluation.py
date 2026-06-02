import numpy as np

from src.evaluation.metrics import evaluate_scores


def test_metrics_for_perfect_separation() -> None:
    metrics = evaluate_scores(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
