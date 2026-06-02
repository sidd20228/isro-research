"""Open-set detectors."""

from src.detectors.open_set import IsolationForestDetector, MahalanobisDetector, NormalizingFlowDetector, OneClassSVMDetector

__all__ = ["IsolationForestDetector", "MahalanobisDetector", "NormalizingFlowDetector", "OneClassSVMDetector"]


def create_detector(name: str, **kwargs: object) -> object:
    """Construct a detector by configuration name."""
    detectors = {
        "mahalanobis": MahalanobisDetector,
        "isolation_forest": IsolationForestDetector,
        "one_class_svm": OneClassSVMDetector,
        "normalizing_flow": NormalizingFlowDetector,
    }
    if name not in detectors:
        raise ValueError(f"Unknown detector: {name}")
    return detectors[name](**kwargs)
