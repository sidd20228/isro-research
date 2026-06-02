import numpy as np

from src.detectors.open_set import IsolationForestDetector, MahalanobisDetector, OneClassSVMDetector


def test_detectors_assign_higher_scores_to_far_outlier() -> None:
    random = np.random.default_rng(42)
    benign = random.normal(0, 0.1, size=(100, 4))
    outlier = np.array([[10.0, 10.0, 10.0, 10.0]])
    for detector in (MahalanobisDetector(), IsolationForestDetector(), OneClassSVMDetector()):
        detector.fit(benign)
        assert detector.score(outlier)[0] > np.median(detector.score(benign))
        assert detector.predict(outlier)[0] == 1
