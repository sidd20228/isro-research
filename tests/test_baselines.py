from pathlib import Path

import numpy as np

from src.models.baselines import TfidfIsolationForest, TfidfOneClassSVM

BENIGN = [
    "METHOD:GET\nPATH:/health\nBODY:",
    "METHOD:GET\nPATH:/products/<ID>\nBODY:",
    "METHOD:POST\nPATH:/login\nBODY:username=alice",
    "METHOD:GET\nPATH:/profile/<ID>\nBODY:",
    "METHOD:POST\nPATH:/search\nBODY:q=shoes",
]
ATTACK = "METHOD:GET\nPATH:/../../etc/passwd\nBODY:' UNION SELECT password FROM users--"


def test_tfidf_baselines_calibrate_threshold_and_round_trip(tmp_path: Path) -> None:
    for model in (TfidfIsolationForest(contamination=0.2, random_state=42), TfidfOneClassSVM(nu=0.2)):
        model.fit(BENIGN, threshold_quantile=0.8)
        assert np.isfinite(model.threshold)
        scores = model.score(BENIGN + [ATTACK])
        assert scores.shape == (6,)
        path = tmp_path / f"{type(model).__name__}.joblib"
        model.save(path)
        loaded = type(model).load(path)
        assert np.allclose(loaded.score(BENIGN + [ATTACK]), scores)
        assert loaded.threshold == model.threshold
