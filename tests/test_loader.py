from pathlib import Path

import pandas as pd

from src.datasets import DatasetLoader, DatasetSource


def test_loader_standardizes_aliases(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    pd.DataFrame(
        [{"request.method": "post", "request.path": "/login", "request_body": None, "status": "201", "body_bytes": "12"}]
    ).to_csv(source, index=False)
    frame = DatasetLoader([DatasetSource(str(source), "demo", "benign")]).load_all()
    assert frame.to_dict(orient="records") == [
        {"source_app": "demo", "label": "benign", "method": "POST", "path": "/login", "body": "", "status_code": 201, "response_size": 12}
    ]


def test_loader_saves_expected_splits(tmp_path: Path) -> None:
    source = tmp_path / "raw.csv"
    pd.DataFrame([{"method": "GET", "path": f"/{index}"} for index in range(10)]).to_csv(source, index=False)
    metadata = DatasetLoader([DatasetSource(str(source), "demo", "benign")]).split_and_save(tmp_path / "processed")
    assert metadata["total_requests"] == 10
    assert (tmp_path / "processed" / "train_benign.csv").exists()
    assert (tmp_path / "processed" / "metadata.json").exists()
