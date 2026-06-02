import json
from pathlib import Path

import pandas as pd
import pytest

from src.datasets.representations import deterministic_sample, load_jsonl_representations


def test_load_jsonl_representations_reads_materialized_text(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"source_app": "dvwa", "label": "benign", "text": "METHOD:GET\nPATH:/"}),
                json.dumps({"source_app": "webgoat", "label": "attack", "text": "METHOD:POST\nPATH:/login"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    frame = load_jsonl_representations(path)
    assert frame.to_dict(orient="records") == [
        {"source_app": "dvwa", "label": "benign", "text": "METHOD:GET\nPATH:/"},
        {"source_app": "webgoat", "label": "attack", "text": "METHOD:POST\nPATH:/login"},
    ]


def test_load_jsonl_representations_reports_invalid_line(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"text":"valid"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"invalid.jsonl:2"):
        load_jsonl_representations(path)


def test_deterministic_sample_is_reproducible() -> None:
    frame = pd.DataFrame({"value": range(100)})
    first = deterministic_sample(frame, 10, random_state=42)
    second = deterministic_sample(frame, 10, random_state=42)
    assert first.equals(second)
    assert len(first) == 10
