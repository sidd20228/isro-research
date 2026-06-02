import json
from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import OFFLINE_ABLATION_TEMPLATE, REQUEST_TIME_TEMPLATE, RequestRenderer
from src.preprocessing.representation_pipeline import RepresentationPipeline


def test_renderer_is_deterministic_and_field_aware() -> None:
    renderer = RequestRenderer()
    rendered = renderer.render({"method": "post", "path": "/Users/12", "body": "a=1", "status_code": 200})
    assert rendered == "METHOD:POST\nPATH:/users/<ID>\nBODY:a=1"


def test_offline_ablation_template_includes_response_fields() -> None:
    renderer = RequestRenderer(OFFLINE_ABLATION_TEMPLATE)
    rendered = renderer.render({"method": "GET", "path": "/", "body": "", "status_code": 404, "response_size": 123})
    assert rendered == "METHOD:GET\nPATH:/\nBODY:\nSTATUS:404\nRESPONSE_SIZE:123"


def test_request_time_template_excludes_response_fields() -> None:
    assert "status_code" not in REQUEST_TIME_TEMPLATE
    assert "response_size" not in REQUEST_TIME_TEMPLATE


def test_renderer_rejects_unknown_template_field() -> None:
    with pytest.raises(ValueError, match="Unsupported representation fields"):
        RequestRenderer("METHOD:{method}\nIP:{source_ip}")


def test_renderer_builds_named_profile_from_config() -> None:
    config = {
        "representation": {
            "default_profile": "request_time",
            "profiles": {
                "request_time": {"template": REQUEST_TIME_TEMPLATE},
                "offline_ablation": {"template": OFFLINE_ABLATION_TEMPLATE},
            },
        }
    }
    assert RequestRenderer.from_config(config).template == REQUEST_TIME_TEMPLATE
    assert RequestRenderer.from_config(config, "offline_ablation").template == OFFLINE_ABLATION_TEMPLATE


def test_representation_pipeline_streams_jsonl_with_stable_hash(tmp_path: Path) -> None:
    source = tmp_path / "normalized.csv"
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    pd.DataFrame(
        [
            {"source_app": "demo", "label": "benign", "method": "GET", "path": "/health", "body": "", "status_code": 200, "response_size": 1},
            {"source_app": "demo", "label": "attack", "method": "POST", "path": "/login", "body": "q=' or 1=1", "status_code": 500, "response_size": 2},
        ]
    ).to_csv(source, index=False)
    pipeline = RepresentationPipeline(RequestRenderer(), chunk_size=1)
    first = pipeline.render_csv(source, first_output)
    second = pipeline.render_csv(source, second_output)
    records = [json.loads(line) for line in first_output.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {"source_app": "demo", "label": "benign", "text": "METHOD:GET\nPATH:/health\nBODY:"},
        {"source_app": "demo", "label": "attack", "text": "METHOD:POST\nPATH:/login\nBODY:q=' or 1=1"},
    ]
    assert first.sha256 == second.sha256
    assert first.to_dict() == {
        "rows": 2,
        "empty_bodies": 1,
        "total_characters": 68,
        "minimum_characters": 29,
        "maximum_characters": 39,
        "sha256": first.sha256,
        "average_characters": 34.0,
    }
