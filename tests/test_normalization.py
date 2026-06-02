from pathlib import Path

import pandas as pd

from src.normalization import NormalizationConfig, normalize_path, normalize_request
from src.normalization.pipeline import DatasetNormalizer


def test_normalize_request_replaces_variable_values_and_sorts_query() -> None:
    request = {
        "method": "get",
        "path": "/User/12345/Profile?z=2&email=alice%40example.com&a=1",
        "body": " token    abcdefghijklmnopqrstuvwxyz ",
        "status_code": "200",
    }
    normalized = normalize_request(request)
    assert normalized["method"] == "GET"
    assert normalized["path"] == "/user/<ID>/profile?a=1&email=<EMAIL>&z=2"
    assert normalized["body"] == "token <TOKEN>"
    assert normalized["status_code"] == 200


def test_normalize_request_can_preserve_path_case() -> None:
    normalized = normalize_request({"path": "/User/42"}, NormalizationConfig(lowercase_paths=False))
    assert normalized["path"] == "/User/<ID>"


def test_query_decoding_does_not_turn_encoded_delimiter_into_new_parameter() -> None:
    normalized = normalize_path("/search?payload=a%26admin%3Dtrue&z=2", NormalizationConfig())
    assert normalized == "/search?payload=a&admin=true&z=2"


def test_normalization_exposes_double_encoded_attack_syntax() -> None:
    normalized = normalize_request({"path": "/download?file=%252e%252e%252fetc%252fpasswd", "body": "q=%2527+OR+1%253D1"})
    assert normalized["path"] == "/download?file=../etc/passwd"
    assert normalized["body"] == "q='+OR+1=1"


def test_normalization_replaces_uuid_jwt_email_and_identifier_query_value() -> None:
    normalized = normalize_request(
        {
            "path": "/users/550e8400-e29b-41d4-a716-446655440000?user_id=123&email=alice@example.com",
            "body": "authorization=Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.signaturepart",
        }
    )
    assert normalized["path"] == "/users/<UUID>?email=<EMAIL>&user_id=<ID>"
    assert normalized["body"] == "authorization=Bearer <TOKEN>"


def test_normalization_applies_nfkc_and_handles_invalid_numeric_metadata() -> None:
    normalized = normalize_request({"path": "/Ｆｏｏ/１２", "status_code": "unknown", "response_size": None})
    assert normalized["path"] == "/foo/<ID>"
    assert normalized["status_code"] == 0
    assert normalized["response_size"] == 0


def test_normalization_decodes_hex_and_unicode_escape_sequences() -> None:
    normalized = normalize_request({"body": r"{\x22payload\x22:\u0020\uFF1Cscript\uFF1E}"})
    assert normalized["body"] == '{"payload": <script>}'


def test_dataset_normalizer_streams_output_and_statistics(tmp_path: Path) -> None:
    source = tmp_path / "input.csv"
    output = tmp_path / "normalized.csv"
    pd.DataFrame(
        [
            {"method": "GET", "path": "/Users/12", "body": "", "status_code": 200, "response_size": 1},
            {"method": "POST", "path": "/login", "body": " email=alice%40example.com ", "status_code": 200, "response_size": 2},
        ]
    ).to_csv(source, index=False)
    statistics = DatasetNormalizer(chunk_size=1).normalize_csv(source, output)
    normalized = pd.read_csv(output, keep_default_na=False)
    assert normalized["path"].tolist() == ["/users/<ID>", "/login"]
    assert normalized["body"].tolist() == ["", "email=<EMAIL>"]
    assert statistics.to_dict() == {
        "rows": 2,
        "changed_paths": 1,
        "changed_bodies": 1,
        "placeholder_counts": {"<ID>": 1, "<UUID>": 0, "<TOKEN>": 0, "<EMAIL>": 1},
    }
