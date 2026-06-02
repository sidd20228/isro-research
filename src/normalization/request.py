from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

UUID_PATTERN = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]*\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
BEARER_TOKEN_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}")
HEX_TOKEN_PATTERN = re.compile(r"\b[0-9a-f]{24,}\b", re.I)
OPAQUE_TOKEN_PATTERN = re.compile(r"\b[A-Za-z0-9_-]{24,}\b")
NUMERIC_SEGMENT_PATTERN = re.compile(r"(?<=/)\d+(?=/|$)")
WHITESPACE_PATTERN = re.compile(r"\s+")
HEX_ESCAPE_PATTERN = re.compile(r"\\x([0-9a-f]{2})", re.I)
UNICODE_ESCAPE_PATTERN = re.compile(r"\\u([0-9a-f]{4})", re.I)
NUMERIC_ID_KEYS = frozenset({"id", "user_id", "userid", "product_id", "productid", "basket_id", "basketid", "order_id", "orderid"})


@dataclass(frozen=True)
class NormalizationConfig:
    """Control stable request canonicalization."""

    lowercase_paths: bool = True
    replace_numeric_ids: bool = True
    replace_uuids: bool = True
    replace_tokens: bool = True
    replace_emails: bool = True
    canonicalize_query: bool = True
    url_decode_iterations: int = 2
    decode_escaped_characters: bool = True

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "NormalizationConfig":
        return cls(**(values or {}))


def _decode_and_normalize(value: str, config: NormalizationConfig) -> str:
    """Decode bounded URL encoding and normalize Unicode without looping forever."""
    decoded = value
    for _ in range(max(config.url_decode_iterations, 0)):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    if config.decode_escaped_characters:
        decoded = HEX_ESCAPE_PATTERN.sub(lambda match: chr(int(match.group(1), 16)), decoded)
        decoded = UNICODE_ESCAPE_PATTERN.sub(_decode_unicode_escape, decoded)
    return unicodedata.normalize("NFKC", decoded)


def _decode_unicode_escape(match: re.Match[str]) -> str:
    value = int(match.group(1), 16)
    return match.group(0) if 0xD800 <= value <= 0xDFFF else chr(value)


def _replace_sensitive_values(value: str, config: NormalizationConfig, *, decode: bool = True) -> str:
    if decode:
        value = _decode_and_normalize(value, config)
    if config.replace_uuids:
        value = UUID_PATTERN.sub("<UUID>", value)
    if config.replace_emails:
        value = EMAIL_PATTERN.sub("<EMAIL>", value)
    if config.replace_tokens:
        value = JWT_PATTERN.sub("<TOKEN>", value)
        value = BEARER_TOKEN_PATTERN.sub("Bearer <TOKEN>", value)
        value = HEX_TOKEN_PATTERN.sub("<TOKEN>", value)
        value = OPAQUE_TOKEN_PATTERN.sub("<TOKEN>", value)
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def _normalize_query(query: str, config: NormalizationConfig) -> str:
    """Canonicalize query fields without letting decoded delimiters alter boundaries."""
    normalized_pairs: list[tuple[str, str, bool]] = []
    for raw_parameter in query.split("&"):
        if not raw_parameter:
            continue
        raw_key, separator, raw_value = raw_parameter.partition("=")
        key = _replace_sensitive_values(raw_key, config)
        value = _replace_sensitive_values(raw_value, config)
        if config.replace_numeric_ids and key.lower() in NUMERIC_ID_KEYS and value.isdigit():
            value = "<ID>"
        normalized_pairs.append((key, value, bool(separator)))
    if config.canonicalize_query:
        normalized_pairs.sort(key=lambda pair: (pair[0], pair[1]))
    return "&".join(f"{key}={value}" if had_separator else key for key, value, had_separator in normalized_pairs)


def normalize_path(path: str, config: NormalizationConfig) -> str:
    """Normalize a URL path and its query string deterministically."""
    raw_path = path or "/"
    if "://" not in raw_path and not raw_path.startswith("/"):
        raw_path = f"/{raw_path}"
    parsed = urlsplit(raw_path)
    normalized_path = _decode_and_normalize(parsed.path, config)
    normalized_path = normalized_path.lower() if config.lowercase_paths else normalized_path
    normalized_path = _replace_sensitive_values(normalized_path, config, decode=False)
    if config.replace_numeric_ids:
        normalized_path = NUMERIC_SEGMENT_PATTERN.sub("<ID>", normalized_path)
    query = _normalize_query(parsed.query, config)
    return urlunsplit(("", "", normalized_path or "/", query, ""))


def _coerce_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def normalize_request(request: dict[str, Any], config: NormalizationConfig | None = None) -> dict[str, Any]:
    """Return a canonical request without mutating the input."""
    active_config = config or NormalizationConfig()
    return {
        **request,
        "method": str(request.get("method", "GET")).upper(),
        "path": normalize_path(str(request.get("path", "/")), active_config),
        "body": _replace_sensitive_values(str(request.get("body", "")), active_config),
        "status_code": _coerce_int(request.get("status_code", 0)),
        "response_size": _coerce_int(request.get("response_size", 0)),
    }
