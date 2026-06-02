from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any

from src.normalization import NormalizationConfig, normalize_request

REQUEST_TIME_TEMPLATE = "METHOD:{method}\nPATH:{path}\nBODY:{body}"
OFFLINE_ABLATION_TEMPLATE = "METHOD:{method}\nPATH:{path}\nBODY:{body}\nSTATUS:{status_code}\nRESPONSE_SIZE:{response_size}"
ALLOWED_FIELDS = frozenset({"method", "path", "body", "status_code", "response_size"})


@dataclass(frozen=True)
class RequestRenderer:
    """Render field-aware HTTP text for downstream models."""

    template: str = REQUEST_TIME_TEMPLATE
    normalization: NormalizationConfig = NormalizationConfig()

    def __post_init__(self) -> None:
        fields = {field for _, field, _, _ in string.Formatter().parse(self.template) if field}
        unknown_fields = fields - ALLOWED_FIELDS
        if unknown_fields:
            raise ValueError(f"Unsupported representation fields: {sorted(unknown_fields)}")

    def render(self, request: dict[str, Any], normalize: bool = True) -> str:
        values = normalize_request(request, self.normalization) if normalize else request
        return self.template.format_map(_DefaultValues(values))

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        profile: str | None = None,
        normalization: NormalizationConfig | None = None,
    ) -> "RequestRenderer":
        """Construct one named representation profile from project configuration."""
        representation = config["representation"]
        profile_name = profile or representation["default_profile"]
        profiles = representation["profiles"]
        if profile_name not in profiles:
            raise ValueError(f"Unknown representation profile: {profile_name}")
        return cls(template=str(profiles[profile_name]["template"]), normalization=normalization or NormalizationConfig())


class _DefaultValues(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return ""
