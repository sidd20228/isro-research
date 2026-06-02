from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSource:
    """Describe one raw CSV source."""

    path: str
    source_app: str
    label: str


class DatasetLoader:
    """Load heterogeneous HTTP CSV datasets into a canonical schema."""

    STANDARD_COLUMNS: ClassVar[list[str]] = [
        "source_app",
        "label",
        "method",
        "path",
        "body",
        "status_code",
        "response_size",
    ]
    ALIASES: ClassVar[dict[str, tuple[str, ...]]] = {
        "method": ("method", "request.method", "http_method", "verb"),
        "path": ("path", "request.path", "url", "uri", "request_uri"),
        "body": ("body", "request_body", "payload", "post_data"),
        "status_code": ("status_code", "status", "response.status"),
        "response_size": (
            "response_size",
            "body_bytes_sent",
            "body_bytes",
            "bytes_sent",
        ),
    }

    def __init__(self, sources: list[DatasetSource], random_state: int = 42) -> None:
        self.sources = sources
        self.random_state = random_state

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DatasetLoader":
        """Build a loader from the data section of a project configuration."""
        sources = [DatasetSource(**item) for item in config["data"]["sources"]]
        return cls(sources=sources, random_state=int(config.get("seed", 42)))

    @classmethod
    def _find_column(cls, columns: pd.Index[str], standard_name: str) -> str | None:
        lowered = {str(column).lower(): str(column) for column in columns}
        for alias in cls.ALIASES[standard_name]:
            if alias.lower() in lowered:
                return lowered[alias.lower()]
        return None

    def load_source(self, source: DatasetSource) -> pd.DataFrame:
        """Load and standardize one source file."""
        path = Path(source.path)
        LOGGER.info("Loading %s (%s, %s)", path, source.source_app, source.label)
        raw = pd.read_csv(path, low_memory=False)
        output = pd.DataFrame(index=raw.index)
        output["source_app"] = source.source_app
        output["label"] = source.label
        for target in self.STANDARD_COLUMNS[2:]:
            column = self._find_column(raw.columns, target)
            if column is None:
                LOGGER.warning("%s has no column for %s; using defaults", path, target)
                output[target] = "" if target in {"method", "path", "body"} else 0
            else:
                output[target] = raw[column]
        output["method"] = output["method"].fillna("GET").astype(str).str.upper()
        output["path"] = output["path"].fillna("/").astype(str)
        output["body"] = output["body"].fillna("").replace({"N/A": ""}).astype(str)
        output["status_code"] = pd.to_numeric(output["status_code"], errors="coerce").fillna(0).astype(int)
        output["response_size"] = pd.to_numeric(output["response_size"], errors="coerce").fillna(0).astype(int)
        return output[self.STANDARD_COLUMNS]

    def load_all(self) -> pd.DataFrame:
        """Load and concatenate every configured source."""
        unified = pd.concat([self.load_source(source) for source in self.sources], ignore_index=True)
        LOGGER.info("Unified %d HTTP requests", len(unified))
        return unified

    def split_and_save(self, output_dir: str | Path, benign_test_size: float = 0.2) -> dict[str, Any]:
        """Persist benign train/test and attack test splits plus metadata."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        unified = self.load_all()
        benign = unified[unified["label"] == "benign"].reset_index(drop=True)
        attacks = unified[unified["label"] != "benign"].reset_index(drop=True)
        stratify = benign["source_app"] if benign["source_app"].value_counts().min() >= 2 else None
        train_benign, test_benign = train_test_split(
            benign,
            test_size=benign_test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        outputs = {
            "train_benign.csv": train_benign,
            "test_benign.csv": test_benign,
            "test_attack.csv": attacks,
        }
        for filename, frame in outputs.items():
            frame.to_csv(output_path / filename, index=False)
            LOGGER.info("Saved %s with %d rows", output_path / filename, len(frame))
        metadata = self._metadata(unified, outputs)
        with (output_path / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return metadata

    def _metadata(self, unified: pd.DataFrame, outputs: dict[str, pd.DataFrame]) -> dict[str, Any]:
        return {
            "total_requests": int(len(unified)),
            "sources": [asdict(source) for source in self.sources],
            "class_distribution": unified["label"].value_counts().sort_index().to_dict(),
            "application_distribution": unified["source_app"].value_counts().sort_index().to_dict(),
            "splits": {filename: int(len(frame)) for filename, frame in outputs.items()},
        }
