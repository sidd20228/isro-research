from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.normalization.request import NormalizationConfig, normalize_request

LOGGER = logging.getLogger(__name__)
SPLIT_FILENAMES = ("train_benign.csv", "test_benign.csv", "test_attack.csv")
PLACEHOLDERS = ("<ID>", "<UUID>", "<TOKEN>", "<EMAIL>")


@dataclass
class NormalizationStatistics:
    """Aggregate auditable normalization effects for one dataset split."""

    rows: int = 0
    changed_paths: int = 0
    changed_bodies: int = 0
    placeholders: Counter[str] = field(default_factory=Counter)

    def update(self, original: dict[str, Any], normalized: dict[str, Any]) -> None:
        self.rows += 1
        self.changed_paths += original.get("path", "") != normalized["path"]
        self.changed_bodies += original.get("body", "") != normalized["body"]
        normalized_text = f"{normalized['path']} {normalized['body']}"
        self.placeholders.update({placeholder: normalized_text.count(placeholder) for placeholder in PLACEHOLDERS})

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "changed_paths": self.changed_paths,
            "changed_bodies": self.changed_bodies,
            "placeholder_counts": {placeholder: self.placeholders[placeholder] for placeholder in PLACEHOLDERS},
        }


class DatasetNormalizer:
    """Stream canonical CSV splits through request normalization."""

    def __init__(self, config: NormalizationConfig | None = None, chunk_size: int = 25_000) -> None:
        self.config = config or NormalizationConfig()
        self.chunk_size = chunk_size

    def normalize_csv(self, input_path: str | Path, output_path: str | Path) -> NormalizationStatistics:
        """Normalize one CSV file without loading the entire split into memory."""
        source = Path(input_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        statistics = NormalizationStatistics()
        wrote_header = False
        for chunk in pd.read_csv(source, chunksize=self.chunk_size, keep_default_na=False):
            records = chunk.to_dict(orient="records")
            normalized_records: list[dict[str, Any]] = []
            for record in records:
                normalized = normalize_request(record, self.config)
                statistics.update(record, normalized)
                normalized_records.append(normalized)
            pd.DataFrame(normalized_records, columns=chunk.columns).to_csv(
                output,
                mode="a" if wrote_header else "w",
                header=not wrote_header,
                index=False,
            )
            wrote_header = True
        LOGGER.info("Normalized %s -> %s (%d rows)", source, output, statistics.rows)
        return statistics

    def normalize_directory(self, input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
        """Normalize all standard unified splits and persist transformation metadata."""
        source_dir = Path(input_dir)
        destination_dir = Path(output_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        splits: dict[str, Any] = {}
        for filename in SPLIT_FILENAMES:
            input_path = source_dir / filename
            if not input_path.exists():
                raise FileNotFoundError(f"Missing processed split: {input_path}")
            splits[filename] = self.normalize_csv(input_path, destination_dir / filename).to_dict()
        metadata = {
            "normalization_config": asdict(self.config),
            "splits": splits,
        }
        with (destination_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        LOGGER.info("Saved normalization metadata to %s", destination_dir / "metadata.json")
        return metadata
