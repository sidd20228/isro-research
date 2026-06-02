from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocessing.representation import RequestRenderer

LOGGER = logging.getLogger(__name__)
SPLIT_FILENAMES = ("train_benign.csv", "test_benign.csv", "test_attack.csv")


@dataclass
class RepresentationStatistics:
    """Auditable text-representation summary for one split."""

    rows: int = 0
    empty_bodies: int = 0
    total_characters: int = 0
    minimum_characters: int | None = None
    maximum_characters: int = 0
    sha256: str = ""

    def update(self, text: str, body: str) -> None:
        length = len(text)
        self.rows += 1
        self.empty_bodies += not body
        self.total_characters += length
        self.minimum_characters = length if self.minimum_characters is None else min(self.minimum_characters, length)
        self.maximum_characters = max(self.maximum_characters, length)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "average_characters": self.total_characters / self.rows if self.rows else 0.0,
        }


class RepresentationPipeline:
    """Stream normalized CSV splits into deterministic field-aware JSONL."""

    def __init__(self, renderer: RequestRenderer, chunk_size: int = 25_000) -> None:
        self.renderer = renderer
        self.chunk_size = chunk_size

    def render_csv(self, input_path: str | Path, output_path: str | Path) -> RepresentationStatistics:
        """Render one normalized CSV split into compact JSONL records."""
        source = Path(input_path)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        statistics = RepresentationStatistics()
        digest = hashlib.sha256()
        with output.open("w", encoding="utf-8") as handle:
            for chunk in pd.read_csv(source, chunksize=self.chunk_size, keep_default_na=False):
                for record in chunk.to_dict(orient="records"):
                    text = self.renderer.render(record, normalize=False)
                    output_record = {
                        "source_app": str(record.get("source_app", "")),
                        "label": str(record.get("label", "")),
                        "text": text,
                    }
                    line = json.dumps(output_record, ensure_ascii=True, separators=(",", ":")) + "\n"
                    handle.write(line)
                    digest.update(line.encode("utf-8"))
                    statistics.update(text, str(record.get("body", "")))
        statistics.sha256 = digest.hexdigest()
        LOGGER.info("Rendered %s -> %s (%d rows)", source, output, statistics.rows)
        return statistics

    def render_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        profile_name: str,
        description: str,
    ) -> dict[str, Any]:
        """Render all normalized splits for one representation profile."""
        source_dir = Path(input_dir)
        destination_dir = Path(output_dir) / profile_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        splits: dict[str, Any] = {}
        for filename in SPLIT_FILENAMES:
            source = source_dir / filename
            if not source.exists():
                raise FileNotFoundError(f"Missing normalized split: {source}")
            output_filename = Path(filename).with_suffix(".jsonl").name
            splits[output_filename] = self.render_csv(source, destination_dir / output_filename).to_dict()
        metadata = {
            "profile": profile_name,
            "description": description,
            "template": self.renderer.template,
            "splits": splits,
        }
        with (destination_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        LOGGER.info("Saved representation metadata to %s", destination_dir / "metadata.json")
        return metadata
