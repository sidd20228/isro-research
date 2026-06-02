from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_jsonl_representations(path: str | Path) -> pd.DataFrame:
    """Load materialized field-aware JSONL records into a dataframe."""
    source = Path(path)
    records: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL record at {source}:{line_number}") from error
            records.append(
                {
                    "source_app": str(record.get("source_app", "")),
                    "label": str(record.get("label", "")),
                    "text": str(record.get("text", "")),
                }
            )
    frame = pd.DataFrame(records, columns=["source_app", "label", "text"])
    LOGGER.info("Loaded %d representations from %s", len(frame), source)
    return frame


def deterministic_sample(frame: pd.DataFrame, maximum_rows: int | None, random_state: int) -> pd.DataFrame:
    """Cap expensive model fitting reproducibly while preserving all rows when feasible."""
    if maximum_rows is None or maximum_rows <= 0 or len(frame) <= maximum_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=maximum_rows, random_state=random_state).reset_index(drop=True)


def load_profile_splits(representations_dir: str | Path, profile: str) -> dict[str, pd.DataFrame]:
    """Load train-benign, test-benign, and test-attack representation splits."""
    directory = Path(representations_dir) / profile
    return {
        name: load_jsonl_representations(directory / f"{name}.jsonl")
        for name in ("train_benign", "test_benign", "test_attack")
    }
