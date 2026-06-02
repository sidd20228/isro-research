from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def save_run_record(output_dir: str | Path, name: str, config: dict[str, Any], metrics: dict[str, Any]) -> Path:
    """Persist a timestamped, reproducible local experiment record."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{timestamp}_{name}.json"
    path.write_text(json.dumps({"name": name, "timestamp": timestamp, "config": config, "metrics": metrics}, indent=2), encoding="utf-8")
    return path
