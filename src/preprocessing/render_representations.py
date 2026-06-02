from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.preprocessing.representation import RequestRenderer
from src.preprocessing.representation_pipeline import RepresentationPipeline
from src.utils.config import configure_logging, load_config


def _profile_names(config: dict[str, Any], requested: str) -> list[str]:
    profiles = config["representation"]["profiles"]
    if requested == "all":
        return list(profiles)
    if requested not in profiles:
        raise ValueError(f"Unknown representation profile: {requested}")
    return [requested]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render normalized HTTP requests into field-aware JSONL text.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--profile", default="all", help="Named profile or 'all'.")
    parser.add_argument("--chunk-size", type=int, default=25_000)
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    profiles = config["representation"]["profiles"]
    for profile_name in _profile_names(config, args.profile):
        renderer = RequestRenderer.from_config(config, profile_name)
        pipeline = RepresentationPipeline(renderer, chunk_size=args.chunk_size)
        pipeline.render_directory(
            Path(config["paths"]["normalized_dir"]),
            Path(config["paths"]["representations_dir"]),
            profile_name,
            str(profiles[profile_name].get("description", "")),
        )


if __name__ == "__main__":
    main()
