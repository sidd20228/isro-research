from __future__ import annotations

import argparse
from pathlib import Path

from src.normalization.pipeline import DatasetNormalizer
from src.normalization.request import NormalizationConfig
from src.utils.config import configure_logging, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize unified HTTP request datasets.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--chunk-size", type=int, default=25_000)
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    normalizer = DatasetNormalizer(
        NormalizationConfig.from_dict(config.get("normalization")),
        chunk_size=args.chunk_size,
    )
    normalizer.normalize_directory(
        Path(config["paths"]["processed_dir"]),
        Path(config["paths"]["normalized_dir"]),
    )


if __name__ == "__main__":
    main()
