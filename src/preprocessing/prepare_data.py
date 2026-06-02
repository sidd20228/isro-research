from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets import DatasetLoader
from src.utils.config import configure_logging, load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Unify raw HTTP request CSV files.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    loader = DatasetLoader.from_config(config)
    output_dir = Path(config["paths"]["processed_dir"])
    loader.split_and_save(output_dir, float(config["data"].get("benign_test_size", 0.2)))


if __name__ == "__main__":
    main()
