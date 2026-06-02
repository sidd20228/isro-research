from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    LOGGER.info("Loaded configuration from %s", config_path)
    return config


def set_seed(seed: int) -> None:
    """Seed supported random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        LOGGER.debug("PyTorch is not installed; skipping torch seed")
    LOGGER.info("Set reproducible seed to %d", seed)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application logging once."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
