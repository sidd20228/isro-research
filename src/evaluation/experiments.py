from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossApplicationExperiment:
    """Define one application-level generalization experiment."""

    name: str
    train_apps: tuple[str, ...]
    test_attack_apps: tuple[str, ...]


CROSS_APPLICATION_EXPERIMENTS = (
    CrossApplicationExperiment("experiment_a", ("dvwa", "juiceshop"), ("webgoat",)),
    CrossApplicationExperiment("experiment_b", ("dvwa", "webgoat"), ("juiceshop",)),
    CrossApplicationExperiment("experiment_c", ("dvwa", "juiceshop", "webgoat"), ("juiceshop", "webgoat")),
)

ATTACK_FAMILY_PATTERNS = {
    "sqli": re.compile(r"(?:sql|union|select|or\s+1=1|injection)", re.I),
    "xss": re.compile(r"(?:xss|crosssitescripting|<script|javascript:|onerror)", re.I),
    "ssrf": re.compile(r"(?:ssrf|169\.254\.169\.254|localhost|127\.0\.0\.1)", re.I),
    "path_traversal": re.compile(r"(?:traversal|\.\./|%2e%2e)", re.I),
    "ssti": re.compile(r"(?:ssti|\{\{.*\}\})", re.I),
    "command_injection": re.compile(r"(?:command|cmd|;\s*(?:cat|ls|whoami|id)\b|\|\s*(?:cat|ls|whoami|id)\b)", re.I),
}


def infer_attack_family(frame: pd.DataFrame) -> pd.Series:
    """Infer coarse attack families when raw datasets do not provide them."""
    text = frame[["path", "body"]].fillna("").astype(str).agg(" ".join, axis=1)
    families = pd.Series("unknown", index=frame.index, dtype="object")
    for family, pattern in ATTACK_FAMILY_PATTERNS.items():
        families = families.mask((families == "unknown") & text.str.contains(pattern, na=False), family)
    LOGGER.info("Inferred attack families: %s", families.value_counts().to_dict())
    return families


def select_cross_application_frames(
    benign: pd.DataFrame,
    attacks: pd.DataFrame,
    experiment: CrossApplicationExperiment,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select train benign, test benign, and attacks for an experiment."""
    train_benign = benign[benign["source_app"].isin(experiment.train_apps)].copy()
    test_benign = benign[benign["source_app"].isin(experiment.test_attack_apps)].copy()
    test_attacks = attacks[attacks["source_app"].isin(experiment.test_attack_apps)].copy()
    return train_benign, test_benign, test_attacks
