from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets import deterministic_sample, load_profile_splits
from src.evaluation.benchmark import benchmark_inference
from src.evaluation.experiments import CROSS_APPLICATION_EXPERIMENTS, infer_attack_family
from src.evaluation.metrics import evaluate_scores
from src.evaluation.reporting import save_baseline_summary, save_metric_table
from src.models.baselines import CharacterCNNBaseline, CharacterCNNConfig, TfidfIsolationForest, TfidfOneClassSVM
from src.utils.config import configure_logging, load_config, set_seed

LOGGER = logging.getLogger(__name__)
DEFAULT_MODELS = ("tfidf_isolation_forest", "tfidf_one_class_svm")


def _build_model(name: str, options: dict[str, Any], random_state: int) -> object:
    if name == "tfidf_isolation_forest":
        return TfidfIsolationForest(
            contamination=float(options["contamination"]),
            max_features=int(options["max_features"]),
            random_state=random_state,
        )
    if name == "tfidf_one_class_svm":
        return TfidfOneClassSVM(nu=float(options["nu"]), max_features=int(options["max_features"]))
    if name == "character_cnn":
        return CharacterCNNBaseline(
            CharacterCNNConfig(
                max_length=int(options["max_length"]),
                embedding_dim=int(options["embedding_dim"]),
                channels=int(options["channels"]),
                epochs=int(options["epochs"]),
                batch_size=int(options["batch_size"]),
                learning_rate=float(options["learning_rate"]),
                quantile=float(options.get("threshold_quantile", 0.99)),
                device=str(options["device"]),
            )
        )
    raise ValueError(f"Unknown baseline model: {name}")


def _enabled_models(config: dict[str, Any], requested: tuple[str, ...] | None) -> tuple[str, ...]:
    options = config["baselines"]["models"]
    names = requested or tuple(name for name, values in options.items() if values.get("enabled", False))
    unknown = set(names) - set(options)
    if unknown:
        raise ValueError(f"Unknown baseline models: {sorted(unknown)}")
    return names


def _family_metrics(
    benign_scores: np.ndarray,
    attack_scores: np.ndarray,
    attack_frame: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for family, family_frame in attack_frame.groupby("attack_family"):
        selected = attack_frame["attack_family"].to_numpy() == family
        metrics[str(family)] = evaluate_scores(
            np.concatenate([np.zeros(len(benign_scores)), np.ones(len(family_frame))]),
            np.concatenate([benign_scores, attack_scores[selected]]),
            threshold,
        )
    return metrics


def run_baselines(
    config: dict[str, Any],
    output_dir: str | Path,
    requested_models: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Fit benign-only baselines and evaluate cross-application generalization."""
    seed = int(config.get("seed", 42))
    baseline_config = config["baselines"]
    profile = str(baseline_config["profile"])
    splits = load_profile_splits(config["paths"]["representations_dir"], profile)
    normalized_attacks = pd.read_csv(Path(config["paths"]["normalized_dir"]) / "test_attack.csv")
    if len(normalized_attacks) != len(splits["test_attack"]):
        raise ValueError("Normalized attack rows and representation rows are not aligned")
    splits["test_attack"]["attack_family"] = infer_attack_family(normalized_attacks).to_numpy()
    models = _enabled_models(config, requested_models)
    output = Path(output_dir)
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "baseline_metrics.json"
    results: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    for experiment in CROSS_APPLICATION_EXPERIMENTS:
        train_frame = splits["train_benign"][splits["train_benign"]["source_app"].isin(experiment.train_apps)].copy()
        benign_frame = splits["test_benign"][splits["test_benign"]["source_app"].isin(experiment.test_attack_apps)].copy()
        attack_frame = splits["test_attack"][splits["test_attack"]["source_app"].isin(experiment.test_attack_apps)].copy()
        if train_frame.empty or benign_frame.empty or attack_frame.empty:
            LOGGER.warning("Skipping %s due to an empty split", experiment.name)
            continue
        results.setdefault(experiment.name, {})
        test_text = benign_frame["text"].tolist() + attack_frame["text"].tolist()
        labels = np.concatenate([np.zeros(len(benign_frame)), np.ones(len(attack_frame))])
        for model_name in models:
            options = baseline_config["models"][model_name]
            fit_frame = deterministic_sample(train_frame, int(options["max_train_samples"]), seed)
            fit_text = fit_frame["text"].tolist()
            model = _build_model(model_name, options, seed)
            started = time.perf_counter()
            model.fit(fit_text, threshold_quantile=float(baseline_config["threshold_quantile"]))
            fit_seconds = time.perf_counter() - started
            scores = model.score(test_text)
            benign_scores = scores[: len(benign_frame)]
            attack_scores = scores[len(benign_frame) :]
            metrics = evaluate_scores(labels, scores, model.threshold)
            metrics.update(
                {
                    "profile": profile,
                    "model_options": options,
                    "fit_seconds": fit_seconds,
                    "fit_sample_count": len(fit_frame),
                    "test_benign_count": len(benign_frame),
                    "test_attack_count": len(attack_frame),
                    "latency": benchmark_inference(
                        model.score,
                        benign_frame["text"].head(int(baseline_config["latency_samples"])).tolist(),
                    ),
                    "per_family": _family_metrics(benign_scores, attack_scores, attack_frame, model.threshold),
                }
            )
            suffix = ".pt" if model_name == "character_cnn" else ".joblib"
            artifact = model_dir / f"{experiment.name}_{model_name}{suffix}"
            model.save(artifact)
            metrics["artifact"] = str(artifact)
            results[experiment.name][model_name] = metrics
            LOGGER.info("Completed %s / %s in %.2fs", experiment.name, model_name, fit_seconds)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_metric_table(results, output / "baseline_metrics.csv")
    save_baseline_summary(results, output / "baseline_summary.md")
    LOGGER.info("Saved Phase 4 artifacts to %s", output)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate benign-only Phase 4 baselines.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="reports/artifacts/phase4_baselines")
    parser.add_argument("--models", nargs="+", choices=("tfidf_isolation_forest", "tfidf_one_class_svm", "character_cnn"))
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_baselines(config, args.output_dir, tuple(args.models) if args.models else None)


if __name__ == "__main__":
    main()
