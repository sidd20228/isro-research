from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.detectors import create_detector
from src.evaluation.experiments import infer_attack_family
from src.evaluation.metrics import evaluate_scores
from src.evaluation.reporting import save_metric_table
from src.models.encoder import ContrastiveDistilBertEncoder
from src.preprocessing import RequestRenderer
from src.utils.config import configure_logging, load_config, set_seed
from src.utils.tracking import save_run_record
from src.visualization.plots import plot_distance_distribution, plot_precision_recall, plot_roc, plot_tsne, plot_umap

LOGGER = logging.getLogger(__name__)


def _timed(stage: str, operation: Any) -> Any:
    started = time.perf_counter()
    LOGGER.info("Starting %s", stage)
    result = operation()
    LOGGER.info("Finished %s in %.2fs", stage, time.perf_counter() - started)
    return result


def _render(frame: pd.DataFrame, renderer: RequestRenderer, name: str) -> list[str]:
    started = time.perf_counter()
    records = frame.fillna("").to_dict(orient="records")
    rendered = [renderer.render(row, normalize=False) for row in records]
    LOGGER.info("Rendered %s: rows=%d elapsed=%.2fs", name, len(rendered), time.perf_counter() - started)
    return rendered


def run_embedding_evaluation(config: dict[str, Any], encoder_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Evaluate trained embeddings with one configured open-set detector."""
    started = time.perf_counter()
    normalized = Path(config["paths"]["normalized_dir"])
    LOGGER.info("Loading normalized evaluation splits from %s", normalized)
    train = pd.read_csv(normalized / "train_benign.csv")
    benign = pd.read_csv(normalized / "test_benign.csv")
    attacks = pd.read_csv(normalized / "test_attack.csv")
    LOGGER.info("Loaded splits: train_benign=%d test_benign=%d test_attack=%d", len(train), len(benign), len(attacks))
    attacks["attack_family"] = infer_attack_family(attacks)
    renderer = RequestRenderer.from_config(config)
    model_config = config["model"]
    evaluation_config = config["embedding_evaluation"]
    LOGGER.info("Loading encoder from %s", encoder_path)
    encoder = ContrastiveDistilBertEncoder.load(encoder_path, int(model_config["embedding_dim"]))
    encode_options = {
        "batch_size": int(evaluation_config["batch_size"]),
        "max_length": int(model_config["max_length"]),
        "device": str(evaluation_config["device"]),
        "mixed_precision": bool(evaluation_config["mixed_precision"]),
    }
    LOGGER.info("Embedding inference options: %s", encode_options)
    train_embeddings = encoder.encode(_render(train, renderer, "train_benign"), description="Encoding train benign", **encode_options)
    benign_embeddings = encoder.encode(_render(benign, renderer, "test_benign"), description="Encoding test benign", **encode_options)
    attack_embeddings = encoder.encode(_render(attacks, renderer, "test_attack"), description="Encoding test attacks", **encode_options)
    detector_config = config["detector"]
    detector_name = str(detector_config["name"])
    detector_options: dict[str, Any] = {}
    if detector_name in {"mahalanobis", "normalizing_flow"}:
        detector_options["quantile"] = float(detector_config["quantile"])
    if detector_name == "isolation_forest":
        detector_options["contamination"] = float(detector_config["contamination"])
    detector = create_detector(detector_name, **detector_options)
    _timed(f"fitting {detector_name} detector", lambda: detector.fit(train_embeddings))
    embeddings = np.concatenate([benign_embeddings, attack_embeddings])
    labels = np.concatenate([np.zeros(len(benign)), np.ones(len(attacks))])
    scores = _timed(f"scoring {detector_name} detector", lambda: detector.score(embeddings))
    metrics = evaluate_scores(labels, scores, detector.threshold)
    metrics["embedding_evaluation"] = encode_options
    metrics["per_family"] = {}
    for family, family_frame in attacks.groupby("attack_family"):
        selected = attacks["attack_family"].to_numpy() == family
        family_labels = np.concatenate([np.zeros(len(benign)), np.ones(len(family_frame))])
        family_scores = np.concatenate([scores[: len(benign)], scores[len(benign) :][selected]])
        metrics["per_family"][family] = evaluate_scores(family_labels, family_scores, detector.threshold)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "embedding_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_metric_table({"all_benign_all_attacks": {detector_name: metrics}}, output / "embedding_metrics.csv")
    detector.save(output / f"{detector_name}.joblib")
    LOGGER.info("Saved metrics and detector artifacts to %s", output)
    _timed("ROC plot", lambda: plot_roc(labels, scores, output / "roc.png"))
    _timed("precision-recall plot", lambda: plot_precision_recall(labels, scores, output / "precision_recall.png"))
    _timed("distance-distribution plot", lambda: plot_distance_distribution(labels, scores, output / "distance_distribution.png"))
    random = np.random.default_rng(int(config["seed"]))
    plot_count = min(int(evaluation_config["plot_samples"]), len(embeddings))
    plot_indices = random.choice(len(embeddings), size=plot_count, replace=False)
    plot_embeddings = embeddings[plot_indices]
    plot_labels = labels[plot_indices]
    _timed(f"t-SNE plot ({plot_count} samples)", lambda: plot_tsne(plot_embeddings, plot_labels, output / "tsne.png", int(config["seed"])))
    try:
        _timed(f"UMAP plot ({plot_count} samples)", lambda: plot_umap(plot_embeddings, plot_labels, output / "umap.png", int(config["seed"])))
    except ImportError:
        LOGGER.warning("UMAP is not installed; skipping UMAP plot")
    save_run_record(output / "runs", detector_name, config, metrics)
    LOGGER.info("Embedding evaluation completed in %.2fs", time.perf_counter() - started)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained security embeddings with an open-set detector.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoder", default="reports/artifacts/distilbert_encoder")
    parser.add_argument("--output", default="reports/artifacts/embedding_evaluation")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Override embedding_evaluation.device.")
    parser.add_argument("--batch-size", type=int, help="Override embedding_evaluation.batch_size.")
    parser.add_argument("--plot-samples", type=int, help="Override embedding_evaluation.plot_samples.")
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    for argument, config_key in (
        (args.device, "device"),
        (args.batch_size, "batch_size"),
        (args.plot_samples, "plot_samples"),
    ):
        if argument is not None:
            config["embedding_evaluation"][config_key] = argument
    run_embedding_evaluation(config, args.encoder, args.output)


if __name__ == "__main__":
    main()
