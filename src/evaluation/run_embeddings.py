from __future__ import annotations

import argparse
import json
import logging
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


def _render(frame: pd.DataFrame, renderer: RequestRenderer) -> list[str]:
    return [renderer.render(row, normalize=False) for row in frame.fillna("").to_dict(orient="records")]


def run_embedding_evaluation(config: dict[str, Any], encoder_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Evaluate trained embeddings with one configured open-set detector."""
    normalized = Path(config["paths"]["normalized_dir"])
    train = pd.read_csv(normalized / "train_benign.csv")
    benign = pd.read_csv(normalized / "test_benign.csv")
    attacks = pd.read_csv(normalized / "test_attack.csv")
    attacks["attack_family"] = infer_attack_family(attacks)
    renderer = RequestRenderer.from_config(config)
    model_config = config["model"]
    encoder = ContrastiveDistilBertEncoder.load(encoder_path, int(model_config["embedding_dim"]))
    encode_options = {"batch_size": int(config["training"]["batch_size"]), "max_length": int(model_config["max_length"]), "device": str(config["training"]["device"])}
    train_embeddings = encoder.encode(_render(train, renderer), **encode_options)
    benign_embeddings = encoder.encode(_render(benign, renderer), **encode_options)
    attack_embeddings = encoder.encode(_render(attacks, renderer), **encode_options)
    detector_config = config["detector"]
    detector_name = str(detector_config["name"])
    detector_options: dict[str, Any] = {}
    if detector_name in {"mahalanobis", "normalizing_flow"}:
        detector_options["quantile"] = float(detector_config["quantile"])
    if detector_name == "isolation_forest":
        detector_options["contamination"] = float(detector_config["contamination"])
    detector = create_detector(detector_name, **detector_options)
    detector.fit(train_embeddings)
    embeddings = np.concatenate([benign_embeddings, attack_embeddings])
    labels = np.concatenate([np.zeros(len(benign)), np.ones(len(attacks))])
    scores = detector.score(embeddings)
    metrics = evaluate_scores(labels, scores, detector.threshold)
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
    plot_roc(labels, scores, output / "roc.png")
    plot_precision_recall(labels, scores, output / "precision_recall.png")
    plot_distance_distribution(labels, scores, output / "distance_distribution.png")
    random = np.random.default_rng(int(config["seed"]))
    plot_indices = random.choice(len(embeddings), size=min(5_000, len(embeddings)), replace=False)
    plot_embeddings = embeddings[plot_indices]
    plot_labels = labels[plot_indices]
    plot_tsne(plot_embeddings, plot_labels, output / "tsne.png", int(config["seed"]))
    try:
        plot_umap(plot_embeddings, plot_labels, output / "umap.png", int(config["seed"]))
    except ImportError:
        LOGGER.warning("UMAP is not installed; skipping UMAP plot")
    save_run_record(output / "runs", detector_name, config, metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoder", default="reports/artifacts/distilbert_encoder")
    parser.add_argument("--output", default="reports/artifacts/embedding_evaluation")
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    run_embedding_evaluation(config, args.encoder, args.output)


if __name__ == "__main__":
    main()
