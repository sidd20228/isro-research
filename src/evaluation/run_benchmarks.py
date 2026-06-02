from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.benchmark import benchmark_inference
from src.models.baselines import CharacterCNNBaseline, TfidfIsolationForest
from src.models.encoder import ContrastiveDistilBertEncoder
from src.preprocessing import RequestRenderer
from src.utils.config import configure_logging, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CPU inference benchmarks.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--encoder")
    parser.add_argument("--cnn")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--output", default="reports/artifacts/latency.json")
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    normalized = Path(config["paths"]["normalized_dir"])
    frame = pd.read_csv(normalized / "test_benign.csv").head(args.samples)
    renderer = RequestRenderer.from_config(config)
    samples = [renderer.render(row, normalize=False) for row in frame.fillna("").to_dict(orient="records")]
    train = pd.read_csv(normalized / "train_benign.csv").head(25_000)
    train_samples = [renderer.render(row, normalize=False) for row in train.fillna("").to_dict(orient="records")]
    tfidf = TfidfIsolationForest(random_state=int(config["seed"])).fit(train_samples)
    reports = {"tfidf_isolation_forest": benchmark_inference(tfidf.score, samples)}
    if args.cnn:
        cnn = CharacterCNNBaseline.load(args.cnn)
        reports["character_cnn"] = benchmark_inference(cnn.score, samples)
    if args.encoder:
        encoder = ContrastiveDistilBertEncoder.load(args.encoder, int(config["model"]["embedding_dim"]))
        reports["distilbert"] = benchmark_inference(
            lambda batch: encoder.encode(batch, batch_size=1, max_length=int(config["model"]["max_length"]), device="cpu"),
            samples,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
