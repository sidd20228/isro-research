from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.models.encoder import ContrastiveDistilBertEncoder, TextPairDataset
from src.preprocessing import RequestRenderer
from src.training.augmentations import RequestAugmenter
from src.utils.config import configure_logging, load_config, set_seed

LOGGER = logging.getLogger(__name__)


def nt_xent_loss(first: torch.Tensor, second: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute symmetric in-batch contrastive loss."""
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels)) / 2


def train_encoder(config: dict[str, object]) -> ContrastiveDistilBertEncoder:
    """Train a security encoder only on benign requests."""
    model_config = config["model"]  # type: ignore[index]
    training_config = config["training"]  # type: ignore[index]
    normalized_dir = Path(config["paths"]["normalized_dir"])  # type: ignore[index]
    frame = pd.read_csv(normalized_dir / "train_benign.csv").fillna("")
    renderer = RequestRenderer.from_config(config)  # type: ignore[arg-type]
    samples = [renderer.render(row, normalize=False) for row in frame.to_dict(orient="records")]
    dataset = TextPairDataset(samples, RequestAugmenter())
    loader = DataLoader(dataset, batch_size=int(training_config["batch_size"]), shuffle=True)
    device = str(training_config["device"])
    model = ContrastiveDistilBertEncoder(str(model_config["encoder_name"]), int(model_config["embedding_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config["learning_rate"]))
    for epoch in range(int(training_config["epochs"])):
        model.train()
        for first, second in loader:
            tokens_first = model.tokenizer(list(first), padding=True, truncation=True, max_length=int(model_config["max_length"]), return_tensors="pt").to(device)
            tokens_second = model.tokenizer(list(second), padding=True, truncation=True, max_length=int(model_config["max_length"]), return_tensors="pt").to(device)
            loss = nt_xent_loss(
                model(tokens_first["input_ids"], tokens_first["attention_mask"]),
                model(tokens_second["input_ids"], tokens_second["attention_mask"]),
                float(model_config["temperature"]),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        LOGGER.info("Finished contrastive epoch %d/%d", epoch + 1, int(training_config["epochs"]))
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="reports/artifacts/distilbert_encoder")
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    train_encoder(config).save(args.output)


if __name__ == "__main__":
    main()
