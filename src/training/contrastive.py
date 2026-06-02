from __future__ import annotations

import argparse
import json
import logging
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import deterministic_sample, load_profile_splits
from src.models.encoder import ContrastiveDistilBertEncoder, TextPairDataset
from src.training.augmentations import RequestAugmenter
from src.utils.config import configure_logging, load_config, set_seed

LOGGER = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Serializable measurements for one completed training epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    elapsed_seconds: float
    optimizer_steps: int


def nt_xent_loss(first: torch.Tensor, second: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute symmetric in-batch contrastive loss."""
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return (torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels)) / 2


def _autocast(device: str, enabled: bool) -> Any:
    return torch.autocast(device_type="cuda", dtype=torch.float16) if enabled and device.startswith("cuda") else nullcontext()


def _create_scaler(device: str, enabled: bool) -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=enabled and device.startswith("cuda"))


def _forward_pair(
    model: ContrastiveDistilBertEncoder,
    first: list[str] | tuple[str, ...],
    second: list[str] | tuple[str, ...],
    device: str,
    max_length: int,
    temperature: float,
    mixed_precision: bool,
) -> torch.Tensor:
    tokens_first = model.tokenizer(list(first), padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    tokens_second = model.tokenizer(list(second), padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    with _autocast(device, mixed_precision):
        return nt_xent_loss(
            model(tokens_first["input_ids"], tokens_first["attention_mask"]),
            model(tokens_second["input_ids"], tokens_second["attention_mask"]),
            temperature,
        )


def _validate(
    model: ContrastiveDistilBertEncoder,
    loader: DataLoader[tuple[str, str]],
    device: str,
    max_length: int,
    temperature: float,
    mixed_precision: bool,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for first, second in tqdm(loader, desc="Validation", leave=False):
            loss = _forward_pair(model, first, second, device, max_length, temperature, mixed_precision)
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def _checkpoint_payload(
    model: ContrastiveDistilBertEncoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    completed_epoch: int,
    history: list[EpochMetrics],
) -> dict[str, Any]:
    return {
        "completed_epoch": completed_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "history": [asdict(item) for item in history],
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def save_checkpoint(
    output_dir: str | Path,
    model: ContrastiveDistilBertEncoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    completed_epoch: int,
    history: list[EpochMetrics],
) -> Path:
    """Persist resumable model, optimizer, scaler, and RNG state."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"epoch_{completed_epoch:03d}.pt"
    torch.save(_checkpoint_payload(model, optimizer, scaler, completed_epoch, history), path)
    LOGGER.info("Saved checkpoint to %s", path)
    return path


def load_checkpoint(
    path: str | Path,
    model: ContrastiveDistilBertEncoder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: str,
) -> tuple[int, list[EpochMetrics]]:
    """Restore training progress from an epoch checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"].cpu())
    if checkpoint["cuda_random_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    history = [EpochMetrics(**item) for item in checkpoint["history"]]
    completed_epoch = int(checkpoint["completed_epoch"])
    LOGGER.info("Resumed checkpoint %s after epoch %d", path, completed_epoch)
    return completed_epoch, history


def _save_training_metadata(
    output_dir: str | Path,
    config: dict[str, Any],
    history: list[EpochMetrics],
    train_samples: int,
    validation_samples: int,
    resumed_from: str | None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "training_config": config["training"],
        "model_config": config["model"],
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "resumed_from": resumed_from,
        "history": [asdict(item) for item in history],
    }
    (output / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def train_encoder(
    config: dict[str, Any],
    output_dir: str | Path,
    resume_from: str | Path | None = None,
    max_train_samples: int | None = None,
) -> ContrastiveDistilBertEncoder:
    """Train a resumable benign-only contrastive security encoder."""
    model_config = config["model"]
    training_config = config["training"]
    seed = int(config.get("seed", 42))
    profile = str(training_config.get("profile", config["representation"]["default_profile"]))
    splits = load_profile_splits(config["paths"]["representations_dir"], profile)
    configured_cap = int(training_config.get("max_train_samples", 0))
    active_cap = configured_cap if max_train_samples is None else max_train_samples
    train_frame = deterministic_sample(splits["train_benign"], active_cap, seed)
    validation_frame = deterministic_sample(splits["test_benign"], int(training_config["validation_samples"]), seed)
    augmenter = RequestAugmenter()
    train_loader = DataLoader(
        TextPairDataset(train_frame["text"].tolist(), augmenter),
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(training_config["num_workers"]),
    )
    validation_loader = DataLoader(
        TextPairDataset(validation_frame["text"].tolist(), augmenter),
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        num_workers=int(training_config["num_workers"]),
    )
    device = str(training_config["device"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no CUDA device is available")
    mixed_precision = bool(training_config["mixed_precision"]) and device.startswith("cuda")
    accumulation_steps = int(training_config["gradient_accumulation_steps"])
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    checkpoint_every_epochs = int(training_config["checkpoint_every_epochs"])
    if checkpoint_every_epochs < 1:
        raise ValueError("checkpoint_every_epochs must be at least 1")
    model = ContrastiveDistilBertEncoder(str(model_config["encoder_name"]), int(model_config["embedding_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training_config["learning_rate"]))
    scaler = _create_scaler(device, mixed_precision)
    start_epoch = 0
    history: list[EpochMetrics] = []
    if resume_from:
        start_epoch, history = load_checkpoint(resume_from, model, optimizer, scaler, device)
    total_epochs = int(training_config["epochs"])
    LOGGER.info(
        "Training %d samples, validating %d samples, epochs=%d, batch=%d, accumulation=%d, amp=%s, device=%s",
        len(train_frame),
        len(validation_frame),
        total_epochs,
        int(training_config["batch_size"]),
        accumulation_steps,
        mixed_precision,
        device,
    )
    for epoch in range(start_epoch, total_epochs):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        optimizer_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
        for step, (first, second) in enumerate(progress, start=1):
            loss = _forward_pair(
                model,
                first,
                second,
                device,
                int(model_config["max_length"]),
                float(model_config["temperature"]),
                mixed_precision,
            )
            losses.append(float(loss.item()))
            scaler.scale(loss / accumulation_steps).backward()
            should_step = step % accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if step % int(training_config["logging_steps"]) == 0:
                LOGGER.info("Epoch %d step %d/%d loss=%.4f", epoch + 1, step, len(train_loader), loss.item())
        validation_loss = _validate(
            model,
            validation_loader,
            device,
            int(model_config["max_length"]),
            float(model_config["temperature"]),
            mixed_precision,
        )
        metrics = EpochMetrics(
            epoch=epoch + 1,
            train_loss=float(np.mean(losses)) if losses else 0.0,
            validation_loss=validation_loss,
            elapsed_seconds=time.perf_counter() - started,
            optimizer_steps=optimizer_steps,
        )
        history.append(metrics)
        LOGGER.info("Finished epoch %d/%d: %s", epoch + 1, total_epochs, asdict(metrics))
        if (epoch + 1) % checkpoint_every_epochs == 0:
            save_checkpoint(output_dir, model, optimizer, scaler, epoch + 1, history)
        model.save(output_dir)
        _save_training_metadata(output_dir, config, history, len(train_frame), len(validation_frame), str(resume_from) if resume_from else None)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a resumable benign-only contrastive DistilBERT encoder.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="reports/artifacts/distilbert_encoder")
    parser.add_argument("--resume-from", help="Epoch checkpoint created under OUTPUT/checkpoints/.")
    parser.add_argument("--max-train-samples", type=int, help="Override the configured cap for pilot runs; use 0 for all rows.")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="Override training.device.")
    parser.add_argument("--epochs", type=int, help="Override training.epochs.")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, help="Override training.gradient_accumulation_steps.")
    args = parser.parse_args()
    configure_logging()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    for argument, config_key in (
        (args.device, "device"),
        (args.epochs, "epochs"),
        (args.batch_size, "batch_size"),
        (args.gradient_accumulation_steps, "gradient_accumulation_steps"),
    ):
        if argument is not None:
            config["training"][config_key] = argument
    train_encoder(config, args.output, args.resume_from, args.max_train_samples)


if __name__ == "__main__":
    main()
