from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

LOGGER = logging.getLogger(__name__)


class ContrastiveDistilBertEncoder(nn.Module):
    """Encoder-only DistilBERT with a compact security-embedding projection."""

    def __init__(self, model_name: str = "distilbert-base-uncased", embedding_dim: int = 128) -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.backbone.config.hidden_size, embedding_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return nn.functional.normalize(self.projection(pooled), dim=1)

    def encode(
        self,
        samples: list[str],
        batch_size: int = 32,
        max_length: int = 256,
        device: str = "cpu",
    ) -> np.ndarray:
        self.to(device)
        self.eval()
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                tokens = self.tokenizer(
                    samples[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(device)
                batches.append(self(tokens["input_ids"], tokens["attention_mask"]).cpu().numpy())
        return np.concatenate(batches) if batches else np.empty((0, self.embedding_dim))

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output / "backbone")
        self.tokenizer.save_pretrained(output / "backbone")
        torch.save(self.projection.state_dict(), output / "projection.pt")

    @classmethod
    def load(cls, path: str | Path, embedding_dim: int = 128) -> "ContrastiveDistilBertEncoder":
        output = Path(path)
        instance = cls(str(output / "backbone"), embedding_dim=embedding_dim)
        instance.projection.load_state_dict(torch.load(output / "projection.pt", map_location="cpu"))
        return instance


class TextPairDataset(Dataset[tuple[str, str]]):
    """Lazy positive-pair dataset."""

    def __init__(self, samples: list[str], augmenter: object) -> None:
        self.samples = samples
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[str, str]:
        sample = self.samples[index]
        return self.augmenter.augment(sample), self.augmenter.augment(sample)  # type: ignore[attr-defined]
