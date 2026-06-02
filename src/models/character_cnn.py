from __future__ import annotations

import torch
from torch import nn


class CharacterCNNAutoencoder(nn.Module):
    """Small convolutional autoencoder for benign-request reconstruction."""

    def __init__(self, vocabulary_size: int, embedding_dim: int, channels: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Conv1d(embedding_dim, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Conv1d(channels, vocabulary_size, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs).transpose(1, 2)
        return self.decoder(self.encoder(embedded)).transpose(1, 2)
