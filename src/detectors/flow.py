from __future__ import annotations

import math

import torch
from torch import nn


class AffineFlow(nn.Module):
    """Diagonal affine flow with exact likelihood and stable optimization."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.location = nn.Parameter(torch.zeros(dimension))
        self.log_scale = nn.Parameter(torch.zeros(dimension))

    def log_prob(self, samples: torch.Tensor) -> torch.Tensor:
        z = (samples - self.location) * torch.exp(-self.log_scale)
        base = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=1)
        return base - self.log_scale.sum()
