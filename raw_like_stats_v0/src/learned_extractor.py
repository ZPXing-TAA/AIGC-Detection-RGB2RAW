from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class RawLikeCNN(nn.Module):
    """Small supervised CNN whose penultimate vector is the learned feature."""

    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 32,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ):
        super(RawLikeCNN, self).__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.backbone = nn.Sequential(
            self._block(in_channels, c1),
            nn.MaxPool2d(2),
            self._block(c1, c2),
            nn.MaxPool2d(2),
            self._block(c2, c3),
            nn.MaxPool2d(2),
            self._block(c3, c3),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        emb = self.embedding(feat)
        logit = self.classifier(emb).squeeze(1)
        return logit, emb
