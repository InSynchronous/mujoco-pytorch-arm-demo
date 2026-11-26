"""Vision + joint-position policy for 6 motor deltas."""

from __future__ import annotations

import torch
import torch.nn as nn


class Policy(nn.Module):
    """Small CNN over the frame, concatenated with joint positions."""

    def __init__(self, q_dim: int = 6, action_dim: int = 6):
        super().__init__()

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),   # 128 -> 64
            block(16, 32, 2),        # 64 -> 32
            block(32, 48, 2),        # 32 -> 16
            block(48, 48, 2),        # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.Linear(48 * 8 * 8 + q_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

    def forward(self, image: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        features = self.features(image).flatten(1)
        return self.head(torch.cat([features, q], dim=1))


def make_policy() -> Policy:
    return Policy()
