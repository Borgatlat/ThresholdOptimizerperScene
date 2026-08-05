"""Lightweight CNN classifier (Ki) for 2D spectrogram inputs."""

from __future__ import annotations

import torch
from torch import nn


def _make_conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SpectrogramCNN(nn.Module):
    """Single-modality CNN for spectrograms.

    Input: (batch, 1, freq_bins, time_frames) -> logits (batch, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        width: int = 16,
        depth: int = 3,
        hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        channels = [width * (2**i) for i in range(depth)]
        blocks: list[nn.Module] = []
        in_ch = in_channels
        for out_ch in channels:
            blocks.append(_make_conv_block(in_ch, out_ch))
            in_ch = out_ch
        blocks.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.features = nn.Sequential(*blocks)

        flat = channels[-1] * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
