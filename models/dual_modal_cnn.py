"""Dual-modality CNN fusing microphone and geophone spectrograms."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.spectrogram_cnn import SpectrogramCNN, _make_conv_block


class _ModalityTower(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        channels = [width * (2**i) for i in range(depth)]
        blocks: list[nn.Module] = []
        in_ch = in_channels
        for out_ch in channels:
            blocks.append(_make_conv_block(in_ch, out_ch))
            in_ch = out_ch
        blocks.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.encoder = nn.Sequential(*blocks)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DualModalCNN(nn.Module):
    """Separate mic/geo encoders fused before classification (Both modality Ki)."""

    def __init__(
        self,
        num_classes: int,
        width: int = 24,
        depth: int = 3,
        hidden: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mic_tower = _ModalityTower(1, width, depth)
        self.geo_tower = _ModalityTower(1, width, depth)

        fused = self.mic_tower.out_channels + self.geo_tower.out_channels
        flat = fused * 4 * 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, mic: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        # Geo time axis is shorter (129x2 vs 129x24); resize inside the encoder only.
        if geo.shape[-2:] != mic.shape[-2:]:
            geo = F.interpolate(
                geo,
                size=mic.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        mic_feat = self.mic_tower(mic)
        geo_feat = self.geo_tower(geo)
        fused = torch.cat([mic_feat, geo_feat], dim=1)
        return self.classifier(fused)


def build_ki_model(ki_name: str, num_classes: int) -> nn.Module:
    """Instantiate architecture sized for paper Table IV param counts."""
    # Target params (paper): K0 129698, K1 356610, K2 130469, K3 1217109,
    # K4/K5 80355, K6 129955.
    presets: dict[str, dict] = {
        "K0": {"kind": "dual", "width": 16, "depth": 2, "hidden": 128},
        "K1": {"kind": "dual", "width": 18, "depth": 3, "hidden": 128},
        "K2": {"kind": "dual", "width": 16, "depth": 2, "hidden": 128},
        # width=27, depth=4, hidden=96 -> ~1,218,173 params (paper: 1,217,109)
        "K3": {"kind": "dual", "width": 27, "depth": 4, "hidden": 96},
        "K4": {"kind": "mic", "width": 16, "depth": 2, "hidden": 64},
        "K5": {"kind": "mic", "width": 16, "depth": 2, "hidden": 64},
        "K6": {"kind": "dual", "width": 16, "depth": 2, "hidden": 128},
        # depth=5 exceeds MaxPool budget for 129x24 spectrograms; use depth=4 + wider towers.
        "Kdet": {"kind": "dual", "width": 48, "depth": 4, "hidden": 224},
    }
    cfg = presets[ki_name]
    if cfg["kind"] == "mic":
        return SpectrogramCNN(
            num_classes=num_classes,
            width=cfg["width"],
            depth=cfg["depth"],
            hidden=cfg["hidden"],
        )
    return DualModalCNN(
        num_classes=num_classes,
        width=cfg["width"],
        depth=cfg["depth"],
        hidden=cfg["hidden"],
    )
