"""Loss functions and factory for Ki training experiments."""

from __future__ import annotations

import torch
from torch import nn


LOSS_KEYS = ("ce", "label_smooth", "weighted_ce", "focal")

# Losses that already re-weight classes — do not also use WeightedRandomSampler.
LOSS_KEYS_WITH_CLASS_WEIGHTS = frozenset({"weighted_ce", "focal"})


def loss_uses_class_weights(loss_key: str) -> bool:
    return loss_key in LOSS_KEYS_WITH_CLASS_WEIGHTS


class FocalLoss(nn.Module):
    """Down-weights easy examples; helps hard misclassified vehicle segments."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        targets_one_hot = nn.functional.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        log_pt = (log_probs * targets_one_hot).sum(dim=1)
        focal = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            class_w = self.weight[targets]
            focal = focal * class_w
        return focal.mean()


def class_weights_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights for imbalanced Ki subsets."""
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def build_loss(
    loss_key: str,
    num_classes: int,
    train_labels: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    device = device or torch.device("cpu")
    if loss_key == "ce":
        return nn.CrossEntropyLoss()
    if loss_key == "label_smooth":
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    if loss_key == "weighted_ce":
        if train_labels is None:
            raise ValueError("weighted_ce requires train_labels")
        w = class_weights_from_labels(train_labels, num_classes).to(device)
        return nn.CrossEntropyLoss(weight=w)
    if loss_key == "focal":
        weight = None
        if train_labels is not None:
            weight = class_weights_from_labels(train_labels, num_classes).to(device)
        return FocalLoss(gamma=2.0, weight=weight)
    raise ValueError(f"Unknown loss_key: {loss_key}. Choose from {LOSS_KEYS}")
