from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.cifar100.train_detector import (
    _head_epoch,
    backbone_state_sha256,
    extract_features,
)


class _TinyConvNext(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 4)
        self.head = nn.Module()
        self.head.fc = nn.Linear(4, 2)
        self.pretrained_cfg = {"classifier": "head.fc"}

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs.mean(dim=(-2, -1)))

    def forward_head(
        self, features: torch.Tensor, *, pre_logits: bool = False
    ) -> torch.Tensor:
        return features if pre_logits else self.head.fc(features)

    def get_classifier(self) -> nn.Module:
        return self.head.fc


class DetectorHeadTests(unittest.TestCase):
    def test_backbone_fingerprint_excludes_new_classifier(self) -> None:
        model = _TinyConvNext()
        initial = backbone_state_sha256(model)
        with torch.no_grad():
            model.head.fc.weight.add_(1.0)
        self.assertEqual(backbone_state_sha256(model), initial)
        with torch.no_grad():
            model.backbone.weight.add_(1.0)
        self.assertNotEqual(backbone_state_sha256(model), initial)

    def test_features_are_extracted_once_and_head_can_be_optimized(self) -> None:
        torch.manual_seed(3)
        model = _TinyConvNext()
        inputs = torch.randn(8, 3, 4, 4)
        targets = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)
        features, extracted_targets = extract_features(
            model, loader, device=torch.device("cpu")
        )
        self.assertEqual(tuple(features.shape), (8, 4))
        self.assertTrue(torch.equal(extracted_targets, targets))

        feature_loader = DataLoader(
            TensorDataset(features, targets), batch_size=4, shuffle=False
        )
        head = model.get_classifier()
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)
        metrics = _head_epoch(
            head,
            feature_loader,
            nn.CrossEntropyLoss(),
            torch.device("cpu"),
            optimizer,
        )
        self.assertEqual(metrics["samples"], 8)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
