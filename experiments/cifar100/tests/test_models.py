from __future__ import annotations

import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from experiments.cifar100.models import (
    CHECKPOINT_SCHEMA_VERSION,
    CandidateSpec,
    build_model,
    build_resnet18,
    build_wrn_16_2,
    build_wrn_28_10,
    candidate_specs,
    config_hash,
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
    transfer_wrn_base_features,
)


def synthetic_groups() -> OrderedDict[str, tuple[str, ...]]:
    return OrderedDict(
        (
            f"coarse_{coarse:02d}",
            tuple(f"fine_{coarse:02d}_{fine}" for fine in range(5)),
        )
        for coarse in range(20)
    )


class CifarModelTests(unittest.TestCase):
    def test_model_output_shapes_and_cifar_stems(self) -> None:
        inputs = torch.randn(2, 3, 32, 32)
        models = (
            (build_wrn_16_2(5), 5),
            (build_wrn_28_10(100), 100),
            (build_resnet18(20, pretrained=False), 20),
        )
        for model, classes in models:
            model.eval()
            with self.subTest(model=type(model).__name__, classes=classes):
                with torch.inference_mode():
                    self.assertEqual(model(inputs).shape, (2, classes))

        resnet = models[-1][0]
        self.assertEqual(resnet.conv1.kernel_size, (3, 3))
        self.assertEqual(resnet.conv1.stride, (1, 1))
        self.assertIsInstance(resnet.maxpool, torch.nn.Identity)

    def test_registry_has_twenty_four_candidates_and_separate_initializer(self) -> None:
        groups = synthetic_groups()
        registry = candidate_specs(groups)

        self.assertEqual(len(registry.candidates), 24)
        self.assertEqual(
            sum(item.role == "specialized" for item in registry.candidates), 20
        )
        self.assertEqual(
            sum(item.role == "intermediate" for item in registry.candidates), 2
        )
        self.assertEqual(sum(item.role == "global" for item in registry.candidates), 2)
        self.assertFalse(registry.wrn_base_initializer.is_candidate)
        self.assertNotIn(registry.wrn_base_initializer, registry.candidates)
        self.assertEqual(registry.detector.role, "detector")
        self.assertEqual(registry.detector.kind, "detector")
        self.assertEqual(registry.detector.architecture, "convnextv2_large")
        self.assertEqual(registry.detector.num_classes, 100)
        self.assertEqual(registry.by_id["wrn16_2_coarse"].kind, "identifier")
        self.assertEqual(
            registry.by_id["wrn16_2_specialist_coarse_00"].output_labels,
            groups["coarse_00"],
        )

    def test_independent_builds_do_not_share_parameters(self) -> None:
        spec = candidate_specs(synthetic_groups()).candidates[0]
        first = build_model(spec)
        second = build_model(spec)

        first_parameter = next(first.parameters())
        second_parameter = next(second.parameters())
        self.assertNotEqual(first_parameter.data_ptr(), second_parameter.data_ptr())
        second_before = second_parameter.detach().clone()
        with torch.no_grad():
            first_parameter.add_(1.0)
        self.assertTrue(torch.equal(second_parameter, second_before))

    def test_checkpoint_round_trip_and_config_hash(self) -> None:
        spec = candidate_specs(synthetic_groups()).candidates[0]
        torch.manual_seed(7)
        model = build_model(spec).eval()
        inputs = torch.randn(2, 3, 32, 32)
        with torch.inference_mode():
            expected = model(inputs)
        training_config = {"epochs": 2, "seed": 7, "schedule": [1, 2]}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "specialist.pt"
            save_checkpoint(
                path,
                model,
                spec,
                training_config,
                metrics={"accuracy": 0.5},
            )
            loaded, metadata = load_checkpoint(path, expected_spec=spec)

        loaded.eval()
        with torch.inference_mode():
            actual = loaded(inputs)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(metadata["schema_version"], CHECKPOINT_SCHEMA_VERSION)
        self.assertEqual(metadata["config_hash"], config_hash(training_config))
        self.assertEqual(len(metadata["checkpoint_sha256"]), 64)

    def test_checkpoint_metadata_load_does_not_construct_model(self) -> None:
        spec = candidate_specs(synthetic_groups()).candidates[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            save_checkpoint(path, build_model(spec), spec, {"seed": 9})
            with mock.patch(
                "experiments.cifar100.models.build_model",
                side_effect=AssertionError("model construction is not allowed"),
            ):
                metadata = load_checkpoint_metadata(path, expected_spec=spec)

        self.assertEqual(metadata["spec"], spec.as_dict())
        self.assertEqual(len(metadata["checkpoint_sha256"]), 64)

    def test_base_initialization_transfers_backbone_but_not_head(self) -> None:
        registry = candidate_specs(synthetic_groups())
        source = build_model(registry.wrn_base_initializer)
        target_spec = registry.candidates[0]
        target = build_model(target_spec)
        with torch.no_grad():
            source.conv1.weight.fill_(0.125)
            source.fc.weight.fill_(0.75)
        target_head_before = target.fc.weight.detach().clone()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.pt"
            save_checkpoint(path, source, registry.wrn_base_initializer, {"seed": 3})
            report = transfer_wrn_base_features(path, target)

        self.assertTrue(torch.equal(target.conv1.weight, source.conv1.weight))
        self.assertTrue(torch.equal(target.fc.weight, target_head_before))
        self.assertIn("conv1.weight", report.copied_keys)
        self.assertIn("fc.weight", report.skipped_keys)

    def test_pretrained_wrn_requires_explicit_base_transfer(self) -> None:
        spec = CandidateSpec(
            candidate_id="specialist",
            role="specialized",
            kind="specialized",
            architecture="wrn16_2",
            output_labels=("a", "b", "c", "d", "e"),
            group="group",
        )
        with self.assertRaisesRegex(ValueError, "transfer_wrn_base_features"):
            build_model(spec, pretrained=True)

    def test_convnext_detector_requests_timm_pretrained_backbone(self) -> None:
        spec = candidate_specs(synthetic_groups()).detector
        fake_model = torch.nn.Linear(3, spec.num_classes)
        fake_timm = SimpleNamespace(create_model=mock.Mock(return_value=fake_model))
        with mock.patch.dict("sys.modules", {"timm": fake_timm}):
            built = build_model(spec, pretrained=True)

        self.assertIs(built, fake_model)
        fake_timm.create_model.assert_called_once_with(
            "convnextv2_large.fcmae_ft_in1k",
            pretrained=True,
            num_classes=100,
        )


if __name__ == "__main__":
    unittest.main()
