from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.cifar100.benchmark import (
    BenchmarkConfig,
    LATENCY_SCHEMA_VERSION,
    benchmark_candidates,
    benchmark_manifest,
    load_cascade_validation_inputs,
    load_manifest_candidate,
)
from experiments.cifar100.labels import (
    CIFAR100_PROFILE,
    COARSE_LABEL_NAMES,
    COARSE_TO_FINE_NAMES,
    FINE_LABEL_NAMES,
    FINE_NAME_TO_INDEX,
    FINE_TO_COARSE_INDEX,
)
from experiments.cifar100.models import (
    build_model,
    candidate_specs,
    config_hash,
    file_sha256,
    save_checkpoint,
)
from experiments.cifar100.report import (
    REPORT_SCHEMA_VERSION,
    _canonical_role,
    _latency_lookup,
    _load_training_metrics,
    _verify_latency_identity,
    generate_report,
)


class _InferenceProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise AssertionError("benchmark did not enable evaluation mode")
        if torch.is_grad_enabled():
            raise AssertionError("benchmark did not disable autograd")
        if inputs.shape[0] != 1 or inputs.device.type != "cpu":
            raise AssertionError("benchmark input is not batch-one CPU")
        self.calls += 1
        return self.projection(inputs.mean(dim=(2, 3)))


class BenchmarkTests(unittest.TestCase):
    def test_production_inputs_use_preloaded_cascade_validation_eval_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_path = root / "split.npz"
            split_manifest = root / "split.json"
            split_path.write_bytes(b"split")
            split_manifest.write_text("{}", encoding="utf-8")
            dataset = SimpleNamespace(
                fine_targets=np.arange(12),
                coarse_targets=np.arange(12),
            )
            splits = SimpleNamespace(cascade_validation=np.asarray([7, 9, 11]))
            evaluation_transform = object()
            view = [
                (torch.full((3, 32, 32), float(index)), index)
                for index in range(2)
            ]
            with (
                patch(
                    "experiments.cifar100.data.load_training_dataset",
                    return_value=dataset,
                ) as load_dataset,
                patch(
                    "experiments.cifar100.data.load_split_bundle",
                    return_value=splits,
                ) as load_splits,
                patch(
                    "experiments.cifar100.data.build_evaluation_transform",
                    return_value=evaluation_transform,
                ),
                patch(
                    "experiments.cifar100.data.build_dataset_view",
                    return_value=view,
                ) as build_view,
            ):
                inputs, provenance = load_cascade_validation_inputs(
                    root / "data",
                    split_path,
                    config=BenchmarkConfig(
                        warmups=1,
                        timed_samples=2,
                        input_pool_size=2,
                    ),
                )

            load_dataset.assert_called_once_with((root / "data").resolve(), download=False)
            load_splits.assert_called_once()
            self.assertTrue(
                Path(load_splits.call_args.kwargs["manifest_path"]).samefile(
                    split_manifest
                )
            )
            self.assertNotIn("train", build_view.call_args.kwargs)
            self.assertIs(
                build_view.call_args.kwargs["transform"], evaluation_transform
            )
            np.testing.assert_array_equal(build_view.call_args.args[1], [7, 9])
            self.assertEqual(len(inputs), 2)
            self.assertTrue(all(value.shape == (1, 3, 32, 32) for value in inputs))
            self.assertEqual(
                provenance["source"], "official_training_cascade_validation"
            )
            self.assertTrue(provenance["preloaded_before_timing"])
            self.assertEqual(provenance["preprocessing"]["pipeline"], ["ToTensor", "Normalize"])

    def test_manifest_loader_reconstructs_independent_checkpoints(self) -> None:
        registry = candidate_specs(
            CIFAR100_PROFILE.groups, CIFAR100_PROFILE.global_classes
        )
        spec = registry.by_id["wrn16_2_coarse"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "router.pt"
            training_config = {"seed": 1, "input_resolution": [32, 32]}
            save_checkpoint(
                checkpoint,
                build_model(spec),
                spec,
                training_config,
            )
            entry = {
                "candidate": spec.as_dict(),
                "checkpoint_path": checkpoint.name,
            }
            first, first_metadata = load_manifest_candidate(entry, manifest_dir=root)
            second, second_metadata = load_manifest_candidate(entry, manifest_dir=root)
            self.assertIsNot(first, second)
            self.assertEqual(
                first_metadata["checkpoint_sha256"],
                second_metadata["checkpoint_sha256"],
            )
            self.assertNotEqual(
                next(first.parameters()).data_ptr(),
                next(second.parameters()).data_ptr(),
            )
            training_record = {
                **spec.as_dict(),
                "candidate": spec.as_dict(),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "config_hash": config_hash(training_config),
                "metrics_path": str(root / "metrics.json"),
            }
            training_manifest = root / "training_manifest.json"
            training_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "cifar100-training-manifest/v1",
                        "dataset_id": CIFAR100_PROFILE.dataset_id,
                        "profile_fingerprint": CIFAR100_PROFILE.fingerprint,
                        "official_test_used": False,
                        "candidates": [training_record],
                    }
                ),
                encoding="utf-8",
            )
            latency_path = root / "latency.json"
            payload = benchmark_manifest(
                training_manifest,
                latency_path,
                config=BenchmarkConfig(
                    warmups=1,
                    timed_samples=2,
                    cpu_threads=1,
                    input_pool_size=1,
                ),
                synthetic_inputs=True,
            )
            self.assertEqual(payload["candidates"][0]["candidate_id"], spec.candidate_id)
            self.assertEqual(
                payload["candidates"][0]["checkpoint_sha256"],
                training_record["checkpoint_sha256"],
            )
            self.assertTrue(latency_path.is_file())

    def test_batch_one_inference_benchmark_and_statistics(self) -> None:
        models: list[_InferenceProbe] = []

        def loader(_entry: dict[str, object]) -> tuple[torch.nn.Module, dict[str, str]]:
            model = _InferenceProbe()
            models.append(model)
            return model, {"architecture": "probe", "config_hash": "abc"}

        config = BenchmarkConfig(
            warmups=2,
            timed_samples=7,
            cpu_threads=1,
            input_pool_size=2,
        )
        payload = benchmark_candidates(
            [
                {
                    "candidate_id": "probe_a",
                    "role": "global",
                    "checkpoint_path": "probe_a.pt",
                },
                {
                    "candidate_id": "probe_b",
                    "role": "intermediate",
                    "checkpoint_path": "probe_b.pt",
                },
            ],
            config=config,
            model_loader=loader,
        )

        self.assertEqual(payload["schema_version"], LATENCY_SCHEMA_VERSION)
        self.assertEqual(payload["environment"]["device"], "cpu")
        self.assertEqual(payload["environment"]["cpu_threads"], 1)
        self.assertEqual(payload["environment"]["input_shape"], [1, 3, 32, 32])
        self.assertEqual(len(payload["candidates"]), 2)
        self.assertTrue(all(model.calls == 9 for model in models))
        for candidate in payload["candidates"]:
            self.assertEqual(candidate["warmups"], 2)
            self.assertEqual(candidate["timed_samples"], 7)
            self.assertEqual(candidate["batch_size"], 1)
            self.assertEqual(
                set(candidate["latency_ms"]),
                {"mean", "median", "std", "p95", "p99", "min", "max"},
            )
            self.assertGreaterEqual(candidate["latency_ms"]["p99"], 0.0)

    def test_benchmark_defaults_meet_measurement_budget(self) -> None:
        config = BenchmarkConfig()
        self.assertGreaterEqual(config.warmups, 20)
        self.assertGreaterEqual(config.timed_samples, 500)


def _probabilities(predictions: list[int], classes: int) -> np.ndarray:
    result = np.full((len(predictions), classes), 0.1 / (classes - 1), dtype=np.float32)
    for row, prediction in enumerate(predictions):
        result[row, prediction] = 0.9
    return result


class ReportTests(unittest.TestCase):
    def test_detector_role_is_reported_separately(self) -> None:
        self.assertEqual(
            _canonical_role(
                {"candidate_id": "endpoint", "role": "detector", "kind": "detector"}
            ),
            "detector",
        )

    def test_duplicate_and_mismatched_latency_identities_are_rejected(self) -> None:
        duplicate = {
            "candidates": [
                {"candidate_id": "same"},
                {"candidate_id": "same"},
            ]
        }
        with self.assertRaisesRegex(ValueError, "Duplicate latency candidate id"):
            _latency_lookup(duplicate)

        manifest_entry = {
            "candidate_id": "model",
            "checkpoint": {"sha256": "checkpoint-a", "config_hash": "config-a"},
        }
        with self.assertRaisesRegex(ValueError, "checkpoint_sha256 differs"):
            _verify_latency_identity(
                manifest_entry,
                {
                    "candidate_id": "model",
                    "checkpoint_sha256": "checkpoint-b",
                    "config_hash": "config-a",
                },
            )
        with self.assertRaisesRegex(ValueError, "config_hash differs"):
            _verify_latency_identity(
                manifest_entry,
                {
                    "candidate_id": "model",
                    "checkpoint_sha256": "checkpoint-a",
                    "config_hash": "config-b",
                },
            )

    def test_nested_training_metrics_checksum_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics_path = root / "metrics.json"
            metrics_path.write_text('{"parameter_count": 7}', encoding="utf-8")
            description = {
                "training_metrics": {
                    "path": metrics_path.name,
                    "sha256": file_sha256(metrics_path),
                }
            }
            self.assertEqual(
                _load_training_metrics(description, root)["parameter_count"], 7
            )
            description["training_metrics"]["sha256"] = "incorrect"
            with self.assertRaisesRegex(ValueError, "checksum differs"):
                _load_training_metrics(description, root)

    def test_collector_manifest_contract_and_report_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fine_names = ["beaver", "dolphin", "apple", "bear", "seal", "bottle"]
            true_fine = [FINE_NAME_TO_INDEX[name] for name in fine_names]
            true_coarse = [FINE_TO_COARSE_INDEX[index] for index in true_fine]
            labels = pd.DataFrame(
                {
                    "sample_id": np.arange(len(fine_names)),
                    "source_sample_id": np.arange(len(fine_names)) + 100,
                    "partition": "validation",
                    "true_global_label": fine_names,
                    "true_fine_index": true_fine,
                    "true_fine_name": fine_names,
                    "true_coarse_index": true_coarse,
                    "true_coarse_name": [
                        COARSE_LABEL_NAMES[index] for index in true_coarse
                    ],
                }
            )
            candidate_definitions = [
                ("router", "intermediate", "identifier", None),
                (
                    "aquatic_specialist",
                    "specialized",
                    "specialized",
                    "aquatic_mammals",
                ),
                ("tree_specialist", "specialized", "specialized", "trees"),
                ("global_small", "global", "global", None),
                ("global_large", "global", "global", None),
            ]
            candidates = pd.DataFrame(
                [
                    {
                        "id": candidate_id,
                        "kind": kind,
                        "group": group,
                        "threshold": 0.5,
                        "cost": float(index + 1),
                    }
                    for index, (candidate_id, _, kind, group) in enumerate(
                        candidate_definitions
                    )
                ]
            )
            predictions = {
                "router": [true_coarse[0], true_coarse[1], 0, true_coarse[3], true_coarse[4], true_coarse[5]],
                "aquatic_specialist": [true_fine[0], true_fine[1], true_fine[0], true_fine[0], true_fine[4], true_fine[0]],
                "tree_specialist": [FINE_NAME_TO_INDEX["maple_tree"]] * len(fine_names),
                "global_small": [true_fine[0], true_fine[1], true_fine[2], 0, 0, true_fine[5]],
                "global_large": [true_fine[0], 0, true_fine[2], true_fine[3], true_fine[4], true_fine[5]],
            }
            confidence = [0.99, 0.92, 0.81, 0.74, 0.61, 0.52]
            outcome_rows = []
            for candidate_id, _, _, _ in candidate_definitions:
                for sample_id, prediction in enumerate(predictions[candidate_id]):
                    outcome_rows.append(
                        {
                            "sample_id": sample_id,
                            "candidate_id": candidate_id,
                            "prediction": prediction,
                            "confidence": confidence[sample_id],
                            "accepted": confidence[sample_id] >= 0.5,
                        }
                    )
            outcomes = pd.DataFrame(outcome_rows)
            bundle_path = root / "empirical.pkl"
            pd.to_pickle(
                {
                    "labels": labels,
                    "candidates": candidates,
                    "outcomes": outcomes,
                },
                bundle_path,
            )

            manifest_candidates = []
            for candidate_id, role, kind, group in candidate_definitions:
                if kind == "identifier":
                    classes = 20
                    output_labels = list(COARSE_LABEL_NAMES)
                elif kind == "specialized":
                    classes = 5
                    output_labels = list(COARSE_TO_FINE_NAMES[str(group)])
                else:
                    classes = 100
                    output_labels = list(FINE_LABEL_NAMES)
                local_predictions = [index % classes for index in range(len(labels))]
                artifact_path = root / f"{candidate_id}.npz"
                probability_values = _probabilities(local_predictions, classes)
                np.savez_compressed(
                    artifact_path,
                    logits=np.log(probability_values),
                    probabilities=probability_values,
                    sample_id=labels["sample_id"].to_numpy(),
                    output_labels=np.asarray(output_labels),
                )
                metrics_dir = root / "training" / candidate_id
                metrics_dir.mkdir(parents=True)
                (metrics_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "cifar100-training-metrics/v1",
                            "best_model_selection_accuracy": 0.75,
                            "parameter_count": 123,
                            "input_resolution": [3, 32, 32],
                        }
                    ),
                    encoding="utf-8",
                )
                metrics_path = metrics_dir / "metrics.json"
                manifest_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "role": role,
                        "kind": kind,
                        "group": group,
                        "architecture": "synthetic",
                        "output_labels": output_labels,
                        "checkpoint": {
                            "path": str(metrics_dir / "best.pt"),
                            "sha256": "synthetic-checksum",
                            "config_hash": "synthetic-config",
                        },
                        "parameter_count": 123,
                        "training_metrics": {
                            "path": str(metrics_path),
                            "sha256": file_sha256(metrics_path),
                        },
                        "probability_artifact": {
                            "path": artifact_path.name,
                            "sha256": file_sha256(artifact_path),
                            "dtype": "float32",
                            "logits_shape": [len(labels), classes],
                            "probabilities_shape": [len(labels), classes],
                            "output_labels": output_labels,
                        },
                    }
                )
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "cifar100-empirical-manifest/v1",
                        "dataset_id": CIFAR100_PROFILE.dataset_id,
                        "profile_fingerprint": CIFAR100_PROFILE.fingerprint,
                        "official_test_used": False,
                        "detector_status": "external_pending",
                        "outcomes_path": bundle_path.name,
                        "outcomes_sha256": file_sha256(bundle_path),
                        "candidates": manifest_candidates,
                    }
                ),
                encoding="utf-8",
            )
            latency_path = root / "latency.json"
            latency_path.write_text(
                json.dumps(
                    {
                        "schema_version": LATENCY_SCHEMA_VERSION,
                        "candidates": [
                            {
                                "candidate_id": candidate_id,
                                "checkpoint_sha256": "synthetic-checksum",
                                "config_hash": "synthetic-config",
                                "latency_ms": {
                                    "mean": float(index + 1),
                                    "median": float(index + 1),
                                    "std": 0.1,
                                    "p95": float(index + 1.2),
                                    "p99": float(index + 1.3),
                                },
                            }
                            for index, (candidate_id, _, _, _) in enumerate(
                                candidate_definitions
                            )
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_payload["latency_results"] = {
                "path": str(latency_path),
                "sha256": file_sha256(latency_path),
            }
            manifest_path.write_text(
                json.dumps(manifest_payload), encoding="utf-8"
            )

            output_dir = root / "report"
            report = generate_report(
                manifest_path,
                latency_path,
                output_dir,
                thresholds=(0.5, 0.8, 0.95),
                complementarity_threshold=0.8,
            )

            self.assertEqual(report["schema_version"], REPORT_SCHEMA_VERSION)
            self.assertEqual(len(report["candidates"]), 5)
            by_id = {item["candidate_id"]: item for item in report["candidates"]}
            self.assertIn("routing_precision_vs_coverage", by_id["router"])
            self.assertEqual(by_id["router"]["model_selection_accuracy"], 0.75)
            specialist = by_id["aquatic_specialist"]["specialist_behavior"]
            self.assertEqual(specialist["in_group_samples"], 3)
            self.assertEqual(specialist["out_group_samples"], 3)
            self.assertEqual(specialist["all_sample_selective"][0]["population"], 6)
            empty_specialist = by_id["tree_specialist"]["specialist_behavior"]
            self.assertEqual(empty_specialist["in_group_samples"], 0)
            self.assertIsNone(empty_specialist["in_group_accuracy"])
            self.assertIsNone(empty_specialist["in_group_mean_confidence"])
            self.assertTrue(
                all(
                    metric["coverage"] is None
                    for metric in empty_specialist["in_group_selective"]
                )
            )
            self.assertEqual(
                empty_specialist["all_sample_selective"][0]["population"], 6
            )
            self.assertEqual(len(report["global_complementarity"]), 1)
            pair = report["global_complementarity"][0]
            self.assertGreater(pair["only_a_correct"], 0.0)
            self.assertGreater(pair["only_b_correct"], 0.0)
            for name in (
                "candidate_report.json",
                "candidate_summary.csv",
                "selective_metrics.csv",
                "global_complementarity.csv",
                "dominated_candidates.csv",
                "candidate_report.md",
            ):
                self.assertTrue((output_dir / name).is_file(), name)
            summary = pd.read_csv(output_dir / "candidate_summary.csv")
            self.assertEqual(len(summary), 5)
            self.assertIn("latency_p99_ms", summary.columns)
            self.assertIn("all_sample_coverage_at_0.5", summary.columns)
            selective = pd.read_csv(output_dir / "selective_metrics.csv")
            specialist_populations = set(
                selective.loc[
                    selective["candidate_id"] == "aquatic_specialist",
                    "population_kind",
                ]
            )
            self.assertEqual(
                specialist_populations,
                {"role", "all_samples", "out_group"},
            )

            changed_latency = json.loads(latency_path.read_text(encoding="utf-8"))
            changed_latency["candidates"][0]["latency_ms"]["mean"] += 123.0
            latency_path.write_text(json.dumps(changed_latency), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Latency artifact checksum"):
                generate_report(manifest_path, latency_path, root / "changed-report")


if __name__ == "__main__":
    unittest.main()
