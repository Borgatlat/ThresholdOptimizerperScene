from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.cifar100.collect_empirical_outcomes import (
    LATENCY_SCHEMA_VERSION,
    TRAINING_MANIFEST_SCHEMA_VERSION,
    _latency_by_id,
    _validate_latency_manifest,
    _validate_training_manifest,
    build_candidate_outcomes,
)
from experiments.cifar100.labels import (
    CIFAR100_PROFILE,
    COARSE_TO_FINE_NAMES,
    FINE_NAME_TO_INDEX,
)
from experiments.cifar100.models import CandidateSpec, file_sha256


class CifarEmpiricalMappingTests(unittest.TestCase):
    def test_source_manifests_are_bound_to_exact_split_and_training_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_npz = root / "split.npz"
            split_json = root / "split.json"
            training_path = root / "training.json"
            split_npz.write_bytes(b"indices")
            split_json.write_text("{}\n", encoding="utf-8")
            training_path.write_text('{"run": 1}\n', encoding="utf-8")
            training = {
                "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
                "dataset_id": CIFAR100_PROFILE.dataset_id,
                "profile_fingerprint": CIFAR100_PROFILE.fingerprint,
                "official_test_used": False,
                "split_indices_sha256": file_sha256(split_npz),
                "split_manifest_sha256": file_sha256(split_json),
            }
            latency = {
                "schema_version": LATENCY_SCHEMA_VERSION,
                "source_manifest": {"sha256": file_sha256(training_path)},
                "input": {
                    "source": "official_training_cascade_validation",
                    "split_indices_sha256": file_sha256(split_npz),
                    "split_manifest_sha256": file_sha256(split_json),
                },
            }

            _validate_training_manifest(
                training,
                split_npz=split_npz,
                split_manifest=split_json,
            )
            _validate_latency_manifest(
                latency,
                training_manifest_path=training_path,
                split_npz=split_npz,
                split_manifest=split_json,
            )

            split_npz.write_bytes(b"different indices")
            with self.assertRaisesRegex(ValueError, "does not match"):
                _validate_training_manifest(
                    training,
                    split_npz=split_npz,
                    split_manifest=split_json,
                )

    def test_duplicate_latency_candidate_ids_are_rejected(self) -> None:
        entry = {"candidate_id": "duplicate", "latency_ms": {"mean": 1.0}}
        with self.assertRaisesRegex(ValueError, "Duplicate latency"):
            _latency_by_id({"candidates": [entry, dict(entry)]})

    def test_specialist_outputs_are_mapped_globally_for_all_samples(self) -> None:
        group = "aquatic_mammals"
        labels = tuple(COARSE_TO_FINE_NAMES[group])
        spec = CandidateSpec(
            candidate_id="specialist_test",
            role="specialized",
            kind="specialized",
            architecture="wrn16_2",
            output_labels=labels,
            group=group,
        )
        logits = np.asarray(
            [[8.0, 0.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        true_fine = np.asarray(
            [FINE_NAME_TO_INDEX["beaver"], FINE_NAME_TO_INDEX["aquarium_fish"]]
        )
        true_coarse = np.asarray(
            [
                CIFAR100_PROFILE.router_index["aquatic_mammals"],
                CIFAR100_PROFILE.router_index["fish"],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "specialist.npz"
            frame, artifact = build_candidate_outcomes(
                spec=spec,
                logits=logits,
                true_fine=true_fine,
                true_coarse=true_coarse,
                sample_ids=np.asarray([0, 1]),
                checkpoint_sha256="checkpoint",
                checkpoint_config_hash="config",
                artifact_path=path,
            )
            with np.load(path, allow_pickle=False) as values:
                probabilities = values["probabilities"]
                stored_labels = tuple(values["output_labels"].tolist())

        beaver_index = FINE_NAME_TO_INDEX["beaver"]
        self.assertEqual(frame["predicted_local_label"].tolist(), [0, 0])
        self.assertEqual(frame["predicted_global_label"].tolist(), [beaver_index] * 2)
        self.assertEqual(frame["prediction"].tolist(), [beaver_index] * 2)
        self.assertEqual(frame["in_specialist_group"].tolist(), [True, False])
        self.assertEqual(frame["role_correct"].tolist(), [True, False])
        self.assertTrue(frame["accepted"].all())
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        self.assertEqual(stored_labels, labels)
        self.assertEqual(artifact["probabilities_shape"], [2, 5])
        self.assertEqual(len(artifact["sha256"]), 64)

    def test_intermediate_prediction_remains_in_coarse_space(self) -> None:
        output_labels = tuple(CIFAR100_PROFILE.router_outputs)
        spec = CandidateSpec(
            candidate_id="router_test",
            role="intermediate",
            kind="identifier",
            architecture="wrn16_2",
            output_labels=output_labels,
        )
        logits = np.zeros((2, 20), dtype=np.float32)
        logits[0, 0] = 5.0
        logits[1, 1] = 5.0
        with tempfile.TemporaryDirectory() as directory:
            frame, _ = build_candidate_outcomes(
                spec=spec,
                logits=logits,
                true_fine=np.asarray([4, 1]),
                true_coarse=np.asarray([0, 1]),
                sample_ids=np.asarray([0, 1]),
                checkpoint_sha256="checkpoint",
                checkpoint_config_hash="config",
                artifact_path=Path(directory) / "router.npz",
            )
        self.assertEqual(frame["prediction"].tolist(), [0, 1])
        self.assertEqual(frame["predicted_global_label"].tolist(), [-1, -1])
        self.assertTrue(frame["role_correct"].all())

    def test_detector_predictions_are_global_and_always_accepted(self) -> None:
        spec = CandidateSpec(
            candidate_id="detector_test",
            role="detector",
            kind="detector",
            architecture="convnextv2_large",
            output_labels=tuple(CIFAR100_PROFILE.global_classes),
        )
        logits = np.zeros((2, 100), dtype=np.float32)
        logits[0, 4] = 8.0
        logits[1, 7] = 8.0
        with tempfile.TemporaryDirectory() as directory:
            frame, _ = build_candidate_outcomes(
                spec=spec,
                logits=logits,
                true_fine=np.asarray([4, 7]),
                true_coarse=np.asarray([0, 7]),
                sample_ids=np.asarray([0, 1]),
                checkpoint_sha256="checkpoint",
                checkpoint_config_hash="config",
                artifact_path=Path(directory) / "detector.npz",
            )
        self.assertEqual(frame["prediction"].tolist(), [4, 7])
        self.assertTrue(frame["accepted"].all())
        self.assertTrue(frame["role_correct"].all())


if __name__ == "__main__":
    unittest.main()
