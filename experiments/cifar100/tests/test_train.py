from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.cifar100.labels import CIFAR100_PROFILE
from experiments.cifar100.models import CandidateSpec, candidate_specs
from experiments.cifar100.train import write_training_manifest


def _record(spec: CandidateSpec) -> dict[str, object]:
    return {
        "schema_version": "cifar100-training-metrics/v1",
        "candidate": spec.as_dict(),
        "candidate_id": spec.candidate_id,
        "config_hash": f"config-{spec.candidate_id}",
        "checkpoint_path": f"{spec.candidate_id}/best.pt",
        "checkpoint_sha256": f"sha-{spec.candidate_id}",
        "metrics_path": f"{spec.candidate_id}/metrics.json",
    }


class CifarTrainingManifestTests(unittest.TestCase):
    def test_sequential_partial_jobs_merge_in_registry_order(self) -> None:
        registry = candidate_specs(
            CIFAR100_PROFILE.groups, CIFAR100_PROFILE.global_classes
        )
        later = registry.candidates[-1]
        earlier = registry.candidates[0]
        config = {"schema_version": "cifar100-training/v1", "seed": 2025}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_npz = root / "split.npz"
            split_manifest = root / "split.json"
            split_npz.write_bytes(b"split indices")
            split_manifest.write_text("{}\n", encoding="utf-8")

            write_training_manifest(
                [_record(later)],
                config=config,
                split_npz=split_npz,
                split_manifest=split_manifest,
                output_dir=root / "training",
                device="job one",
            )
            manifest = write_training_manifest(
                [_record(earlier)],
                config=config,
                split_npz=split_npz,
                split_manifest=split_manifest,
                output_dir=root / "training",
                device="job two",
            )
            manifest = write_training_manifest(
                [_record(registry.detector)],
                config=config,
                split_npz=split_npz,
                split_manifest=split_manifest,
                output_dir=root / "training",
                device="detector job",
            )

        self.assertEqual(
            [item["candidate_id"] for item in manifest["candidates"]],
            [earlier.candidate_id, later.candidate_id],
        )
        self.assertEqual(manifest["candidate_count"], 2)
        self.assertFalse(manifest["complete_candidate_set"])
        self.assertEqual(
            manifest["detector"]["candidate_id"],
            registry.detector.candidate_id,
        )
        self.assertEqual(len(manifest["split_indices_sha256"]), 64)
        self.assertEqual(len(manifest["split_manifest_sha256"]), 64)

    def test_partial_merge_rejects_another_split(self) -> None:
        registry = candidate_specs(
            CIFAR100_PROFILE.groups, CIFAR100_PROFILE.global_classes
        )
        config = {"schema_version": "cifar100-training/v1", "seed": 2025}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_npz = root / "split.npz"
            split_manifest = root / "split.json"
            split_npz.write_bytes(b"first")
            split_manifest.write_text("{}\n", encoding="utf-8")
            write_training_manifest(
                [_record(registry.candidates[0])],
                config=config,
                split_npz=split_npz,
                split_manifest=split_manifest,
                output_dir=root / "training",
                device="job one",
            )
            split_npz.write_bytes(b"second")
            with self.assertRaisesRegex(ValueError, "another split index"):
                write_training_manifest(
                    [_record(registry.candidates[1])],
                    config=config,
                    split_npz=split_npz,
                    split_manifest=split_manifest,
                    output_dir=root / "training",
                    device="job two",
                )


if __name__ == "__main__":
    unittest.main()
