from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from empirical_outcomes import load_empirical_outcomes, save_empirical_outcomes
from experiments.m3n_vc.checkpoint_paths import file_fingerprint
from experiments.m3n_vc.prepare_jetson_empirical_outcomes import (
    MODEL_IDS,
    prepare_jetson_empirical_outcomes,
)
from experiments.m3n_vc.run_jetson_cascade_experiments import (
    validate_jetson_inputs,
)


ROOT = Path(__file__).resolve().parents[3]


def _small_h24_packet(path: Path) -> dict[str, object]:
    payload = load_empirical_outcomes(
        ROOT / "checkpoints/empirical_outcomes_h24_with_run9.pkl"
    )
    labels = payload["labels"]
    selected_ids = set(
        labels.groupby("run_id", sort=False).head(1)["sample_id"].tolist()
    )
    reduced = dict(payload)
    reduced["labels"] = labels[labels["sample_id"].isin(selected_ids)].copy()
    reduced["candidates"] = payload["candidates"].copy(deep=True)
    reduced["outcomes"] = payload["outcomes"][
        payload["outcomes"]["sample_id"].isin(selected_ids)
    ].copy()
    reduced.pop("collection", None)
    save_empirical_outcomes(reduced, path)
    return reduced


def _jetson_registry(path: Path) -> dict[str, tuple[float, float]]:
    payload = json.loads(
        (ROOT / "checkpoints/classifier_registry.json").read_text(encoding="utf-8")
    )
    timings: dict[str, tuple[float, float]] = {}
    for index, model_id in enumerate(MODEL_IDS):
        runtime_ms = 10.0 + index
        wcet_ms = 20.0 + index
        row = next(item for item in payload["classifiers"] if item["name"] == model_id)
        row["runtime_ms"] = runtime_ms
        row["wcet_ms"] = wcet_ms
        timings[model_id] = (runtime_ms, wcet_ms)
    detector = next(
        item for item in payload["classifiers"] if item["name"] == "Kdet"
    )
    detector["runtime_ms"] = 10_000.0
    detector["wcet_ms"] = 10_000.0
    payload["runtime_profile"] = {
        "model_checkpoints": {
            model_id: file_fingerprint(ROOT / f"checkpoints/{model_id}.pt")
            for model_id in MODEL_IDS
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return timings


class PrepareJetsonEmpiricalOutcomesTests(unittest.TestCase):
    def test_reuses_predictions_and_injects_jetson_costs_and_oracle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.pkl"
            registry_path = root / "registry.json"
            output_path = root / "jetson.pkl"
            source = _small_h24_packet(source_path)
            timings = _jetson_registry(registry_path)

            prepared = prepare_jetson_empirical_outcomes(
                source_path,
                registry_path=registry_path,
                checkpoint_dir=ROOT / "checkpoints",
                output_path=output_path,
            )
            result = load_empirical_outcomes(output_path)

            indexed = result["candidates"].set_index("id")
            for model_id, (runtime_ms, wcet_ms) in timings.items():
                self.assertEqual(indexed.loc[model_id, "cost"], runtime_ms)
                self.assertEqual(indexed.loc[model_id, "wcet"], wcet_ms)
            self.assertEqual(indexed.loc["Kdet", "cost"], 10_000.0)
            self.assertEqual(indexed.loc["Kdet", "wcet"], 10_000.0)

            source_non_detector = source["outcomes"][
                source["outcomes"]["candidate_id"] != "Kdet"
            ].reset_index(drop=True)
            result_non_detector = result["outcomes"][
                result["outcomes"]["candidate_id"] != "Kdet"
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(result_non_detector, source_non_detector)

            detector = result["outcomes"][
                result["outcomes"]["candidate_id"] == "Kdet"
            ].set_index("sample_id")
            labels = result["labels"].set_index("sample_id")
            label_order = list(result["profile"]["global_classes"])
            expected = labels["true_global_label"].map(
                {label: index for index, label in enumerate(label_order)}
            )
            self.assertTrue(detector["prediction"].equals(expected.astype("int64")))
            self.assertTrue(detector["accepted"].all())
            self.assertTrue((detector["confidence"] == 1.0).all())
            self.assertEqual(result["detector"]["p_correct"], 1.0)
            self.assertIs(result["collection"]["paper_detector"], True)
            self.assertIs(
                result["collection"]["runtime_profile"][
                    "model_inference_performed"
                ],
                False,
            )
            self.assertEqual(len(prepared["labels"]), 5)

            processed = root / "processed"
            processed.mkdir()
            for name in (
                "h24_paired_mic_norm.npy",
                "h24_paired_geo_norm.npy",
                "h24_metadata.parquet",
            ):
                (processed / name).write_bytes(b"test-placeholder")
            validated = validate_jetson_inputs(
                output_path,
                registry_path,
                processed,
                ROOT / "checkpoints",
            )
            self.assertEqual(
                validated["candidate_cost_ms"]["K0"], timings["K0"][0]
            )

    def test_refuses_to_overwrite_an_existing_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.pkl"
            registry_path = root / "registry.json"
            output_path = root / "jetson.pkl"
            _small_h24_packet(source_path)
            _jetson_registry(registry_path)
            output_path.write_bytes(b"keep-me")

            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                prepare_jetson_empirical_outcomes(
                    source_path,
                    registry_path=registry_path,
                    checkpoint_dir=ROOT / "checkpoints",
                    output_path=output_path,
                )
            self.assertEqual(output_path.read_bytes(), b"keep-me")


if __name__ == "__main__":
    unittest.main()
