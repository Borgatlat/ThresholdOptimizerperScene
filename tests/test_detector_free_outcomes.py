from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cascade_profile import HierarchyProfile
from empirical_outcomes import (
    EvaluationSplit,
    PredictionBatch,
    PretrainedClassifier,
    build_empirical_outcomes,
    ensure_empirical_outcomes,
    load_empirical_outcomes,
    validate_empirical_outcomes,
)
from hierarchy_optimizer import Cascade, HierarchyOptimizer


class DetectorFreeOutcomeTests(unittest.TestCase):
    @staticmethod
    def _role_complete_payload() -> dict[str, object]:
        profile = HierarchyProfile(
            dataset_id="validation/test",
            global_classes=("leaf_a", "leaf_b"),
            groups={"group_a": ("leaf_a",), "group_b": ("leaf_b",)},
            router_outputs=("group_a", "group_b"),
        )
        split = EvaluationSplit(
            "validation", None, ("leaf_a", "leaf_b"), sample_ids=(1, 2)
        )
        payload = build_empirical_outcomes(
            profile,
            (split,),
            (
                PretrainedClassifier(
                    "router",
                    "identifier",
                    lambda _: PredictionBatch(
                        predictions=("group_a", "group_b"),
                        confidence=(0.9, 0.9),
                    ),
                    profile.router_outputs,
                    1.0,
                    threshold=0.5,
                ),
                PretrainedClassifier(
                    "specialist",
                    "specialized",
                    lambda _: PredictionBatch(
                        predictions=("leaf_a", "leaf_a"), confidence=(0.9, 0.4)
                    ),
                    profile.groups["group_a"],
                    1.0,
                    threshold=0.5,
                    group="group_a",
                ),
                PretrainedClassifier(
                    "global",
                    "global",
                    lambda _: PredictionBatch(
                        predictions=("leaf_a", "leaf_b"), confidence=(0.9, 0.9)
                    ),
                    profile.global_classes,
                    2.0,
                    threshold=0.5,
                ),
            ),
        )
        payload["candidates"]["role"] = payload["candidates"]["kind"].map(
            {
                "identifier": "intermediate",
                "specialized": "specialized",
                "global": "global",
            }
        )
        validate_empirical_outcomes(payload)
        return payload

    def test_candidate_only_bundle_is_valid_but_cannot_be_optimized(self) -> None:
        profile = HierarchyProfile(
            dataset_id="candidate-only/test",
            global_classes=("leaf_a", "leaf_b"),
            groups={"group_a": ("leaf_a",), "group_b": ("leaf_b",)},
            router_outputs=("group_a", "group_b"),
        )
        split = EvaluationSplit(
            name="validation",
            inputs=None,
            true_labels=("leaf_a", "leaf_b"),
            sample_ids=(11, 22),
        )
        classifier = PretrainedClassifier(
            id="global",
            kind="global",
            predict=lambda _: PredictionBatch(
                predictions=("leaf_a", "leaf_b"), confidence=(0.8, 0.9)
            ),
            output_labels=profile.global_classes,
            expected_cost_ms=1.0,
            threshold=0.0,
            model_fingerprint="checkpoint-sha256",
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcomes.pkl"
            payload = ensure_empirical_outcomes(
                path, profile, (split,), (classifier,)
            )
            loaded = load_empirical_outcomes(path)

        self.assertEqual(payload["detector_status"], "external_pending")
        self.assertIsNone(payload["detector"])
        self.assertEqual(len(loaded["candidates"]), 1)
        with self.assertRaisesRegex(ValueError, "external_pending"):
            HierarchyOptimizer(loaded, detector_mode="paper")

    def test_hierarchy_replay_orders_labels_by_sample_id(self) -> None:
        profile = HierarchyProfile(
            dataset_id="row-order/test",
            global_classes=("leaf_a", "leaf_b"),
            groups={"group_a": ("leaf_a",), "group_b": ("leaf_b",)},
            router_outputs=("group_a", "group_b"),
        )
        split = EvaluationSplit(
            "validation", None, ("leaf_a", "leaf_b"), sample_ids=(1, 2)
        )
        correct = lambda _: PredictionBatch(
            predictions=("leaf_a", "leaf_b"), confidence=(0.9, 0.9)
        )
        payload = build_empirical_outcomes(
            profile,
            (split,),
            (
                PretrainedClassifier(
                    "global",
                    "global",
                    correct,
                    profile.global_classes,
                    1.0,
                    threshold=0.0,
                ),
                PretrainedClassifier(
                    "detector",
                    "detector",
                    correct,
                    profile.global_classes,
                    10.0,
                ),
            ),
        )
        payload["labels"]["sample_id"] = payload["labels"]["sample_id"].map(
            {0: 10, 1: 20}
        )
        payload["outcomes"]["sample_id"] = payload["outcomes"]["sample_id"].map(
            {0: 10, 1: 20}
        )
        payload["labels"] = payload["labels"].iloc[::-1].reset_index(drop=True)
        validate_empirical_outcomes(payload)
        optimizer = HierarchyOptimizer(payload, detector_mode="trained")
        result = optimizer.evaluate_cascade(
            Cascade(0.0, ["global", "detector"], {}, "detector")
        )
        self.assertEqual(result["accuracy"], 1.0)

    def test_candidate_roles_groups_and_output_mappings_are_validated(self) -> None:
        changes = (
            ("kind", "global", "mystery", "unknown kind"),
            ("role", "router", "global", "does not match kind"),
            ("group", "global", "group_a", "Only specialized"),
            ("group", "specialist", "missing", "unknown group"),
            ("output_labels", "router", ["leaf_a"], "label space"),
            ("output_labels", "specialist", ["leaf_b"], "label space"),
        )
        for column, candidate_id, value, message in changes:
            with self.subTest(column=column, candidate_id=candidate_id):
                payload = copy.deepcopy(self._role_complete_payload())
                candidates = payload["candidates"]
                row = candidates.index[candidates["id"] == candidate_id][0]
                candidates.at[row, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_empirical_outcomes(payload)

    def test_confidence_acceptance_and_prediction_values_are_validated(self) -> None:
        payload = self._role_complete_payload()
        cases = (
            ("confidence", "global", np.nan, "finite and within"),
            ("confidence", "global", 1.1, "finite and within"),
            ("accepted", "global", 1, "Boolean"),
            ("accepted", "global", False, "confidence threshold"),
            ("prediction", "router", 2, "shared identifier mapping"),
            ("prediction", "specialist", 1, "shared specialized mapping"),
            ("prediction", "global", 0.5, "integer indices"),
        )
        for column, candidate_id, value, message in cases:
            with self.subTest(column=column, candidate_id=candidate_id, value=value):
                invalid = copy.deepcopy(payload)
                outcomes = invalid["outcomes"]
                if column == "accepted":
                    outcomes[column] = outcomes[column].astype(object)
                elif column == "prediction":
                    outcomes[column] = outcomes[column].astype(float)
                row = outcomes.index[outcomes["candidate_id"] == candidate_id][0]
                outcomes.at[row, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_empirical_outcomes(invalid)

    def test_legacy_m3n_shape_without_output_labels_remains_valid(self) -> None:
        payload = self._role_complete_payload()
        payload["candidates"] = payload["candidates"].drop(
            columns=["output_labels", "role"]
        )
        payload["candidates"]["group"] = payload["candidates"]["group"].where(
            payload["candidates"]["kind"] == "specialized", np.nan
        )
        specialist_rows = payload["outcomes"]["candidate_id"] == "specialist"
        rejected_row = payload["outcomes"].index[
            specialist_rows & ~payload["outcomes"]["accepted"]
        ][0]
        payload["outcomes"].at[rejected_row, "prediction"] = -1

        validate_empirical_outcomes(payload)


if __name__ == "__main__":
    unittest.main()
