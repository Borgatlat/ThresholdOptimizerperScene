from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmark_ga_outer_schedules import (
    _two_sided_sign_test_p_value,
    load_validation_oracle,
)
from joint_optimize_hierarchy_ga import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_ITERATIONS,
    DEFAULT_QUANTILE_POINTS,
    DEFAULT_SEED,
    DEFAULT_SPLIT_STRATEGY,
    FIG1_K3_TARGET_ACCURACY,
    PAPER_DETECTOR_COST_MS,
    REMOVED_CANDIDATES,
)


def _oracle_payload() -> dict[str, object]:
    return {
        "layout_id": "layout",
        "layout_index": 7,
        "settings": {
            "target_accuracy": FIG1_K3_TARGET_ACCURACY,
            "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
            "iterations": DEFAULT_ITERATIONS,
            "quantile_points": DEFAULT_QUANTILE_POINTS,
            "seed": DEFAULT_SEED,
            "split_strategy": DEFAULT_SPLIT_STRATEGY,
            "detector_mode": "paper",
            "detector_cost_ms": PAPER_DETECTOR_COST_MS,
            "removed_candidates": list(REMOVED_CANDIDATES),
        },
        "validation": {"accuracy": 0.99, "expected_cost": 123.0},
        "holdout": {"do_not_copy": True},
    }


class OuterScheduleOracleTests(unittest.TestCase):
    def test_loader_copies_validation_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.jsonl"
            path.write_text(
                json.dumps(_oracle_payload()) + "\n", encoding="utf-8"
            )
            loaded = load_validation_oracle(path)

        self.assertEqual(set(loaded), {"layout"})
        self.assertEqual(
            set(loaded["layout"]),
            {"layout_id", "layout_index", "validation"},
        )
        self.assertNotIn("holdout", loaded["layout"])

    def test_loader_rejects_a_different_inner_contract(self) -> None:
        payload = _oracle_payload()
        settings = payload["settings"]
        assert isinstance(settings, dict)
        settings["iterations"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.jsonl"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires 8000"):
                load_validation_oracle(path)

    def test_exact_sign_test_ignores_ties_outside_its_inputs(self) -> None:
        self.assertEqual(_two_sided_sign_test_p_value(0, 0), 1.0)
        self.assertEqual(_two_sided_sign_test_p_value(3, 0), 0.25)
        self.assertEqual(_two_sided_sign_test_p_value(2, 2), 1.0)


if __name__ == "__main__":
    unittest.main()
