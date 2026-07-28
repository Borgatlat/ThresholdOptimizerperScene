from types import SimpleNamespace
import unittest

import pandas as pd

from hierarchy_optimizer import Cascade
from threshold_optimizer import (
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_exhaustive,
    optimize_fixed_layout_thresholds_simulated_annealing,
)


def repeated_model_evaluator() -> FixedLayoutThresholdEvaluator:
    """Build a small replay where K2 occurs in two different locations."""

    candidates = pd.DataFrame(
        [
            {"id": "K0", "kind": "identifier", "group": None, "cost": 1.0, "threshold": 0.5},
            {"id": "K2", "kind": "global", "group": None, "cost": 2.0, "threshold": 0.85},
        ]
    ).set_index("id", drop=False)
    labels = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "true_global_label": ["gle350"] * 4,
        }
    )
    outcomes = pd.DataFrame(
        [
            # K0 sends samples 0/1 to the SUV branch; samples 2/3 continue
            # down the initial chain.
            *(
                {
                    "candidate_id": "K0",
                    "sample_id": sample_id,
                    "prediction": 0,
                    "confidence": confidence,
                }
                for sample_id, confidence in enumerate([0.9, 0.9, 0.1, 0.1])
            ),
            # K2 is correct except for sample 1. Independent thresholds can
            # accept sample 0 in the branch and sample 2 in the initial chain
            # without accepting the incorrect sample 1.
            *(
                {
                    "candidate_id": "K2",
                    "sample_id": sample_id,
                    "prediction": prediction,
                    "confidence": confidence,
                }
                for sample_id, (prediction, confidence) in enumerate(
                    [(0, 0.9), (1, 0.8), (0, 0.7), (0, 0.6)]
                )
            ),
        ]
    )
    optimizer = SimpleNamespace(
        candidates=candidates,
        labels=labels,
        outcomes=outcomes,
        detector_id="detector",
        detector_mode="paper",
        detector_outcome_id="Kdet",
        detector_cost=10.0,
        identifier_ids=("K0",),
        groups=("suv",),
        _intermediate_idx_to_group={0: "suv", 1: "coupe", 2: "background"},
    )
    cascade = Cascade(
        expected_cost=0.0,
        initial=["K0", "K2", "detector"],
        specialized={("K0", "suv"): ["K2", "detector"]},
        detector="detector",
    )
    return FixedLayoutThresholdEvaluator(optimizer, cascade)


class PerOccurrenceThresholdTests(unittest.TestCase):
    def test_repeated_model_gets_distinct_location_keys(self) -> None:
        evaluator = repeated_model_evaluator()

        self.assertEqual(
            evaluator.tunable_ids,
            (
                "K0",
                "K2@initial[1]",
                "K2@specialized[K0:suv][0]",
            ),
        )
        expanded = evaluator.evaluate(
            {"K0": 0.5, "K2": 0.75},
            include_route_counts=False,
        )["thresholds"]
        self.assertEqual(expanded["K2@initial[1]"], 0.75)
        self.assertEqual(expanded["K2@specialized[K0:suv][0]"], 0.75)

    def test_exhaustive_search_optimizes_occurrences_independently(self) -> None:
        evaluator = repeated_model_evaluator()
        result = optimize_fixed_layout_thresholds_exhaustive(
            evaluator,
            target_accuracy=1.0,
            # A legacy model-level grid is copied to both occurrence
            # coordinates, but the Cartesian search chooses them separately.
            grids={"K0": [0.5], "K2": [0.65, 0.85]},
            max_combinations=4,
        )

        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["thresholds"]["K2@initial[1]"], 0.65)
        self.assertEqual(
            result["thresholds"]["K2@specialized[K0:suv][0]"],
            0.85,
        )

    def test_annealer_and_polish_optimize_occurrences_independently(self) -> None:
        evaluator = repeated_model_evaluator()
        result = optimize_fixed_layout_thresholds_simulated_annealing(
            evaluator,
            target_accuracy=1.0,
            grids={"K0": [0.5], "K2": [0.65, 0.85]},
            n_iterations=1,
            random_seed=0,
            coordinate_descent_passes=3,
        )

        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["thresholds"]["K2@initial[1]"], 0.65)
        self.assertEqual(
            result["thresholds"]["K2@specialized[K0:suv][0]"],
            0.85,
        )


if __name__ == "__main__":
    unittest.main()
