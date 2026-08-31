from types import SimpleNamespace
import unittest

import pandas as pd

from hierarchy_optimizer import Cascade
from threshold_optimizer import (
    FixedLayoutThresholdEvaluator,
    enumerate_threshold_slots,
    optimize_fixed_layout_thresholds_exhaustive,
    optimize_fixed_layout_thresholds_chellapilla_sa,
    optimize_fixed_layout_thresholds_legacy_grid_sa,
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
        global_class_names=("gle350", "other"),
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
    def test_threshold_coordinates_ignore_specialized_dict_insertion_order(self) -> None:
        slots = enumerate_threshold_slots(
            ["K0", "detector"],
            {
                ("K0", "suv"): ["K4", "detector"],
                ("K0", "coupe"): ["K6", "detector"],
            },
        )

        self.assertEqual(
            [slot.location for slot in slots],
            [
                "initial[0]",
                "specialized[K0:coupe][0]",
                "specialized[K0:suv][0]",
            ],
        )

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

    def test_route_counts_distinguish_repeated_model_occurrences(self) -> None:
        metrics = repeated_model_evaluator().evaluate(
            {
                "K0": 0.5,
                "K2@initial[1]": 0.65,
                "K2@specialized[K0:suv][0]": 0.85,
            }
        )

        self.assertEqual(
            metrics["route_counts"],
            {
                "K2@initial[1]": 1,
                "K2@specialized[K0:suv][0]": 1,
                "detector": 2,
            },
        )
        self.assertEqual(sum(metrics["route_counts"].values()), metrics["total"])

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
        result = optimize_fixed_layout_thresholds_legacy_grid_sa(
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

    def test_annealer_can_skip_coordinate_descent(self) -> None:
        evaluator = repeated_model_evaluator()
        result = optimize_fixed_layout_thresholds_legacy_grid_sa(
            evaluator,
            target_accuracy=1.0,
            grids={"K0": [0.5], "K2": [0.65, 0.85]},
            n_iterations=1,
            random_seed=0,
            coordinate_descent_passes=0,
            random_proposal_rate=0.0,
            show_progress=False,
        )

        self.assertEqual(result["method"], "simulated_annealing")
        self.assertEqual(result["coordinate_descent_passes"], 0)
        self.assertEqual(result["coordinate_descent_evaluations"], 0)
        self.assertEqual(result["evaluations"], result["annealing_evaluations"])
        self.assertEqual(result["random_proposal_rate"], 0.0)

    def test_annealer_rejects_negative_coordinate_descent_passes(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            optimize_fixed_layout_thresholds_legacy_grid_sa(
                repeated_model_evaluator(),
                coordinate_descent_passes=-1,
                show_progress=False,
            )

    def test_annealer_rejects_invalid_random_proposal_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            optimize_fixed_layout_thresholds_legacy_grid_sa(
                repeated_model_evaluator(),
                random_proposal_rate=1.1,
                show_progress=False,
            )

    def test_chellapilla_sa_is_continuous_deterministic_and_unpolished(self) -> None:
        first = optimize_fixed_layout_thresholds_chellapilla_sa(
            repeated_model_evaluator(),
            target_accuracy=1.0,
            n_iterations=25,
            random_seed=7,
            show_progress=False,
        )
        second = optimize_fixed_layout_thresholds_chellapilla_sa(
            repeated_model_evaluator(),
            target_accuracy=1.0,
            n_iterations=25,
            random_seed=7,
            show_progress=False,
        )

        self.assertEqual(first["method"], "chellapilla_continuous_gaussian_sa")
        self.assertEqual(first["thresholds"], second["thresholds"])
        self.assertEqual(first["expected_cost"], second["expected_cost"])
        self.assertNotIn("coordinate_descent_evaluations", first)
        self.assertEqual(first["proposal"], "all_thresholds_continuous_gaussian")
        self.assertTrue(all(0.0 <= value <= 1.0 for value in first["thresholds"].values()))

    def test_canonical_annealer_selects_best_paper_restart(self) -> None:
        evaluator = repeated_model_evaluator()
        independent = [
            optimize_fixed_layout_thresholds_chellapilla_sa(
                evaluator,
                target_accuracy=1.0,
                n_iterations=20,
                random_seed=seed,
                show_progress=False,
            )
            for seed in range(3)
        ]
        result = optimize_fixed_layout_thresholds_simulated_annealing(
            evaluator,
            target_accuracy=1.0,
            n_iterations=20,
            random_seed=0,
            restarts=3,
            show_progress=False,
        )

        self.assertEqual(
            result["expected_cost"],
            min(item["expected_cost"] for item in independent),
        )
        self.assertEqual(result["restart_count"], 3)
        self.assertEqual(result["iterations_per_restart"], 20)
        self.assertEqual(
            result["method"], "best_of_3_chellapilla_continuous_gaussian_sa"
        )

    def test_canonical_annealer_requires_a_restart(self) -> None:
        with self.assertRaisesRegex(ValueError, "restarts must be at least 1"):
            optimize_fixed_layout_thresholds_simulated_annealing(
                repeated_model_evaluator(), restarts=0, show_progress=False
            )

    def test_paper_pruning_removes_reject_all_stages_and_their_cost(self) -> None:
        evaluator = repeated_model_evaluator()
        thresholds = {slot_id: 1.0 for slot_id in evaluator.tunable_ids}

        charged = evaluator.evaluate(thresholds, strict_thresholds=True)
        pruned = evaluator.evaluate(
            thresholds,
            prune_reject_all_stages=True,
            strict_thresholds=True,
        )

        self.assertEqual(charged["expected_cost"], 13.0)
        self.assertEqual(pruned["expected_cost"], 10.0)
        self.assertEqual(pruned["active_slots"], [])
        self.assertEqual(pruned["pruned_slots"], list(evaluator.tunable_ids))

    def test_validation_active_slots_are_frozen_for_holdout_replay(self) -> None:
        evaluator = repeated_model_evaluator()
        validation = evaluator.evaluate(
            {
                "K0": 0.5,
                "K2@initial[1]": 1.0,
                "K2@specialized[K0:suv][0]": 1.0,
            },
            prune_reject_all_stages=True,
            strict_thresholds=True,
        )
        replay = evaluator.evaluate(
            {
                "K0": 0.5,
                "K2@initial[1]": 0.0,
                "K2@specialized[K0:suv][0]": 0.0,
            },
            strict_thresholds=True,
            active_slots=validation["active_slots"],
        )

        self.assertEqual(validation["active_slots"], ["K0"])
        self.assertEqual(replay["active_slots"], ["K0"])
        self.assertEqual(replay["expected_cost"], 11.0)

    def test_frozen_active_slots_reject_unknown_occurrences(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fixed-layout occurrences"):
            repeated_model_evaluator().evaluate(active_slots=["not-a-slot"])

    def test_frozen_but_unreached_slot_remains_in_deployed_policy(self) -> None:
        evaluator = repeated_model_evaluator()
        specialized_slot = "K2@specialized[K0:suv][0]"

        replay = evaluator.evaluate(
            strict_thresholds=True,
            active_slots=[specialized_slot],
        )

        self.assertEqual(replay["active_slots"], [specialized_slot])
        self.assertNotIn(specialized_slot, replay["pruned_slots"])


if __name__ == "__main__":
    unittest.main()
