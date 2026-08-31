from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.m3n_vc import benchmark_ga_vs_random_sampling_with_k1 as benchmark


def _threshold_settings() -> dict[str, object]:
    return {
        "method": "best_of_10_chellapilla_continuous_gaussian_sa",
        "iterations_per_restart": 1_000,
        "restarts": 10,
        "restart_seeds": list(range(10)),
        "continuous_thresholds": True,
        "quantile_points_used": False,
        "prune_stages_accepting_zero_validation_samples": True,
        "freeze_validation_active_slots_on_holdout": True,
    }


def _common_settings(outcomes_sha256: str) -> dict[str, object]:
    return {
        "dataset": "m3n_vc/h24",
        "target_accuracy": 0.90,
        "removed_candidates": [],
        "layout_grammar": "depth_one_K0_K1",
        "layout_space_size": benchmark.LAYOUT_SPACE_SIZE,
        "detector_mode": "paper",
        "detector_cost_ms": 10_000.0,
        "outcomes_sha256": outcomes_sha256,
        "split_seed": 0,
        "split_strategy": "blocked_per_run",
        "holdout_fraction": 0.20,
        "inner_seed": 0,
        "quantile_points_compatibility_argument": 50,
        "threshold_optimizer": _threshold_settings(),
    }


def _ga_settings(seed: int, outcomes_sha256: str) -> dict[str, object]:
    return {
        **_common_settings(outcomes_sha256),
        "algorithm": "dynamic_constrained_memetic_genetic_algorithm",
        "outer_seed": seed,
        "random_seed": seed,
        "population_size": 32,
        "generations": 24,
        "evaluation_budget": 512,
        "elite_count": 4,
        "tournament_size": 2,
        "crossover_rate": 0.8,
        "mutation_rate": 0.8,
        "random_immigrant_rate": 0.2,
        "component_resample_rate": 0.3,
        "allow_cached_reentry": False,
        "fitness_implementation_sha256": benchmark._ga_implementation_sha256(),
    }


def _random_settings(
    seed: int, outcomes_sha256: str, budget: float
) -> dict[str, object]:
    return {
        **_common_settings(outcomes_sha256),
        "schema_version": "random-joint-layout-search-with-k1/v1",
        "algorithm": "uniform_random_implicit_layout_sampling_without_replacement",
        "sampling_seed": seed,
        "time_budget_seconds": budget,
        "max_layouts": None,
        "fitness_implementation_sha256": benchmark._random_implementation_sha256(),
        "layout_sampler": {
            "schema_version": "implicit-depth-one-layout-sampler/v1",
            "selection": "exact_uniform_without_replacement",
            "random_generator": "numpy.default_rng",
            "deduplication_key": "canonical_space_index",
            "materializes_full_space": False,
        },
    }


def _split() -> dict[str, object]:
    return {
        "strategy": "blocked_per_run",
        "random_seed": 0,
        "holdout_fraction": 0.20,
        "validation_samples": 80,
        "holdout_samples": 20,
        "per_run": {"run": {"validation": 80, "holdout": 20}},
    }


def _metrics(cost: float, accuracy: float = 0.90) -> dict[str, object]:
    return {
        "expected_cost": cost,
        "accuracy": accuracy,
        "feasible": accuracy >= 0.90,
        "thresholds": {"K3": 0.5},
        "active_slots": ["K3"],
        "pruned_slots": [],
        "route_counts": {"K3": 80},
    }


def _winner(
    layout_id: str, layout_index: int, cost: float, holdout_cost: float
) -> dict[str, object]:
    return {
        "layout_id": layout_id,
        "layout_index": layout_index,
        "layout": {"initial": ["K3", "detector"], "specialized": {}},
        "validation": _metrics(cost),
        "holdout": _metrics(holdout_cost, 0.91),
    }


class K1PairedBenchmarkTests(unittest.TestCase):
    def test_dry_run_freezes_the_requested_contract(self) -> None:
        report = benchmark.dry_run_report(
            output_dir=Path("unused"), parallel_trials=10
        )

        self.assertEqual(report["trial_seeds"], list(range(10)))
        self.assertEqual(report["conditions"]["target_accuracy"], 0.90)
        self.assertEqual(
            report["conditions"]["layout_space_size"], 11_589_085
        )
        self.assertLess(
            report["idealized_parallel_lower_bound_hours"],
            report["estimated_sequential_hours"],
        )

    def test_contract_discloses_both_sampling_asymmetries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outcomes = Path(temporary) / "outcomes.pkl"
            outcomes.write_bytes(b"fixture")
            contract = benchmark.experiment_contract(outcomes)

        limitation = str(contract["search_asymmetry"])
        self.assertIn("mandatory K3->detector", limitation)
        self.assertIn("overweights shorter layouts", limitation)
        self.assertIn("No exhaustive K1-enabled oracle", contract["oracle_limitation"])

    def test_summary_validators_enforce_seeds_budget_and_current_sources(self) -> None:
        outcomes_sha256 = "abc123"
        ga_summary = {
            "settings": _ga_settings(3, outcomes_sha256),
            "layout_space_size": benchmark.LAYOUT_SPACE_SIZE,
            "unique_layouts_evaluated": 512,
            "elapsed_seconds_this_invocation": 10.0,
            "split": _split(),
            "winner": _winner("ga", 0, 5.0, 5.2),
            "holdout_usage": "winner_only_after_validation_search",
        }
        random_summary = {
            "settings": _random_settings(3, outcomes_sha256, 10.0),
            "unique_layouts_evaluated": 2,
            "stop_reason": "time_budget_reached",
            "search_elapsed_seconds": 10.25,
            "time_budget_overshoot_seconds": 0.25,
            "split": _split(),
            "winner": _winner("random", 1, 4.0, 4.3),
            "holdout_usage": "winner_only_after_validation_search",
        }

        benchmark.validate_ga_summary(
            ga_summary, seed=3, outcomes_sha256=outcomes_sha256
        )
        benchmark.validate_random_summary(
            random_summary,
            seed=3,
            ga_elapsed_seconds=10.0,
            outcomes_sha256=outcomes_sha256,
        )
        random_summary["settings"]["sampling_seed"] = 4
        with self.assertRaisesRegex(ValueError, "sampling_seed"):
            benchmark.validate_random_summary(
                random_summary,
                seed=3,
                ga_elapsed_seconds=10.0,
                outcomes_sha256=outcomes_sha256,
            )

    def test_deadline_censor_excludes_atomic_overtime_layout(self) -> None:
        records = [
            {
                "layout_id": "before",
                "layout_index": 0,
                "validation": _metrics(6.0),
                "search_elapsed_seconds_at_completion": 9.0,
            },
            {
                "layout_id": "after",
                "layout_index": 1,
                "validation": _metrics(4.0),
                "search_elapsed_seconds_at_completion": 10.1,
            },
        ]

        winner, included, excluded = benchmark._deadline_censored_random_winner(
            records, 10.0
        )

        self.assertEqual(winner["layout_id"], "before")
        self.assertEqual((included, excluded), (1, 1))

    def test_trial_packet_uses_validation_only_and_keeps_both_sources(self) -> None:
        outcomes_sha256 = "fixture-sha"
        with tempfile.TemporaryDirectory() as temporary:
            trial_dir = Path(temporary) / "trial_00"
            ga_dir = trial_dir / "ga"
            random_dir = trial_dir / "random"
            ga_dir.mkdir(parents=True)
            random_dir.mkdir(parents=True)
            ga_settings = _ga_settings(0, outcomes_sha256)
            random_settings = _random_settings(0, outcomes_sha256, 10.0)

            ga_records = []
            for index in range(512):
                ga_records.append(
                    {
                        "layout_id": f"ga-{index}",
                        "layout_index": index,
                        "layout": {
                            "initial": ["K3", "detector"],
                            "specialized": {},
                        },
                        "settings": ga_settings,
                        "validation": _metrics(5.0 + index),
                    }
                )
            ga_evaluations = ga_dir / "evaluations.jsonl"
            ga_evaluations.write_text(
                "".join(json.dumps(record) + "\n" for record in ga_records),
                encoding="utf-8",
            )
            ga_summary = {
                "settings": ga_settings,
                "layout_space_size": benchmark.LAYOUT_SPACE_SIZE,
                "unique_layouts_evaluated": 512,
                "elapsed_seconds_this_invocation": 10.0,
                "split": _split(),
                "winner": _winner("ga-0", 0, 5.0, 5.2),
                "holdout_usage": "winner_only_after_validation_search",
            }
            ga_summary_path = ga_dir / "summary.json"
            ga_summary_path.write_text(json.dumps(ga_summary), encoding="utf-8")

            random_records = [
                {
                    "layout_id": "random-0",
                    "layout_index": 0,
                    "sample_rank": 0,
                    "space_index": 100,
                    "layout": {
                        "initial": ["K1", "detector"],
                        "specialized": {
                            "K1:coupe": ["detector"],
                            "K1:suv": ["detector"],
                        },
                    },
                    "settings": random_settings,
                    "validation": _metrics(6.0),
                    "search_elapsed_seconds_at_completion": 9.0,
                },
                {
                    "layout_id": "random-1",
                    "layout_index": 1,
                    "sample_rank": 1,
                    "space_index": 200,
                    "layout": {
                        "initial": ["K0", "K1", "detector"],
                        "specialized": {},
                    },
                    "settings": random_settings,
                    "validation": _metrics(4.0),
                    "search_elapsed_seconds_at_completion": 10.25,
                },
            ]
            random_evaluations = random_dir / "evaluations.jsonl"
            random_evaluations.write_text(
                "".join(json.dumps(record) + "\n" for record in random_records),
                encoding="utf-8",
            )
            random_summary = {
                "settings": random_settings,
                "unique_layouts_evaluated": 2,
                "stop_reason": "time_budget_reached",
                "search_elapsed_seconds": 10.25,
                "time_budget_overshoot_seconds": 0.25,
                "split": _split(),
                "evaluations": str(random_evaluations.resolve()),
                "evaluations_sha256": benchmark._file_sha256(random_evaluations),
                "winner": _winner("random-1", 1, 4.0, 4.3),
                "holdout_usage": "winner_only_after_validation_search",
            }
            random_summary_path = random_dir / "summary.json"
            random_summary_path.write_text(json.dumps(random_summary), encoding="utf-8")
            contract = {
                "outcomes_sha256": outcomes_sha256,
                "search_asymmetry": "seed and representation asymmetries",
                "oracle_limitation": "no exhaustive oracle",
            }

            packet = benchmark._trial_summary(
                seed=0,
                contract=contract,
                ga_summary_path=ga_summary_path,
                random_summary_path=random_summary_path,
                trial_dir=trial_dir,
            )

            self.assertEqual(
                packet["primary_validation_comparison"]["selection_outcome"],
                "random_better",
            )
            self.assertEqual(
                packet["deadline_censored_random_validation_secondary"][
                    "selection_outcome"
                ],
                "ga_better",
            )
            self.assertTrue(packet["holdout_descriptive"]["not_used_for_selection"])
            self.assertTrue((trial_dir / "summary.json").is_file())

    def test_sign_test_and_ties(self) -> None:
        self.assertEqual(benchmark.exact_two_sided_sign_test_p_value(10, 0), 2 / 1024)
        self.assertEqual(benchmark.exact_two_sided_sign_test_p_value(0, 0), 1.0)
        self.assertEqual(
            benchmark.paired_outcome(_metrics(5.0), _metrics(5.0)), "tie"
        )

    def test_partial_ga_archive_is_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            method_dir = Path(temporary) / "trial" / "ga"
            method_dir.mkdir(parents=True)
            (method_dir / "evaluations.jsonl").write_text("partial", encoding="utf-8")

            archived = benchmark._archive_partial_ga(method_dir)

            self.assertFalse(method_dir.exists())
            self.assertEqual(
                (archived / "evaluations.jsonl").read_text(encoding="utf-8"),
                "partial",
            )


if __name__ == "__main__":
    unittest.main()
