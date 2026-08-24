from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    InnerAnnealingFitness,
    TopologyGenome,
    _load_jsonl,
    build_layout_catalogue,
    cascade_from_genome,
    compare_with_exhaustive,
    crossover_genomes,
    initial_population,
    mutate_genome,
    next_population,
    outer_ga_parameters,
    repair_genome,
    restart_population,
    run_joint_search,
    topology_selection_key,
)


def _record(layout_id: str, layout_index: int, cost: float, accuracy: float) -> dict:
    return {
        "layout_id": layout_id,
        "layout_index": layout_index,
        "validation": {
            "expected_cost": cost,
            "accuracy": accuracy,
            "feasible": accuracy >= 0.98,
        },
    }


class JointHierarchyGenomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalogue = build_layout_catalogue()

    def test_catalogue_round_trips_all_legal_layouts(self) -> None:
        self.assertEqual(len(self.catalogue.entries), 5_545)
        self.assertEqual(len(self.catalogue.genome_to_id), 5_545)
        for genome, expected_id in self.catalogue.genome_to_id.items():
            self.assertEqual(
                self.catalogue.entry_for_genome(genome).layout_id,
                expected_id,
            )

    def test_repair_enforces_k1_free_router_grammar(self) -> None:
        repaired = repair_genome(
            TopologyGenome(
                ("K3", "K1", "K3", "K0", "K2"),
                ("K3", "K5", "K5", "K1", "K2"),
                ("K3", "K4", "K1", "K2", "K4"),
            )
        )
        self.assertEqual(repaired.initial, ("K3", "K0", "K2"))
        # K3 precedes K0 and is therefore ineligible on either branch.
        self.assertEqual(repaired.coupe, ("K5", "K2"))
        self.assertEqual(repaired.suv, ("K4", "K2"))
        self.assertNotIn("K1", str(repaired))

        no_router = repair_genome(
            TopologyGenome(("K2",), ("K5",), ("K4",))
        )
        self.assertEqual(no_router, TopologyGenome(("K2",)))

    def test_mutation_and_crossover_always_stay_in_catalogue(self) -> None:
        rng = np.random.default_rng(12)
        entries = self.catalogue.entries
        for _ in range(500):
            first = self.catalogue.genome_for_id(
                entries[int(rng.integers(0, len(entries)))].layout_id
            )
            second = self.catalogue.genome_for_id(
                entries[int(rng.integers(0, len(entries)))].layout_id
            )
            mutated = mutate_genome(first, rng)
            crossed = crossover_genomes(first, second, rng)
            self.assertIn(mutated, self.catalogue.genome_to_id)
            self.assertIn(crossed, self.catalogue.genome_to_id)

    def test_cascade_materializes_coupe_before_suv(self) -> None:
        cascade = cascade_from_genome(
            TopologyGenome(("K0", "K3"), ("K5",), ("K4",))
        )
        self.assertEqual(
            list(cascade.specialized),
            [("K0", "coupe"), ("K0", "suv")],
        )

    def test_initial_population_has_only_apriori_k3_seed_plus_random(self) -> None:
        first = initial_population(
            self.catalogue, 16, np.random.default_rng(4)
        )
        second = initial_population(
            self.catalogue, 16, np.random.default_rng(4)
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        k3_id = self.catalogue.entry_for_genome(
            TopologyGenome(("K3",))
        ).layout_id
        self.assertEqual(first[0], k3_id)

    def test_offspring_are_new_except_for_elites(self) -> None:
        rng = np.random.default_rng(9)
        population = initial_population(self.catalogue, 12, rng)
        records = {
            candidate_id: _record(candidate_id, index, 2_000.0 + index, 0.99)
            for index, candidate_id in enumerate(population)
        }
        children = next_population(
            population,
            records,
            self.catalogue,
            rng,
            target_accuracy=0.98,
            population_size=12,
            elite_count=2,
            tournament_size=2,
            crossover_rate=0.8,
            mutation_rate=0.8,
            random_immigrant_rate=0.2,
            excluded_layout_ids=set(records),
        )
        self.assertEqual(children[:2], population[:2])
        self.assertTrue(set(children[2:]).isdisjoint(records))
        self.assertEqual(len(children), len(set(children)))

    def test_restart_keeps_global_elite_and_uses_unseen_layouts(self) -> None:
        population = initial_population(
            self.catalogue, 10, np.random.default_rng(5)
        )
        records = {
            candidate_id: _record(candidate_id, index, 1_900.0 - index, 0.99)
            for index, candidate_id in enumerate(population)
        }
        restarted = restart_population(
            records,
            self.catalogue,
            np.random.default_rng(6),
            target_accuracy=0.98,
            population_size=10,
        )
        self.assertEqual(restarted[0], population[-1])
        self.assertTrue(set(restarted[1:]).isdisjoint(records))


class OuterScheduleTests(unittest.TestCase):
    def test_annealed_schedule_hits_endpoints_and_clamps_progress(self) -> None:
        start = outer_ga_parameters(-1.0, annealed=True)
        end = outer_ga_parameters(2.0, annealed=True)

        self.assertEqual(start.elite_count, 2)
        self.assertEqual(start.tournament_size, 2)
        self.assertAlmostEqual(start.crossover_rate, 0.60)
        self.assertAlmostEqual(start.mutation_rate, 0.95)
        self.assertAlmostEqual(start.random_immigrant_rate, 0.40)
        self.assertAlmostEqual(start.component_resample_rate, 0.60)

        self.assertEqual(end.elite_count, 6)
        self.assertEqual(end.tournament_size, 4)
        self.assertAlmostEqual(end.crossover_rate, 0.90)
        self.assertAlmostEqual(end.mutation_rate, 0.50)
        self.assertAlmostEqual(end.random_immigrant_rate, 0.05)
        self.assertAlmostEqual(end.component_resample_rate, 0.10)

    def test_annealed_schedule_linearly_interpolates_midpoint(self) -> None:
        middle = outer_ga_parameters(0.5, annealed=True)

        self.assertEqual(middle.elite_count, 4)
        self.assertEqual(middle.tournament_size, 3)
        self.assertAlmostEqual(middle.crossover_rate, 0.75)
        self.assertAlmostEqual(middle.mutation_rate, 0.725)
        self.assertAlmostEqual(middle.random_immigrant_rate, 0.225)
        self.assertAlmostEqual(middle.component_resample_rate, 0.35)

    def test_fixed_schedule_passes_custom_parameters_through(self) -> None:
        fixed = outer_ga_parameters(
            0.73,
            annealed=False,
            elite_count=3,
            tournament_size=5,
            crossover_rate=0.12,
            mutation_rate=0.34,
            random_immigrant_rate=0.56,
            component_resample_rate=0.78,
        )

        self.assertEqual(
            fixed.as_dict(),
            {
                "elite_count": 3,
                "tournament_size": 5,
                "crossover_rate": 0.12,
                "mutation_rate": 0.34,
                "random_immigrant_rate": 0.56,
                "component_resample_rate": 0.78,
            },
        )

    def test_annealed_schedule_rejects_population_smaller_than_max_elites(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "maximum elite count"):
            run_joint_search(
                outcomes=Path("not-loaded.pkl"),
                population_size=6,
                evaluation_budget=6,
                annealed_outer_schedule=True,
            )


class JointHierarchyFitnessTests(unittest.TestCase):
    def test_selection_is_hard_constrained_and_matches_brute_order(self) -> None:
        feasible_fast = _record("fast", 8, 100.0, 0.98)
        feasible_slow = _record("slow", 2, 200.0, 1.00)
        infeasible = _record("miss", 0, 1.0, 0.979)
        target = 0.98
        self.assertLess(
            topology_selection_key(feasible_fast, target),
            topology_selection_key(feasible_slow, target),
        )
        self.assertLess(
            topology_selection_key(feasible_slow, target),
            topology_selection_key(infeasible, target),
        )

    def test_inner_fitness_passes_exact_annealing_contract(self) -> None:
        catalogue = build_layout_catalogue()
        indexed = catalogue.entry_for_genome(TopologyGenome(("K3",)))
        expected_metrics = {
            "accuracy": 0.99,
            "expected_cost": 1_500.0,
            "correct": 99,
            "total": 100,
            "thresholds": {"K3": 0.8},
            "feasible": True,
        }
        with (
            patch(
                "experiments.m3n_vc.joint_optimize_hierarchy_ga."
                "FixedLayoutThresholdEvaluator",
                return_value=object(),
            ) as evaluator,
            patch(
                "experiments.m3n_vc.joint_optimize_hierarchy_ga."
                "optimize_fixed_layout_thresholds_simulated_annealing",
                return_value=expected_metrics,
            ) as anneal,
        ):
            fitness = InnerAnnealingFitness(
                object(),
                target_accuracy=0.9833763718528082,
                quantile_points=50,
                iterations=8_000,
                inner_seed=0,
                settings={"test": True},
            )
            result = fitness(indexed)

        evaluator.assert_called_once()
        anneal.assert_called_once_with(
            evaluator.return_value,
            0.9833763718528082,
            quantile_points=50,
            n_iterations=8_000,
            random_seed=0,
            show_progress=False,
            restarts=10,
        )
        self.assertEqual(result["validation"]["thresholds"], {"K3": 0.8})

    def test_jsonl_resume_repairs_an_interrupted_final_record(self) -> None:
        first = _record("first", 0, 100.0, 0.99)
        second = _record("second", 1, 101.0, 0.99)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.jsonl"
            path.write_bytes(
                (json.dumps(first) + "\n").encode("utf-8")
                + b'{"layout_id":"partial"'
            )
            loaded = _load_jsonl(path)
            self.assertEqual(set(loaded), {"first"})
            self.assertEqual(path.read_text(encoding="utf-8"), json.dumps(first) + "\n")

            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(second) + "\n")
            self.assertEqual(set(_load_jsonl(path)), {"first", "second"})

    def test_jsonl_resume_separates_complete_line_without_newline(self) -> None:
        first = _record("first", 0, 100.0, 0.99)
        second = _record("second", 1, 101.0, 0.99)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.jsonl"
            path.write_text(json.dumps(first), encoding="utf-8")
            self.assertEqual(set(_load_jsonl(path)), {"first"})
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(second) + "\n")
            self.assertEqual(set(_load_jsonl(path)), {"first", "second"})

    def test_partial_reference_is_never_reported_as_an_optimum(self) -> None:
        winner = _record("winner", 0, 100.0, 0.99)
        brute_settings = {
            "detector_mode": "paper",
            "detector_cost_ms": 10_000.0,
            "holdout_fraction": 0.2,
            "iterations": 8_000,
            "outcomes": "outcomes.pkl",
            "quantile_points": 50,
            "removed_candidates": ["K1"],
            "split_strategy": "blocked_per_run",
            "target_accuracy": 0.98,
            "seed": 0,
        }
        reference = dict(winner)
        reference["settings"] = brute_settings
        summary = {
            "best": reference,
            "completed_layouts": 1,
            "expected_total_layouts": 5_545,
        }
        ga_settings = dict(brute_settings)
        ga_settings.pop("seed")
        ga_settings.update({"inner_seed": 0, "split_seed": 0, "outer_seed": 0})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(summary), encoding="utf-8")
            comparison = compare_with_exhaustive(
                winner,
                target_accuracy=0.98,
                settings=ga_settings,
                evaluated_layout_count=1,
                summary_path=path,
                results_path=None,
            )

        self.assertIsNotNone(comparison)
        assert comparison is not None
        self.assertFalse(comparison["comparison_available"])
        self.assertNotIn("validation_cost_regret_ms", comparison)


if __name__ == "__main__":
    unittest.main()
