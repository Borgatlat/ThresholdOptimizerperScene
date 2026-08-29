from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.m3n_vc import benchmark_chellapilla_single_vs_best10 as benchmark


TARGET = 0.90


def _method(
    cost: float,
    accuracy: float = 0.95,
    *,
    selected_restart_index: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "expected_cost": float(cost),
        "accuracy": float(accuracy),
        "feasible": bool(accuracy >= TARGET),
        "thresholds": {"slot": float(cost)},
    }
    if selected_restart_index is not None:
        result["selected_restart_index"] = int(selected_restart_index)
    return result


def _layout(index: int) -> dict[str, object]:
    return {
        "layout_index": index,
        "layout_id": f"layout-{index}",
        "layout": {"initial": ["K3", "detector"], "specialized": {}},
        "distinct_non_detector_classifiers": 1,
    }


def _summary_record(
    layout_index: int,
    single_cost: float,
    best_cost: float,
    *,
    single_accuracy: float = 0.95,
    best_accuracy: float = 0.95,
    selected_restart_index: int = 1,
) -> dict[str, object]:
    return {
        "layout_index": layout_index,
        "single": _method(single_cost, single_accuracy),
        "best_of_10": _method(
            best_cost,
            best_accuracy,
            selected_restart_index=selected_restart_index,
        ),
        "group_elapsed_seconds": 1.0 + layout_index,
    }


def _validated_record(
    *, layout_index: int = 0, trial_index: int = 0, settings_sha256: str = "sha"
) -> dict[str, object]:
    metadata = _layout(layout_index)
    base_seed = benchmark.trial_base_seed(layout_index, trial_index)
    seeds = list(range(base_seed, base_seed + benchmark.RESTARTS))
    restarts: list[dict[str, object]] = []
    for restart_index, seed in enumerate(seeds):
        cost = float(10 - restart_index)
        restarts.append(
            {
                "expected_cost": cost,
                "accuracy": 0.95,
                "feasible": True,
                "thresholds": {"slot": cost},
                "method": "chellapilla_continuous_gaussian_sa",
                "elapsed_seconds": 0.1 + restart_index / 100.0,
                "annealing_iterations": 1_000,
                "restart_index": restart_index,
                "restart_seed": seed,
            }
        )
    selected_index = benchmark.RESTARTS - 1
    restart_zero = {
        key: value
        for key, value in restarts[0].items()
        if key not in {"restart_index", "restart_seed"}
    }
    selected = restarts[selected_index]
    best = {
        "expected_cost": selected["expected_cost"],
        "accuracy": selected["accuracy"],
        "feasible": selected["feasible"],
        "thresholds": selected["thresholds"],
        "method": "best_of_10_chellapilla_continuous_gaussian_sa",
        "selected_restart_index": selected_index,
        "selected_restart_seed": seeds[selected_index],
        "restart_count": benchmark.RESTARTS,
        "iterations_per_restart": 1_000,
        "restart_seeds": seeds,
        "restart_costs_ms": [packet["expected_cost"] for packet in restarts],
        "restart_accuracies": [packet["accuracy"] for packet in restarts],
        "restart_feasible": [packet["feasible"] for packet in restarts],
        "restart_elapsed_seconds": [
            packet["elapsed_seconds"] for packet in restarts
        ],
        "group_elapsed_seconds": 1.25,
    }
    return {
        "schema_version": benchmark.RECORD_SCHEMA_VERSION,
        "settings_sha256": settings_sha256,
        "dataset": "m3n_vc/h24",
        "partition": "validation",
        "holdout_usage": "not_evaluated",
        **metadata,
        "trial_index": trial_index,
        "base_seed": base_seed,
        "restart_seeds": seeds,
        "iterations_per_restart": 1_000,
        "restarts": restarts,
        "single": restart_zero,
        "best_of_10": best,
        "group_elapsed_seconds": 1.25,
    }


class ChellapillaSingleVsBestTenTests(unittest.TestCase):
    def test_default_seed_blocks_are_disjoint_and_restart_zero_is_paired(self) -> None:
        blocks = []
        for layout_index in range(benchmark.DEFAULT_LAYOUT_COUNT):
            for trial_index in range(benchmark.DEFAULT_TRIALS_PER_LAYOUT):
                base = benchmark.trial_base_seed(layout_index, trial_index)
                block = tuple(range(base, base + benchmark.RESTARTS))
                blocks.append(block)
                self.assertEqual(block[0], base)
                self.assertEqual(len(set(block)), benchmark.RESTARTS)

        flattened = [seed for block in blocks for seed in block]
        self.assertEqual(len(flattened), 10_000)
        self.assertEqual(len(set(flattened)), len(flattened))
        self.assertEqual(benchmark.trial_base_seed(0, 1), 10)
        self.assertEqual(benchmark.trial_base_seed(1, 0), 1_000_000)

    def test_feasibility_aware_restart_selection_and_stable_ties(self) -> None:
        results = [
            _method(1.0, 0.89),
            _method(100.0, 0.90),
            _method(50.0, 0.91),
        ]
        self.assertEqual(benchmark._best_restart_index(results, TARGET), 2)

        infeasible = [_method(1.0, 0.80), _method(100.0, 0.89)]
        self.assertEqual(benchmark._best_restart_index(infeasible, TARGET), 1)

        tied = [_method(5.0, 0.95), _method(5.0, 0.95)]
        self.assertEqual(benchmark._best_restart_index(tied, TARGET), 0)

    def test_per_layout_and_pooled_mean_median_and_maximum(self) -> None:
        layouts = [_layout(0), _layout(1)]
        records = [
            _summary_record(0, 10.0, 5.0),
            _summary_record(0, 20.0, 15.0, selected_restart_index=0),
            _summary_record(
                1,
                30.0,
                25.0,
                single_accuracy=0.85,
                best_accuracy=0.95,
            ),
        ]

        per_layout, pooled = benchmark.summarize_records(records, layouts, TARGET)

        layout_zero_single = per_layout[0]["methods"]["single"]["cost_ms"]
        layout_zero_best = per_layout[0]["methods"]["best_of_10"]["cost_ms"]
        self.assertEqual(layout_zero_single["mean"], 15.0)
        self.assertEqual(layout_zero_single["median"], 15.0)
        self.assertEqual(layout_zero_single["maximum"], 20.0)
        self.assertEqual(layout_zero_single["highest"], 20.0)
        self.assertEqual(layout_zero_best["mean"], 10.0)
        self.assertEqual(layout_zero_best["median"], 10.0)
        self.assertEqual(layout_zero_best["maximum"], 15.0)

        pooled_single = pooled["methods"]["single"]
        pooled_best = pooled["methods"]["best_of_10"]
        self.assertEqual(pooled_single["cost_ms"]["mean"], 20.0)
        self.assertEqual(pooled_single["cost_ms"]["median"], 20.0)
        self.assertEqual(pooled_single["cost_ms"]["maximum"], 30.0)
        self.assertEqual(pooled_best["cost_ms"]["mean"], 15.0)
        self.assertEqual(pooled_best["cost_ms"]["median"], 15.0)
        self.assertEqual(pooled_best["cost_ms"]["maximum"], 25.0)
        self.assertEqual(pooled_single["feasible_count"], 2)
        self.assertEqual(pooled_best["feasible_count"], 3)
        self.assertEqual(
            pooled["paired"]["comparison"],
            {"best_of_10_better": 3, "tie": 0, "single_better": 0},
        )

    def test_record_validation_checks_pairing_provenance_and_resume_keys(self) -> None:
        layouts = [_layout(0)]
        record = _validated_record()

        completed = benchmark._validate_records(
            [record],
            settings_sha256="sha",
            layouts=layouts,
            trials_per_layout=2,
            target_accuracy=TARGET,
            iterations=1_000,
        )
        self.assertEqual(completed, {(0, 0)})

        with self.assertRaisesRegex(ValueError, "Duplicate completed trial"):
            benchmark._validate_records(
                [record, deepcopy(record)],
                settings_sha256="sha",
                layouts=layouts,
                trials_per_layout=2,
                target_accuracy=TARGET,
                iterations=1_000,
            )

        wrong_hash = deepcopy(record)
        wrong_hash["settings_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "another experiment"):
            benchmark._validate_records(
                [wrong_hash],
                settings_sha256="sha",
                layouts=layouts,
                trials_per_layout=2,
                target_accuracy=TARGET,
                iterations=1_000,
            )

        wrong_seed = deepcopy(record)
        wrong_seed["restarts"][4]["restart_seed"] += 1
        with self.assertRaisesRegex(ValueError, "restart seed metadata"):
            benchmark._validate_records(
                [wrong_seed],
                settings_sha256="sha",
                layouts=layouts,
                trials_per_layout=2,
                target_accuracy=TARGET,
                iterations=1_000,
            )

        wrong_single = deepcopy(record)
        wrong_single["single"]["thresholds"] = {"slot": -1.0}
        with self.assertRaisesRegex(ValueError, "not reused from restart zero"):
            benchmark._validate_records(
                [wrong_single],
                settings_sha256="sha",
                layouts=layouts,
                trials_per_layout=2,
                target_accuracy=TARGET,
                iterations=1_000,
            )

        wrong_partition = deepcopy(record)
        wrong_partition["partition"] = "holdout"
        with self.assertRaisesRegex(ValueError, "not a validation record"):
            benchmark._validate_records(
                [wrong_partition],
                settings_sha256="sha",
                layouts=layouts,
                trials_per_layout=2,
                target_accuracy=TARGET,
                iterations=1_000,
            )

        holdout_leak = deepcopy(record)
        holdout_leak["holdout"] = {"expected_cost": 1.0, "accuracy": 1.0}
        with self.assertRaisesRegex(ValueError, "validation-only contract"):
            benchmark._validate_records(
                [holdout_leak],
                settings_sha256="sha",
                layouts=layouts,
                trials_per_layout=2,
                target_accuracy=TARGET,
                iterations=1_000,
            )

    def test_tiny_real_benchmark_and_noop_resume(self) -> None:
        outcomes = Path("checkpoints/empirical_outcomes.pkl")
        if not outcomes.is_file():
            self.skipTest("The h24 empirical outcomes are not available.")
        self.assertEqual(benchmark.DEFAULT_TRIALS_PER_LAYOUT, 100)
        self.assertEqual(benchmark.DEFAULT_ITERATIONS, 1_000)
        self.assertEqual(benchmark.DEFAULT_WORKERS, 16)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            arguments = {
                "outcomes": outcomes,
                "output_dir": output_dir,
                "target_accuracy": 0.9662,
                "layout_count": 1,
                "trials_per_layout": 1,
                "iterations": 2,
                "layout_seed": 20260818,
                "minimum_classifiers": 5,
                "workers": 1,
            }
            summary = benchmark.run_benchmark(**arguments)
            records_path = output_dir / "trial_packets.jsonl"
            before = records_path.read_bytes()
            records = [json.loads(line) for line in before.splitlines()]

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["completed_trials"], 1)
            self.assertEqual(summary["completed_trajectories"], 10)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["restart_seeds"], list(range(10)))
            self.assertEqual(len(records[0]["restarts"]), 10)
            restart_zero = {
                key: value
                for key, value in records[0]["restarts"][0].items()
                if key not in {"restart_index", "restart_seed"}
            }
            self.assertEqual(records[0]["single"], restart_zero)

            resumed = benchmark.run_benchmark(**arguments)
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["completed_trials"], 1)
            self.assertEqual(records_path.read_bytes(), before)

            with self.assertRaisesRegex(ValueError, "different settings"):
                benchmark.run_benchmark(**{**arguments, "iterations": 3})


if __name__ == "__main__":
    unittest.main()
