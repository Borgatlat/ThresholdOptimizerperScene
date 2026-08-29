from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.m3n_vc.brute_force_k1_free_layouts import (
    IndexedLayout,
    _cascade_payload,
)
from experiments.m3n_vc.compare_random_ga_exhaustive import (
    SCHEMA_VERSION,
    build_comparison,
    run_comparison,
)
from experiments.m3n_vc.random_joint_optimize_hierarchy import (
    _order_sha256,
    uniform_layout_order,
)
from hierarchy_optimizer import Cascade


TARGET = 0.96
OUTCOMES_SHA256 = "a" * 64
LAYOUT_SPACE_SIZE = 6


def _catalogue() -> tuple[IndexedLayout, ...]:
    initials = (
        ("detector",),
        ("K3", "detector"),
        ("K2", "detector"),
        ("K0", "detector"),
        ("K2", "K3", "detector"),
        ("K3", "K2", "detector"),
    )
    return tuple(
        IndexedLayout(
            index,
            f"layout-{index}",
            Cascade(
                expected_cost=0.0,
                initial=list(initial),
                specialized={},
                detector="detector",
            ),
        )
        for index, initial in enumerate(initials)
    )


def _split() -> dict[str, object]:
    return {
        "strategy": "blocked_per_run",
        "random_seed": 0,
        "holdout_fraction": 0.2,
        "validation_samples": 80,
        "holdout_samples": 20,
        "per_run": {"run1": {"validation": 80, "holdout": 20}},
    }


def _settings(*, layout_size_key: str) -> dict[str, object]:
    return {
        "target_accuracy": TARGET,
        "outcomes": "empirical_outcomes.pkl",
        "outcomes_sha256": OUTCOMES_SHA256,
        "removed_candidates": ["K1"],
        "detector_mode": "paper",
        "detector_cost_ms": 10_000.0,
        "split_strategy": "blocked_per_run",
        "split_seed": 0,
        "holdout_fraction": 0.2,
        layout_size_key: LAYOUT_SPACE_SIZE,
        "threshold_optimizer": {
            "method": "best_of_10_chellapilla_continuous_gaussian_sa",
            "iterations_per_restart": 1_000,
            "restarts": 10,
            "restart_seeds": list(range(10)),
            "prune_stages_accepting_zero_validation_samples": True,
            "freeze_validation_active_slots_on_holdout": True,
        },
    }


def _metrics(cost: float, *, total: int, accuracy: float = TARGET) -> dict[str, object]:
    detector_routes = total // 4
    return {
        "expected_cost": cost,
        "accuracy": accuracy,
        "correct": round(total * accuracy),
        "total": total,
        "thresholds": {"K3@initial[0]": 0.75},
        "route_counts": {
            "K3@initial[0]": total - detector_routes,
            "detector": detector_routes,
        },
    }


def _record(cost: float, index: int) -> dict[str, object]:
    indexed = _catalogue()[index]
    return {
        "layout_id": f"layout-{index}",
        "layout_index": index,
        "layout": _cascade_payload(indexed.cascade),
        "validation": _metrics(cost, total=80),
        "holdout": _metrics(cost + 0.5, total=20, accuracy=0.95),
    }


def _summaries() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    ga = {
        "settings": _settings(layout_size_key="layout_space_size"),
        "split": _split(),
        "elapsed_seconds_this_invocation": 100.0,
        "unique_layouts_evaluated": 2,
        "winner": _record(10.0, 3),
    }
    random_settings = _settings(layout_size_key="layout_space_size")
    random_settings.update(
        {
            "sampling_seed": 0,
            "layout_order_sha256": _order_sha256(
                uniform_layout_order(_catalogue(), 0)
            ),
        }
    )
    random = {
        "settings": random_settings,
        "split": _split(),
        "search_elapsed_seconds": 100.0,
        "unique_layouts_evaluated": 3,
        "winner": _record(10.0, 3),
    }
    exhaustive_record = _record(9.0, 0)
    exhaustive_record.update(
        {
            "best_layout_id": exhaustive_record.pop("layout_id"),
            "best_layout_index": exhaustive_record.pop("layout_index"),
            "completion_seconds": 500.0,
            "completed_layouts": LAYOUT_SPACE_SIZE,
        }
    )
    benchmark = {
        "settings": _settings(layout_size_key="expected_layout_count"),
        "split": _split(),
        "target_accuracy": TARGET,
        "methods": {"exhaustive_joint": exhaustive_record},
    }
    return ga, random, benchmark


def _layout_results() -> list[dict[str, object]]:
    costs = (9.0, 12.0, 11.0, 10.0, 13.0, 14.0)
    results = [
        {
            "layout_id": record["layout_id"],
            "layout_index": record["layout_index"],
            "layout": record["layout"],
            "validation": record["validation"],
        }
        for record in (_record(cost, index) for index, cost in enumerate(costs))
    ]
    # Direct-detector records produced by the exhaustive benchmark have no
    # threshold mapping because there are no tunable stages.
    results[0]["validation"].pop("thresholds")
    return results


class ComparisonPacketTests(unittest.TestCase):
    def test_builds_standard_packet_and_pairwise_regrets(self) -> None:
        ga, random, benchmark = _summaries()
        result = build_comparison(ga, random, benchmark)

        self.assertEqual(result["schema_version"], SCHEMA_VERSION)
        self.assertEqual(
            result["pairwise_validation_cost_ms"],
            {
                "random_minus_ga": 0.0,
                "ga_minus_exhaustive": 1.0,
                "random_minus_exhaustive": 1.0,
            },
        )
        self.assertEqual(
            result["validation_regret_to_exhaustive_ms"]["random_sampling"],
            1.0,
        )
        self.assertEqual(
            result["methods"]["genetic_algorithm"]["validation"]["route_counts"],
            {"K3@initial[0]": 60, "detector": 20},
        )
        self.assertEqual(
            result["methods"]["exhaustive_search"]["layout_id"], "layout-0"
        )

    def test_reconstructs_equal_time_and_equal_evaluation_prefixes(self) -> None:
        ga, random, benchmark = _summaries()
        result = build_comparison(
            ga,
            random,
            benchmark,
            layout_results=_layout_results(),
            catalogue=_catalogue(),
        )

        analysis = result["sampling_analysis"]
        self.assertEqual(
            analysis["actual_equal_time_prefix"],
            {
                "evaluated_layouts": 3,
                "best_layout_id": "layout-3",
                "best_layout_index": 3,
                "best_validation_cost_ms": 10.0,
                "best_validation_accuracy": TARGET,
            },
        )
        self.assertEqual(
            analysis["equal_ga_evaluation_prefix"]["evaluated_layouts"], 2
        )
        self.assertEqual(
            analysis["equal_ga_evaluation_prefix"]["best_validation_cost_ms"],
            10.0,
        )
        self.assertEqual(
            analysis["global_optimum"],
            {
                "validation_cost_ms": 9.0,
                "nominal_layout_count": 1,
                "first_sample_rank_zero_based": 4,
                "first_sample_rank_one_based": 5,
            },
        )
        self.assertEqual(
            analysis["exhaustive_layout_counts_relative_to_random"],
            {"strictly_better": 1, "at_or_better": 2},
        )
        self.assertTrue(analysis["actual_prefix_reproduces_random_winner"])

    def test_rejects_target_split_and_sa_mismatches(self) -> None:
        for mutation in ("target", "split", "sa"):
            with self.subTest(mutation=mutation):
                ga, random, benchmark = _summaries()
                if mutation == "target":
                    random["settings"]["target_accuracy"] = 0.95
                elif mutation == "split":
                    random["split"]["validation_samples"] = 79
                else:
                    random["settings"]["threshold_optimizer"][
                        "restart_seeds"
                    ] = list(range(1, 11))
                with self.assertRaises(ValueError):
                    build_comparison(ga, random, benchmark)

    def test_rejects_an_incomplete_exhaustive_reference(self) -> None:
        ga, random, benchmark = _summaries()
        benchmark["methods"]["exhaustive_joint"]["completed_layouts"] = 5
        with self.assertRaisesRegex(ValueError, "not a complete exhaustive search"):
            build_comparison(ga, random, benchmark)

    def test_rejects_holdout_threshold_drift_and_bad_route_totals(self) -> None:
        ga, random, benchmark = _summaries()
        random["winner"]["holdout"]["thresholds"] = {"K3@initial[0]": 0.8}
        with self.assertRaisesRegex(ValueError, "did not replay"):
            build_comparison(ga, random, benchmark)

        ga, random, benchmark = _summaries()
        ga["winner"]["validation"]["route_counts"]["detector"] = 19
        with self.assertRaisesRegex(ValueError, "do not sum"):
            build_comparison(ga, random, benchmark)

    def test_atomically_writes_under_supplied_checkpoints_root(self) -> None:
        ga, random, benchmark = _summaries()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            root.mkdir()
            paths = {
                "ga": root / "ga.json",
                "random": root / "random.json",
                "benchmark": root / "benchmark.json",
            }
            layout_results_path = root / "layout_results.jsonl"
            layout_results_path.write_text(
                "".join(json.dumps(record) + "\n" for record in _layout_results()),
                encoding="utf-8",
            )
            benchmark["methods"]["exhaustive_joint"].update(
                {
                    "layout_results": str(layout_results_path.resolve()),
                    "layout_results_sha256": hashlib.sha256(
                        layout_results_path.read_bytes()
                    ).hexdigest(),
                }
            )
            for key, payload in (
                ("ga", ga),
                ("random", random),
                ("benchmark", benchmark),
            ):
                paths[key].write_text(json.dumps(payload), encoding="utf-8")
            output = root / "comparison" / "summary.json"

            result = run_comparison(
                ga_summary_path=paths["ga"],
                random_summary_path=paths["random"],
                benchmark_summary_path=paths["benchmark"],
                output_path=output,
                checkpoints_root=root,
                catalogue=_catalogue(),
            )

            self.assertTrue(output.is_file())
            self.assertFalse(output.with_suffix(".json.tmp").exists())
            written = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(written["schema_version"], SCHEMA_VERSION)
            self.assertEqual(set(written["inputs"]), set(result["inputs"]))

            with self.assertRaisesRegex(ValueError, "under the checkpoints"):
                run_comparison(
                    ga_summary_path=paths["ga"],
                    random_summary_path=paths["random"],
                    benchmark_summary_path=paths["benchmark"],
                    output_path=Path(directory) / "outside.json",
                    checkpoints_root=root,
                    catalogue=_catalogue(),
                )

    def test_rejects_order_hash_and_saved_random_winner_mismatches(self) -> None:
        ga, random, benchmark = _summaries()
        random["settings"]["layout_order_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "layout-order hash"):
            build_comparison(
                ga,
                random,
                benchmark,
                layout_results=_layout_results(),
                catalogue=_catalogue(),
            )

        ga, random, benchmark = _summaries()
        random["winner"] = _record(11.0, 2)
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            build_comparison(
                ga,
                random,
                benchmark,
                layout_results=_layout_results(),
                catalogue=_catalogue(),
            )

    def test_verifies_exhaustive_layout_results_hash(self) -> None:
        ga, random, benchmark = _summaries()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            root.mkdir()
            paths = [root / name for name in ("ga.json", "random.json", "bench.json")]
            layout_results_path = root / "layout_results.jsonl"
            layout_results_path.write_text(
                "".join(json.dumps(record) + "\n" for record in _layout_results()),
                encoding="utf-8",
            )
            benchmark["methods"]["exhaustive_joint"].update(
                {
                    "layout_results": str(layout_results_path.resolve()),
                    "layout_results_sha256": "0" * 64,
                }
            )
            for path, payload in zip(paths, (ga, random, benchmark), strict=True):
                path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "SHA-256 does not match"):
                run_comparison(
                    ga_summary_path=paths[0],
                    random_summary_path=paths[1],
                    benchmark_summary_path=paths[2],
                    output_path=root / "comparison.json",
                    checkpoints_root=root,
                    catalogue=_catalogue(),
                )

    def test_does_not_allow_output_to_overwrite_an_input(self) -> None:
        ga, random, benchmark = _summaries()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkpoints"
            root.mkdir()
            paths = [root / name for name in ("ga.json", "random.json", "bench.json")]
            layout_results_path = root / "layout_results.jsonl"
            layout_results_path.write_text(
                "".join(json.dumps(record) + "\n" for record in _layout_results()),
                encoding="utf-8",
            )
            benchmark["methods"]["exhaustive_joint"].update(
                {
                    "layout_results": str(layout_results_path.resolve()),
                    "layout_results_sha256": hashlib.sha256(
                        layout_results_path.read_bytes()
                    ).hexdigest(),
                }
            )
            for path, payload in zip(paths, (ga, random, benchmark), strict=True):
                path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot overwrite"):
                run_comparison(
                    ga_summary_path=paths[0],
                    random_summary_path=paths[1],
                    benchmark_summary_path=paths[2],
                    layout_results_path=layout_results_path,
                    output_path=paths[0],
                    checkpoints_root=root,
                    catalogue=_catalogue(),
                )


if __name__ == "__main__":
    unittest.main()
