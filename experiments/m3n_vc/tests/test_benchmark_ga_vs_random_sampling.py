from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from experiments.m3n_vc import benchmark_ga_vs_random_sampling as benchmark


def _record(
    layout_id: str,
    index: int,
    *,
    cost: float,
    accuracy: float = 0.97,
    feasible: bool = True,
    completion: float | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "layout_id": layout_id,
        "layout_index": index,
        "validation": {
            "expected_cost": cost,
            "accuracy": accuracy,
            "feasible": feasible,
        },
    }
    if completion is not None:
        record["search_elapsed_seconds_at_completion"] = completion
    return record


class PairedComparisonTests(unittest.TestCase):
    def test_sign_test_known_outcomes(self) -> None:
        self.assertEqual(
            benchmark.exact_two_sided_sign_test_p_value(0, 0), 1.0
        )
        self.assertAlmostEqual(
            benchmark.exact_two_sided_sign_test_p_value(10, 0),
            2.0 / 1024.0,
        )
        self.assertAlmostEqual(
            benchmark.exact_two_sided_sign_test_p_value(9, 1),
            22.0 / 1024.0,
        )

    def test_outcome_uses_feasibility_aware_selection(self) -> None:
        feasible = {"feasible": True, "accuracy": 0.9662, "expected_cost": 900.0}
        cheap_infeasible = {
            "feasible": False,
            "accuracy": 0.9661,
            "expected_cost": 1.0,
        }
        self.assertEqual(
            benchmark.paired_outcome(feasible, cheap_infeasible), "ga_better"
        )
        self.assertEqual(
            benchmark.paired_outcome(feasible, dict(feasible)), "tie"
        )

    def test_deadline_censor_excludes_crossing_layout(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "evaluations.jsonl"
            records = [
                _record("a", 0, cost=800.0, completion=9.0),
                _record("b", 1, cost=700.0, completion=10.5),
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            winner, included, excluded = benchmark._deadline_censored_random_winner(
                evaluations_path=path, budget_seconds=10.0
            )
        self.assertIsNotNone(winner)
        assert winner is not None
        self.assertEqual(winner["layout_id"], "a")
        self.assertEqual((included, excluded), (1, 1))

    def test_oracle_prefix_is_seeded_without_replacement(self) -> None:
        catalogue = tuple(
            SimpleNamespace(layout_id=layout_id) for layout_id in ("a", "b", "c")
        )
        oracle = {
            "a": _record("a", 0, cost=900.0),
            "b": _record("b", 1, cost=800.0),
            "c": _record("c", 2, cost=700.0),
        }
        order = benchmark.uniform_layout_order(catalogue, 4)
        expected = min(
            (oracle[item.layout_id] for item in order[:2]),
            key=benchmark._layout_selection_key,
        )
        actual = benchmark.oracle_random_prefix_winner(
            oracle_records=oracle, catalogue=catalogue, seed=4, count=2
        )
        self.assertEqual(actual["layout_id"], expected["layout_id"])

    def test_dry_run_is_fixed_to_ten_seeds_and_phase_barrier(self) -> None:
        report = benchmark.dry_run_report(
            output_dir=Path("unused"), parallel_trials=10
        )
        self.assertEqual(report["trial_seeds"], list(range(10)))
        self.assertEqual(report["phase_order"], ["all_GA", "all_random"])
        self.assertLess(
            report["idealized_parallel_lower_bound_hours"],
            report["estimated_sequential_hours"],
        )
        self.assertIn("K3->detector", report["search_asymmetry"])


class ResumeAndProvenanceTests(unittest.TestCase):
    def test_partial_ga_is_archived_without_deletion(self) -> None:
        with TemporaryDirectory() as temporary:
            method_dir = Path(temporary) / "ga"
            method_dir.mkdir()
            (method_dir / "evaluations.jsonl").write_text("partial", encoding="utf-8")
            archived = benchmark._archive_partial_ga(method_dir)
            self.assertFalse(method_dir.exists())
            self.assertEqual(
                (archived / "evaluations.jsonl").read_text(encoding="utf-8"),
                "partial",
            )

    def test_import_reference_hashes_but_does_not_copy_source(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / "external"
            source_dir.mkdir()
            source = source_dir / "summary.json"
            source.write_text('{"complete": true}\n', encoding="utf-8")
            method_dir = root / "trial_00" / "ga"
            benchmark._import_reference(
                method_dir=method_dir,
                phase="genetic_algorithm",
                seed=0,
                source_summary=source,
            )
            reference = json.loads(
                (method_dir / "harness_phase.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reference["source_summary"], str(source.resolve()))
            self.assertEqual(
                reference["source_summary_sha256"], benchmark._file_sha256(source)
            )
            self.assertEqual(
                sorted(path.name for path in method_dir.iterdir()),
                ["harness_phase.json"],
            )
            self.assertEqual(source.read_text(encoding="utf-8"), '{"complete": true}\n')

    def test_phase_executor_is_sequential_at_parallelism_one(self) -> None:
        seen: list[int] = []

        def worker(payload: dict[str, object]) -> dict[str, object]:
            seed = int(payload["seed"])
            seen.append(seed)
            return {"seed": seed, "wall": 0.01}

        benchmark._execute_phase(
            name="test phase",
            tasks=({"seed": 2}, {"seed": 3}),
            worker=worker,
            parallel_trials=1,
        )
        self.assertEqual(seen, [2, 3])


if __name__ == "__main__":
    unittest.main()
