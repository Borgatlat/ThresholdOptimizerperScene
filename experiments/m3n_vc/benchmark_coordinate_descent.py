"""Paired ablation of SA threshold optimization with coordinate descent.

The benchmark samples ten legal K0/K1-enabled h24 layouts.  For each layout,
ten random seeds run the same 8,000-step simulated annealer twice: once with
no local polish and once followed by coordinate descent.  Only validation is
used; the held-out partition is never evaluated during this ablation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Mapping

import numpy as np

from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.brute_force_k1_free_layouts import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_ITERATIONS,
    DEFAULT_OUTCOMES,
    DEFAULT_SPLIT_STRATEGY,
    _cascade_payload,
    _compact_optimization,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    _file_sha256,
    _write_json_atomic,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga_with_k1 import (
    DEFAULT_TARGET_ACCURACY,
    build_k1_layout_space,
)
from hierarchy_optimizer import HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from layout_search import cascade_from_genome, layout_id, random_genome
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    FixedLayoutThresholdEvaluator,
    build_threshold_grids,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


DEFAULT_OUTPUT_DIR = Path("checkpoints/coordinate_descent_ablation_h24")
DEFAULT_LAYOUT_COUNT = 10
DEFAULT_TRIAL_COUNT = 10
DEFAULT_LAYOUT_SEED = 20260814
DEFAULT_COORDINATE_DESCENT_PASSES = 25


def _sample_layouts(space, count: int, seed: int):
    rng = np.random.default_rng(seed)
    selected = []
    selected_ids: set[str] = set()
    while len(selected) < count:
        genome = random_genome(space, rng)
        # Direct detector has no threshold and therefore cannot test polishing.
        if not genome.initial:
            continue
        candidate_id = layout_id(genome, space)
        if candidate_id in selected_ids:
            continue
        selected_ids.add(candidate_id)
        selected.append(genome)
    return tuple(selected)


def _selection_key(metrics: Mapping[str, object], target: float) -> tuple[float, ...]:
    accuracy = float(metrics["accuracy"])
    cost = float(metrics["expected_cost"])
    if accuracy >= target:
        return (0.0, cost, -accuracy)
    return (1.0, target - accuracy, cost)


def _two_sided_sign_pvalue(negative: int, positive: int) -> float | None:
    non_ties = negative + positive
    if non_ties == 0:
        return None
    extreme = min(negative, positive)
    tail = sum(math.comb(non_ties, index) for index in range(extreme + 1))
    return min(1.0, 2.0 * tail / (2**non_ties))


def _bootstrap_mean_interval(values: list[float], seed: int = 0) -> list[float] | None:
    if not values:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(20_000, len(array)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def _summarize(records: list[dict[str, object]], layout_count: int) -> dict[str, object]:
    trial_deltas: list[float] = []
    trial_improvements = 0
    trial_ties = 0
    trial_regressions = 0
    selection_improvements = 0
    trajectory_mismatches = 0
    layout_results: list[dict[str, object]] = []

    for record in records:
        without = record["without_coordinate_descent"]
        with_descent = record["with_coordinate_descent"]
        assert isinstance(without, Mapping) and isinstance(with_descent, Mapping)
        if (
            without.get("annealing_evaluations")
            != with_descent.get("annealing_evaluations")
            or without.get("annealing_accepted_moves")
            != with_descent.get("annealing_accepted_moves")
        ):
            trajectory_mismatches += 1
        if _selection_key(with_descent, float(with_descent["target_accuracy"])) < _selection_key(
            without, float(without["target_accuracy"])
        ):
            selection_improvements += 1
        if bool(without["feasible"]) and bool(with_descent["feasible"]):
            delta = float(with_descent["expected_cost"]) - float(without["expected_cost"])
            trial_deltas.append(delta)
            if delta < -1e-9:
                trial_improvements += 1
            elif delta > 1e-9:
                trial_regressions += 1
            else:
                trial_ties += 1

    layout_best_deltas: list[float] = []
    for layout_index in range(layout_count):
        layout_records = [
            record for record in records if int(record["layout_index"]) == layout_index
        ]
        without_costs = [
            float(record["without_coordinate_descent"]["expected_cost"])
            for record in layout_records
            if bool(record["without_coordinate_descent"]["feasible"])
        ]
        with_costs = [
            float(record["with_coordinate_descent"]["expected_cost"])
            for record in layout_records
            if bool(record["with_coordinate_descent"]["feasible"])
        ]
        best_without = min(without_costs) if without_costs else None
        best_with = min(with_costs) if with_costs else None
        delta = (
            best_with - best_without
            if best_without is not None and best_with is not None
            else None
        )
        if delta is not None:
            layout_best_deltas.append(delta)
        first = layout_records[0]
        layout_results.append(
            {
                "layout_index": layout_index,
                "layout_id": first["layout_id"],
                "layout": first["layout"],
                "feasible_trials_without": len(without_costs),
                "feasible_trials_with": len(with_costs),
                "best_cost_without": best_without,
                "best_cost_with": best_with,
                "best_cost_delta_ms": delta,
            }
        )

    negative = sum(delta < -1e-9 for delta in layout_best_deltas)
    positive = sum(delta > 1e-9 for delta in layout_best_deltas)
    ties = len(layout_best_deltas) - negative - positive
    return {
        "paired_trials": len(records),
        "paired_feasible_trials": len(trial_deltas),
        "identical_sa_trajectory_checks_failed": trajectory_mismatches,
        "selection_improved_after_coordinate_descent": selection_improvements,
        "trial_cost_improvements": trial_improvements,
        "trial_cost_ties": trial_ties,
        "trial_cost_regressions": trial_regressions,
        "trial_delta_cost_ms": {
            "mean": mean(trial_deltas) if trial_deltas else None,
            "median": median(trial_deltas) if trial_deltas else None,
            "minimum": min(trial_deltas) if trial_deltas else None,
            "maximum": max(trial_deltas) if trial_deltas else None,
        },
        "layout_best_cost": {
            "layouts_compared": len(layout_best_deltas),
            "improvements": negative,
            "ties": ties,
            "regressions": positive,
            "mean_delta_ms": mean(layout_best_deltas) if layout_best_deltas else None,
            "median_delta_ms": median(layout_best_deltas) if layout_best_deltas else None,
            "bootstrap_95pct_mean_delta_ms": _bootstrap_mean_interval(layout_best_deltas),
            "two_sided_sign_test_pvalue": _two_sided_sign_pvalue(negative, positive),
        },
        "layouts": layout_results,
    }


def run_benchmark(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    iterations: int = DEFAULT_ITERATIONS,
    quantile_points: int = DEFAULT_QUANTILE_POINTS,
    layout_count: int = DEFAULT_LAYOUT_COUNT,
    trial_count: int = DEFAULT_TRIAL_COUNT,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
    coordinate_descent_passes: int = DEFAULT_COORDINATE_DESCENT_PASSES,
    overwrite: bool = False,
) -> dict[str, object]:
    if layout_count < 1 or trial_count < 1:
        raise ValueError("layout_count and trial_count must be positive.")
    payload = load_empirical_outcomes(outcomes)
    validation_payload, _, split = split_empirical_outcomes(
        payload,
        holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
        split_strategy=DEFAULT_SPLIT_STRATEGY,
        random_seed=0,
    )
    optimizer = HierarchyOptimizer(
        validation_payload,
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )
    space = build_k1_layout_space(payload)
    layouts = _sample_layouts(space, layout_count, layout_seed)
    settings = {
        "dataset": "m3n_vc/h24",
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "partition": "validation",
        "holdout_usage": "not_evaluated",
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
        "split_seed": 0,
        "target_accuracy": target_accuracy,
        "detector_mode": "paper",
        "detector_cost_ms": PAPER_DETECTOR_COST_MS,
        "iterations": iterations,
        "quantile_points": quantile_points,
        "layout_count": layout_count,
        "trial_count_per_layout": trial_count,
        "layout_seed": layout_seed,
        "trial_seeds": list(range(trial_count)),
        "coordinate_descent_passes": coordinate_descent_passes,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "paired_trials.jsonl"
    summary_path = output_dir / "summary.json"
    if overwrite:
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if records_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"{output_dir} already contains benchmark output; pass --overwrite."
        )

    records: list[dict[str, object]] = []
    started = perf_counter()
    for layout_index, genome in enumerate(layouts):
        cascade = cascade_from_genome(genome, space)
        evaluator = FixedLayoutThresholdEvaluator(optimizer, cascade)
        grids = build_threshold_grids(evaluator, quantile_points)
        candidate_id = layout_id(genome, space)
        for trial_seed in range(trial_count):
            without = optimize_fixed_layout_thresholds_simulated_annealing(
                evaluator,
                target_accuracy,
                grids=grids,
                n_iterations=iterations,
                random_seed=trial_seed,
                coordinate_descent_passes=0,
                show_progress=False,
            )
            with_descent = optimize_fixed_layout_thresholds_simulated_annealing(
                evaluator,
                target_accuracy,
                grids=grids,
                n_iterations=iterations,
                random_seed=trial_seed,
                coordinate_descent_passes=coordinate_descent_passes,
                show_progress=False,
            )
            record = {
                "layout_index": layout_index,
                "layout_id": candidate_id,
                "layout": _cascade_payload(cascade),
                "trial_seed": trial_seed,
                "without_coordinate_descent": _compact_optimization(without),
                "with_coordinate_descent": _compact_optimization(with_descent),
            }
            records.append(record)
            with records_path.open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
            print(
                f"layout {layout_index + 1:02d}/{layout_count}, "
                f"trial {trial_seed + 1:02d}/{trial_count}: "
                f"SA={float(without['expected_cost']):.3f} ms, "
                f"SA+CD={float(with_descent['expected_cost']):.3f} ms"
            )

    summary = {
        "settings": settings,
        "split": split,
        "elapsed_seconds": perf_counter() - started,
        "results": _summarize(records, layout_count),
    }
    _write_json_atomic(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--layout-count", type=int, default=DEFAULT_LAYOUT_COUNT)
    parser.add_argument("--trial-count", type=int, default=DEFAULT_TRIAL_COUNT)
    parser.add_argument("--layout-seed", type=int, default=DEFAULT_LAYOUT_SEED)
    parser.add_argument(
        "--coordinate-descent-passes",
        type=int,
        default=DEFAULT_COORDINATE_DESCENT_PASSES,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_benchmark(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        target_accuracy=args.target_accuracy,
        iterations=args.iterations,
        quantile_points=args.quantile_points,
        layout_count=args.layout_count,
        trial_count=args.trial_count,
        layout_seed=args.layout_seed,
        coordinate_descent_passes=args.coordinate_descent_passes,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary["results"], indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
