"""Compare the DAS 2006 continuous SA with the repository's SA+CD.

Ten layouts are sampled uniformly from the complete legal K0/K1-enabled
layout space, conditional on containing at least five distinct non-detector
classifiers.  Ten paired seeds are evaluated at 1,000, 4,000, and 8,000 SA
iterations.  This variant gives the repository optimizer 100 quantile points
and disables its global-random proposals; the Chellapilla optimizer is
continuous and consequently has no quantile grid.
"""

from __future__ import annotations

import argparse
from itertools import permutations
import json
import math
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.brute_force_k1_free_layouts import (
    DEFAULT_HOLDOUT_FRACTION,
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
    legal_layout_count,
)
from hierarchy_optimizer import HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from layout_search import LayoutSpace, TopologyGenome, cascade_from_genome, layout_id
from threshold_optimizer import (
    FixedLayoutThresholdEvaluator,
    build_threshold_grids,
    optimize_fixed_layout_thresholds_chellapilla_sa,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


DEFAULT_OUTPUT_DIR = Path("checkpoints/chellapilla_sa_comparison_h24_q100_no_random")
DEFAULT_ITERATION_BUDGETS = (1_000, 4_000, 8_000)
DEFAULT_QUANTILE_POINTS = 100
DEFAULT_LAYOUT_COUNT = 10
DEFAULT_TRIAL_COUNT = 10
DEFAULT_LAYOUT_SEED = 20260818
DEFAULT_MINIMUM_CLASSIFIERS = 5
DEFAULT_COORDINATE_DESCENT_PASSES = 25
DEFAULT_RANDOM_PROPOSAL_RATE = 0.0
DEFAULT_PAPER_CACHE = Path(
    "checkpoints/chellapilla_sa_comparison_h24/paired_trials.jsonl"
)
PAPER_URL = (
    "https://www.microsoft.com/en-us/research/wp-content/uploads/"
    "2016/02/chellapilla_das06.pdf"
)


def _ordered_subsets(values: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sequence
        for length in range(len(values) + 1)
        for sequence in permutations(values, length)
    )


def _initial_sequences(space: LayoutSpace):
    for length in range(len(space.initial_ids) + 1):
        yield from permutations(space.initial_ids, length)


def _branch_options(
    space: LayoutSpace, initial: Sequence[str]
) -> tuple[tuple[str, str, tuple[tuple[str, ...], ...]], ...]:
    return tuple(
        (
            router_id,
            group,
            _ordered_subsets(space.allowed_branch_ids(initial, router_id, group)),
        )
        for router_id in initial
        if router_id in space.router_ids
        for group in space.profile.group_ids
    )


def _uniform_random_genome(
    space: LayoutSpace, rng: np.random.Generator
) -> TopologyGenome:
    """Sample exactly uniformly without materializing all 11.6M layouts."""

    weighted_initials = []
    total = 0
    for initial in _initial_sequences(space):
        options = _branch_options(space, initial)
        weight = math.prod(len(chains) for _, _, chains in options)
        total += weight
        weighted_initials.append((total, initial, options))
    draw = int(rng.integers(total))
    for cumulative, initial, options in weighted_initials:
        if draw < cumulative:
            branches = tuple(
                (
                    router_id,
                    group,
                    chains[int(rng.integers(len(chains)))],
                )
                for router_id, group, chains in options
            )
            return TopologyGenome(tuple(initial), branches)
    raise RuntimeError("Uniform layout draw fell outside the legal space.")


def _classifier_ids(genome: TopologyGenome) -> set[str]:
    return {
        *genome.initial,
        *(candidate_id for _, _, chain in genome.branches for candidate_id in chain),
    }


def sample_layouts(
    space: LayoutSpace,
    count: int,
    seed: int,
    minimum_classifiers: int,
) -> tuple[TopologyGenome, ...]:
    rng = np.random.default_rng(seed)
    selected: list[TopologyGenome] = []
    selected_ids: set[str] = set()
    while len(selected) < count:
        genome = _uniform_random_genome(space, rng)
        if len(_classifier_ids(genome)) < minimum_classifiers:
            continue
        candidate_id = layout_id(genome, space)
        if candidate_id in selected_ids:
            continue
        selected_ids.add(candidate_id)
        selected.append(genome)
    return tuple(selected)


def _compact(metrics: Mapping[str, object]) -> dict[str, object]:
    result = _compact_optimization(metrics)
    for key in (
        "infeasible_proposals_rejected",
        "initial_temperature",
        "final_temperature",
        "cost_normalization",
        "proposal",
        "cooling",
    ):
        if key in metrics:
            result[key] = metrics[key]
    return result


def _two_sided_sign_pvalue(negative: int, positive: int) -> float | None:
    non_ties = negative + positive
    if non_ties == 0:
        return None
    extreme = min(negative, positive)
    tail = sum(math.comb(non_ties, index) for index in range(extreme + 1))
    return min(1.0, 2.0 * tail / (2**non_ties))


def _method_stats(records: list[dict[str, object]], key: str) -> dict[str, object]:
    metrics = [record[key] for record in records]
    return {
        "feasible_trials": sum(bool(item["feasible"]) for item in metrics),
        "mean_cost_ms": mean(float(item["expected_cost"]) for item in metrics),
        "median_cost_ms": median(float(item["expected_cost"]) for item in metrics),
        "mean_accuracy": mean(float(item["accuracy"]) for item in metrics),
        "mean_elapsed_seconds": mean(float(item["elapsed_seconds"]) for item in metrics),
        "mean_evaluations": mean(int(item["evaluations"]) for item in metrics),
    }


def _summarize_budget(
    records: list[dict[str, object]], layout_count: int
) -> dict[str, object]:
    paired_deltas = [
        float(record["paper_continuous_sa"]["expected_cost"])
        - float(record["ours_sa_coordinate_descent"]["expected_cost"])
        for record in records
        if bool(record["paper_continuous_sa"]["feasible"])
        and bool(record["ours_sa_coordinate_descent"]["feasible"])
    ]
    tolerance = 1e-9
    layout_details = []
    layout_best_deltas = []
    for layout_index in range(layout_count):
        subset = [
            record for record in records if int(record["layout_index"]) == layout_index
        ]
        paper_costs = [
            float(record["paper_continuous_sa"]["expected_cost"])
            for record in subset
            if bool(record["paper_continuous_sa"]["feasible"])
        ]
        ours_costs = [
            float(record["ours_sa_coordinate_descent"]["expected_cost"])
            for record in subset
            if bool(record["ours_sa_coordinate_descent"]["feasible"])
        ]
        paper_best = min(paper_costs) if paper_costs else None
        ours_best = min(ours_costs) if ours_costs else None
        delta = (
            paper_best - ours_best
            if paper_best is not None and ours_best is not None
            else None
        )
        if delta is not None:
            layout_best_deltas.append(delta)
        layout_details.append(
            {
                "layout_index": layout_index,
                "layout_id": subset[0]["layout_id"],
                "distinct_classifier_count": subset[0]["distinct_classifier_count"],
                "paper_best_cost_ms": paper_best,
                "ours_best_cost_ms": ours_best,
                "paper_minus_ours_best_cost_ms": delta,
            }
        )

    paper_wins = sum(delta < -tolerance for delta in layout_best_deltas)
    ours_wins = sum(delta > tolerance for delta in layout_best_deltas)
    ties = len(layout_best_deltas) - paper_wins - ours_wins
    return {
        "paired_trials": len(records),
        "paired_feasible_trials": len(paired_deltas),
        "paper_continuous_sa": _method_stats(records, "paper_continuous_sa"),
        "ours_sa_coordinate_descent": _method_stats(
            records, "ours_sa_coordinate_descent"
        ),
        "paired_cost_ms_paper_minus_ours": {
            "mean": mean(paired_deltas) if paired_deltas else None,
            "median": median(paired_deltas) if paired_deltas else None,
            "paper_trial_wins": sum(delta < -tolerance for delta in paired_deltas),
            "ours_trial_wins": sum(delta > tolerance for delta in paired_deltas),
            "ties": sum(abs(delta) <= tolerance for delta in paired_deltas),
        },
        "best_of_ten_by_layout": {
            "layouts_compared": len(layout_best_deltas),
            "paper_wins": paper_wins,
            "ours_wins": ours_wins,
            "ties": ties,
            "mean_paper_minus_ours_ms": (
                mean(layout_best_deltas) if layout_best_deltas else None
            ),
            "median_paper_minus_ours_ms": (
                median(layout_best_deltas) if layout_best_deltas else None
            ),
            "two_sided_sign_test_pvalue": _two_sided_sign_pvalue(
                paper_wins, ours_wins
            ),
        },
        "layouts": layout_details,
    }


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL record {line_number} in {path}") from error
    return records


def _load_paper_cache(path: Path | None) -> dict[tuple[int, str, int], dict[str, object]]:
    if path is None or not path.exists():
        return {}
    return {
        (
            int(record["iterations"]),
            str(record["layout_id"]),
            int(record["trial_seed"]),
        ): dict(record["paper_continuous_sa"])
        for record in _read_records(path)
    }


def run_benchmark(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    iteration_budgets: Sequence[int] = DEFAULT_ITERATION_BUDGETS,
    quantile_points: int = DEFAULT_QUANTILE_POINTS,
    layout_count: int = DEFAULT_LAYOUT_COUNT,
    trial_count: int = DEFAULT_TRIAL_COUNT,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
    minimum_classifiers: int = DEFAULT_MINIMUM_CLASSIFIERS,
    coordinate_descent_passes: int = DEFAULT_COORDINATE_DESCENT_PASSES,
    random_proposal_rate: float = DEFAULT_RANDOM_PROPOSAL_RATE,
    paper_cache: Path | None = DEFAULT_PAPER_CACHE,
    overwrite: bool = False,
) -> dict[str, object]:
    if layout_count < 1 or trial_count < 1 or minimum_classifiers < 1:
        raise ValueError("Layout, trial, and minimum-classifier counts must be positive.")
    budgets = tuple(sorted(set(int(value) for value in iteration_budgets)))
    if not budgets or budgets[0] < 1:
        raise ValueError("Iteration budgets must be positive.")

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
    layouts = sample_layouts(
        space, layout_count, layout_seed, minimum_classifiers
    )
    settings = {
        "paper": {
            "citation": "Chellapilla, Shilman, and Simard (DAS 2006)",
            "url": PAPER_URL,
            "continuous_thresholds": True,
            "proposal": "simultaneous_zero_mean_gaussian",
            "cooling": "geometric_1_to_1_over_validation_samples",
            "constraint": "reject_infeasible_proposals",
            "post_sa_polisher": False,
            "random_global_proposal": False,
        },
        "ours": {
            "quantile_points": quantile_points,
            "coordinate_descent_passes": coordinate_descent_passes,
            "random_global_proposal_rate": random_proposal_rate,
        },
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
        "iteration_budgets": list(budgets),
        "layout_sampling": "uniform_over_legal_layouts_conditioned_on_minimum",
        "legal_layout_space_size": legal_layout_count(space),
        "layout_count": layout_count,
        "minimum_distinct_non_detector_classifiers": minimum_classifiers,
        "layout_seed": layout_seed,
        "trial_count_per_layout": trial_count,
        "trial_seeds": list(range(trial_count)),
        "paper_cache_source": str(paper_cache.resolve()) if paper_cache else None,
        "paper_cache_sha256": (
            _file_sha256(paper_cache)
            if paper_cache is not None and paper_cache.exists()
            else None
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "settings.json"
    records_path = output_dir / "paired_trials.jsonl"
    summary_path = output_dir / "summary.json"
    if overwrite:
        settings_path.unlink(missing_ok=True)
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if settings_path.exists():
        existing_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if existing_settings != settings:
            raise ValueError(
                f"{output_dir} belongs to a different experiment; use --overwrite."
            )
    else:
        _write_json_atomic(settings_path, settings)

    records = _read_records(records_path)
    cached_paper = _load_paper_cache(paper_cache)
    completed = {
        (int(record["iterations"]), int(record["layout_index"]), int(record["trial_seed"]))
        for record in records
    }
    started = perf_counter()
    for layout_index, genome in enumerate(layouts):
        cascade = cascade_from_genome(genome, space)
        evaluator = FixedLayoutThresholdEvaluator(optimizer, cascade)
        grids = build_threshold_grids(evaluator, quantile_points)
        candidate_id = layout_id(genome, space)
        distinct_count = len(_classifier_ids(genome))
        for iterations in budgets:
            for trial_seed in range(trial_count):
                key = (iterations, layout_index, trial_seed)
                if key in completed:
                    continue
                paper_key = (iterations, candidate_id, trial_seed)
                paper_source = "cache" if paper_key in cached_paper else "computed"
                paper = cached_paper.get(paper_key)
                if paper is None:
                    paper = optimize_fixed_layout_thresholds_chellapilla_sa(
                        evaluator,
                        target_accuracy,
                        n_iterations=iterations,
                        random_seed=trial_seed,
                        show_progress=False,
                    )
                ours = optimize_fixed_layout_thresholds_simulated_annealing(
                    evaluator,
                    target_accuracy,
                    grids=grids,
                    n_iterations=iterations,
                    random_seed=trial_seed,
                    coordinate_descent_passes=coordinate_descent_passes,
                    random_proposal_rate=random_proposal_rate,
                    show_progress=False,
                )
                record = {
                    "iterations": iterations,
                    "layout_index": layout_index,
                    "layout_id": candidate_id,
                    "layout": _cascade_payload(cascade),
                    "distinct_classifier_count": distinct_count,
                    "trial_seed": trial_seed,
                    "paper_result_source": paper_source,
                    "paper_continuous_sa": _compact(paper),
                    "ours_sa_coordinate_descent": _compact(ours),
                }
                records.append(record)
                completed.add(key)
                with records_path.open("a", encoding="utf-8", buffering=1) as handle:
                    handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
                print(
                    f"iterations={iterations:5d}, layout={layout_index + 1:02d}/{layout_count}, "
                    f"trial={trial_seed + 1:02d}/{trial_count}: "
                    f"paper={float(paper['expected_cost']):.3f} ms, "
                    f"ours={float(ours['expected_cost']):.3f} ms"
                )

    summaries = {
        str(iterations): _summarize_budget(
            [record for record in records if int(record["iterations"]) == iterations],
            layout_count,
        )
        for iterations in budgets
    }
    summary = {
        "settings": settings,
        "split": split,
        "new_elapsed_seconds": perf_counter() - started,
        "paired_trials": len(records),
        "results_by_iterations": summaries,
    }
    _write_json_atomic(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY)
    parser.add_argument(
        "--iteration-budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_ITERATION_BUDGETS),
    )
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--layout-count", type=int, default=DEFAULT_LAYOUT_COUNT)
    parser.add_argument("--trial-count", type=int, default=DEFAULT_TRIAL_COUNT)
    parser.add_argument("--layout-seed", type=int, default=DEFAULT_LAYOUT_SEED)
    parser.add_argument(
        "--minimum-classifiers", type=int, default=DEFAULT_MINIMUM_CLASSIFIERS
    )
    parser.add_argument(
        "--coordinate-descent-passes",
        type=int,
        default=DEFAULT_COORDINATE_DESCENT_PASSES,
    )
    parser.add_argument(
        "--random-proposal-rate",
        type=float,
        default=DEFAULT_RANDOM_PROPOSAL_RATE,
    )
    parser.add_argument("--paper-cache", type=Path, default=DEFAULT_PAPER_CACHE)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_benchmark(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        target_accuracy=args.target_accuracy,
        iteration_budgets=args.iteration_budgets,
        quantile_points=args.quantile_points,
        layout_count=args.layout_count,
        trial_count=args.trial_count,
        layout_seed=args.layout_seed,
        minimum_classifiers=args.minimum_classifiers,
        coordinate_descent_passes=args.coordinate_descent_passes,
        random_proposal_rate=args.random_proposal_rate,
        paper_cache=args.paper_cache,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary["results_by_iterations"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
