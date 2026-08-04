"""Replay fixed and annealed outer-GA schedules against validation ground truth.

This is a cheap diagnostic for the topology-search policy.  It uses the
completed exhaustive run as a validation-fitness oracle, so it does *not* run
the 8,000-step inner threshold annealer again.  Every GA decision is based only
on each layout's validation accuracy and expected cost; holdout fields in the
source JSONL are deliberately ignored and are never copied into memory used by
the replay.

The paired comparison mirrors the production search defaults: 32 individuals,
24 generations, a hard budget of exactly 512 unique layouts, and identical
stagnation/restart behavior.  For each seed, the fixed and annealed variants
start from the same initial population.

Examples
--------
Run 250 paired seeds and write the full report::

    python benchmark_ga_outer_schedules.py --runs 250

Run a quick smoke comparison without writing a report::

    python benchmark_ga_outer_schedules.py --runs 2 --no-output
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from joint_optimize_hierarchy_ga import (
    ANNEALED_OUTER_SCHEDULE,
    DEFAULT_COMPONENT_RESAMPLE_RATE,
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_ELITE_COUNT,
    DEFAULT_EVALUATION_BUDGET,
    DEFAULT_GENERATIONS,
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_RESTARTS,
    DEFAULT_MUTATION_RATE,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_QUANTILE_POINTS,
    DEFAULT_RANDOM_IMMIGRANT_RATE,
    DEFAULT_SEED,
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_STAGNATION_GENERATIONS,
    DEFAULT_TOURNAMENT_SIZE,
    FIG1_K3_TARGET_ACCURACY,
    LayoutCatalogue,
    PAPER_DETECTOR_COST_MS,
    REMOVED_CANDIDATES,
    build_layout_catalogue,
    initial_population,
    next_population,
    outer_ga_parameters,
    restart_population,
    topology_selection_key,
)


DEFAULT_RESULTS = Path(
    "checkpoints/brute_force_k1_free_h24/results_shard_00000_of_00001.jsonl"
)
DEFAULT_OUTPUT = Path(
    "checkpoints/joint_ga_outer_schedule_oracle_benchmark.json"
)


ValidationRecord = dict[str, object]


def _validate_source_contract(
    payload: Mapping[str, object], line_number: int
) -> None:
    """Reject fitness tables produced under a different experiment contract."""

    settings = payload.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError(f"Line {line_number} has no experiment settings.")
    expected: dict[str, object] = {
        "target_accuracy": FIG1_K3_TARGET_ACCURACY,
        "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
        "iterations": DEFAULT_ITERATIONS,
        "quantile_points": DEFAULT_QUANTILE_POINTS,
        "seed": DEFAULT_SEED,
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
        "detector_mode": "paper",
        "detector_cost_ms": PAPER_DETECTOR_COST_MS,
    }
    for name, expected_value in expected.items():
        if settings.get(name) != expected_value:
            raise ValueError(
                f"Line {line_number} uses {name}={settings.get(name)!r}; "
                f"this replay requires {expected_value!r}."
            )
    if list(settings.get("removed_candidates", [])) != list(REMOVED_CANDIDATES):
        raise ValueError(
            f"Line {line_number} does not use the required K1-free model set."
        )


def load_validation_oracle(path: Path) -> dict[str, ValidationRecord]:
    """Load only the fields needed by the validation selection objective."""

    if not path.is_file():
        raise FileNotFoundError(f"Exhaustive results do not exist: {path}")

    records: dict[str, ValidationRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            _validate_source_contract(payload, line_number)
            validation = payload.get("validation")
            if not isinstance(validation, Mapping):
                raise ValueError(
                    f"Line {line_number} has no validation metric mapping."
                )
            try:
                record: ValidationRecord = {
                    "layout_id": str(payload["layout_id"]),
                    "layout_index": int(payload["layout_index"]),
                    "validation": {
                        "accuracy": float(validation["accuracy"]),
                        "expected_cost": float(validation["expected_cost"]),
                    },
                }
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Line {line_number} has malformed validation metrics."
                ) from error

            layout_id = str(record["layout_id"])
            if layout_id in records:
                raise ValueError(f"Duplicate exhaustive layout id: {layout_id}")
            records[layout_id] = record

    if not records:
        raise ValueError(f"No exhaustive validation records found in {path}.")
    return records


def validate_oracle(
    records: Mapping[str, ValidationRecord], catalogue: LayoutCatalogue
) -> None:
    """Require a complete, one-to-one oracle for the legal layout catalogue."""

    catalogue_ids = set(catalogue.by_id)
    record_ids = set(records)
    missing = catalogue_ids - record_ids
    unexpected = record_ids - catalogue_ids
    if missing or unexpected:
        raise ValueError(
            "Exhaustive validation oracle does not match the legal catalogue: "
            f"missing={len(missing)}, unexpected={len(unexpected)}."
        )
    for layout_id, record in records.items():
        expected_index = catalogue.by_id[layout_id].index
        if int(record["layout_index"]) != expected_index:
            raise ValueError(
                f"Layout {layout_id} has index {record['layout_index']}, "
                f"expected {expected_index}."
            )


def _best_record(
    records: Mapping[str, ValidationRecord], target_accuracy: float
) -> ValidationRecord:
    return min(
        records.values(),
        key=lambda record: topology_selection_key(record, target_accuracy),
    )


def replay_schedule(
    oracle: Mapping[str, ValidationRecord],
    catalogue: LayoutCatalogue,
    *,
    seed: int,
    target_accuracy: float,
    annealed: bool,
) -> dict[str, object]:
    """Replay one production-equivalent outer search using cached fitness."""

    rng = np.random.default_rng(seed)
    population = initial_population(catalogue, DEFAULT_POPULATION_SIZE, rng)
    evaluated: dict[str, ValidationRecord] = {}
    previous_best_id: str | None = None
    stagnant_generations = 0
    restart_count = 0
    generation = 0

    while generation < DEFAULT_GENERATIONS:
        remaining_budget = DEFAULT_EVALUATION_BUDGET - len(evaluated)
        missing = [item for item in population if item not in evaluated]
        if len(missing) > remaining_budget:
            cached = [item for item in population if item in evaluated]
            population = [*cached, *missing[:remaining_budget]]
            missing = [item for item in population if item not in evaluated]

        for layout_id in missing:
            evaluated[layout_id] = oracle[layout_id]
        if not population:
            raise RuntimeError("GA replay produced an empty population.")

        best = _best_record(evaluated, target_accuracy)
        best_id = str(best["layout_id"])
        if previous_best_id is None or previous_best_id != best_id:
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        previous_best_id = best_id

        if len(evaluated) >= DEFAULT_EVALUATION_BUDGET:
            break
        if generation + 1 >= DEFAULT_GENERATIONS:
            break

        progress = min(1.0, len(evaluated) / DEFAULT_EVALUATION_BUDGET)
        parameters = outer_ga_parameters(progress, annealed=annealed)
        remaining_unique = DEFAULT_EVALUATION_BUDGET - len(evaluated)
        if (
            stagnant_generations >= DEFAULT_STAGNATION_GENERATIONS
            and restart_count < DEFAULT_MAX_RESTARTS
        ):
            # Production restarts retain one cached global elite, whereas
            # ordinary breeding retains the scheduled number of cached elites.
            desired_size = min(DEFAULT_POPULATION_SIZE, 1 + remaining_unique)
            population = restart_population(
                evaluated,
                catalogue,
                rng,
                target_accuracy=target_accuracy,
                population_size=desired_size,
            )
            restart_count += 1
            stagnant_generations = 0
        else:
            desired_size = min(
                DEFAULT_POPULATION_SIZE,
                parameters.elite_count + remaining_unique,
            )
            population = next_population(
                population,
                evaluated,
                catalogue,
                rng,
                target_accuracy=target_accuracy,
                population_size=desired_size,
                elite_count=min(parameters.elite_count, desired_size - 1),
                tournament_size=parameters.tournament_size,
                crossover_rate=parameters.crossover_rate,
                mutation_rate=parameters.mutation_rate,
                random_immigrant_rate=parameters.random_immigrant_rate,
                component_resample_rate=parameters.component_resample_rate,
                excluded_layout_ids=set(evaluated),
            )
        generation += 1

    if len(evaluated) != DEFAULT_EVALUATION_BUDGET:
        raise RuntimeError(
            "Replay did not consume the exact unique-layout budget: "
            f"expected {DEFAULT_EVALUATION_BUDGET}, got {len(evaluated)}."
        )

    winner = _best_record(evaluated, target_accuracy)
    validation = winner["validation"]
    assert isinstance(validation, Mapping)
    return {
        "winner_layout_id": str(winner["layout_id"]),
        "winner_layout_index": int(winner["layout_index"]),
        "validation_accuracy": float(validation["accuracy"]),
        "validation_cost_ms": float(validation["expected_cost"]),
        "feasible": float(validation["accuracy"]) >= target_accuracy,
        "unique_layouts_evaluated": len(evaluated),
        "last_generation": generation,
        "restart_count": restart_count,
        # Kept transiently for paired overlap; removed before serialization.
        "_visited_layout_ids": set(evaluated),
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "best": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "worst": float(np.max(array)),
    }


def _two_sided_sign_test_p_value(annealed_better: int, fixed_better: int) -> float:
    """Exact binomial sign test under equal non-tied win probabilities."""

    non_ties = annealed_better + fixed_better
    if non_ties == 0:
        return 1.0
    lower_tail = min(annealed_better, fixed_better)
    tail_outcomes = sum(math.comb(non_ties, k) for k in range(lower_tail + 1))
    return min(1.0, 2.0 * tail_outcomes / (1 << non_ties))


def _variant_summary(
    runs: Sequence[Mapping[str, object]], *, top_one_percent_rank: int
) -> dict[str, object]:
    ranks = [float(item["exhaustive_rank"]) for item in runs]
    regrets = [float(item["validation_cost_regret_ms"]) for item in runs]
    return {
        "exhaustive_rank": _numeric_summary(ranks),
        "validation_cost_regret_ms": _numeric_summary(regrets),
        "exact_optimum_rate": sum(rank == 1 for rank in ranks) / len(ranks),
        "top_10_rate": sum(rank <= 10 for rank in ranks) / len(ranks),
        "top_1_percent_rate": (
            sum(rank <= top_one_percent_rank for rank in ranks) / len(ranks)
        ),
        "feasible_rate": (
            sum(bool(item["feasible"]) for item in runs) / len(runs)
        ),
        "mean_restart_count": float(
            np.mean([int(item["restart_count"]) for item in runs])
        ),
    }


def run_benchmark(
    *,
    results_path: Path = DEFAULT_RESULTS,
    runs: int = 250,
    seed_start: int = 0,
) -> dict[str, object]:
    """Run paired fixed/annealed oracle replays and return a JSON-safe report."""

    if runs < 1:
        raise ValueError("runs must be at least 1.")
    target_accuracy = FIG1_K3_TARGET_ACCURACY

    catalogue = build_layout_catalogue()
    oracle = load_validation_oracle(results_path)
    validate_oracle(oracle, catalogue)
    exhaustive_order = sorted(
        oracle,
        key=lambda layout_id: topology_selection_key(
            oracle[layout_id], target_accuracy
        ),
    )
    rank_by_id = {
        layout_id: rank
        for rank, layout_id in enumerate(exhaustive_order, start=1)
    }
    optimum = oracle[exhaustive_order[0]]
    optimum_validation = optimum["validation"]
    assert isinstance(optimum_validation, Mapping)
    optimum_cost = float(optimum_validation["expected_cost"])

    fixed_runs: list[dict[str, object]] = []
    annealed_runs: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    comparison_counts = {"annealed_better": 0, "equal": 0, "fixed_better": 0}
    progress_interval = max(1, runs // 20)

    for offset in range(runs):
        seed = seed_start + offset
        fixed = replay_schedule(
            oracle,
            catalogue,
            seed=seed,
            target_accuracy=target_accuracy,
            annealed=False,
        )
        annealed = replay_schedule(
            oracle,
            catalogue,
            seed=seed,
            target_accuracy=target_accuracy,
            annealed=True,
        )
        fixed_visited = fixed.pop("_visited_layout_ids")
        annealed_visited = annealed.pop("_visited_layout_ids")
        assert isinstance(fixed_visited, set)
        assert isinstance(annealed_visited, set)

        for result in (fixed, annealed):
            winner_id = str(result["winner_layout_id"])
            result["exhaustive_rank"] = rank_by_id[winner_id]
            result["validation_cost_regret_ms"] = (
                float(result["validation_cost_ms"]) - optimum_cost
                if bool(result["feasible"])
                else None
            )

        fixed_rank = int(fixed["exhaustive_rank"])
        annealed_rank = int(annealed["exhaustive_rank"])
        if annealed_rank < fixed_rank:
            outcome = "annealed_better"
        elif annealed_rank == fixed_rank:
            outcome = "equal"
        else:
            outcome = "fixed_better"
        comparison_counts[outcome] += 1

        intersection = len(fixed_visited & annealed_visited)
        union = len(fixed_visited | annealed_visited)
        fixed_runs.append(fixed)
        annealed_runs.append(annealed)
        pairs.append(
            {
                "seed": seed,
                "outcome": outcome,
                "annealed_minus_fixed_rank": annealed_rank - fixed_rank,
                "annealed_minus_fixed_cost_regret_ms": (
                    float(annealed["validation_cost_regret_ms"])
                    - float(fixed["validation_cost_regret_ms"])
                ),
                "visited_layout_intersection": intersection,
                "visited_layout_jaccard": intersection / union,
                "fixed": fixed,
                "annealed": annealed,
            }
        )
        if (offset + 1) % progress_interval == 0 or offset + 1 == runs:
            print(f"Completed paired replay {offset + 1:,}/{runs:,}")

    top_one_percent_rank = math.ceil(len(oracle) * 0.01)
    outcome_rates = {
        name: count / runs for name, count in comparison_counts.items()
    }
    sign_test_p_value = _two_sided_sign_test_p_value(
        comparison_counts["annealed_better"],
        comparison_counts["fixed_better"],
    )
    rank_differences = [
        float(item["annealed_minus_fixed_rank"]) for item in pairs
    ]
    regret_differences = [
        float(item["annealed_minus_fixed_cost_regret_ms"]) for item in pairs
    ]
    overlap_values = [float(item["visited_layout_jaccard"]) for item in pairs]

    return {
        "method": "paired_validation_oracle_replay",
        "validation_only": True,
        "holdout_fields_used": False,
        "interpretation_limit": (
            "Outer-search diagnostic using previously optimized validation "
            "fitness; not an independent threshold-optimization or holdout run."
        ),
        "source_results": str(results_path.resolve()),
        "contract": {
            "population_size": DEFAULT_POPULATION_SIZE,
            "maximum_generations": DEFAULT_GENERATIONS,
            "unique_layout_budget": DEFAULT_EVALUATION_BUDGET,
            "stagnation_generations": DEFAULT_STAGNATION_GENERATIONS,
            "max_restarts": DEFAULT_MAX_RESTARTS,
            "fixed_parameters": {
                "elite_count": DEFAULT_ELITE_COUNT,
                "tournament_size": DEFAULT_TOURNAMENT_SIZE,
                "crossover_rate": DEFAULT_CROSSOVER_RATE,
                "mutation_rate": DEFAULT_MUTATION_RATE,
                "random_immigrant_rate": DEFAULT_RANDOM_IMMIGRANT_RATE,
                "component_resample_rate": DEFAULT_COMPONENT_RESAMPLE_RATE,
            },
            "annealed_schedule": ANNEALED_OUTER_SCHEDULE,
            "schedule_progress": "unique_layouts_evaluated / 512",
            "restart_logic": "identical_between_variants",
        },
        "target_accuracy": target_accuracy,
        "oracle_layout_count": len(oracle),
        "oracle_optimum": {
            "layout_id": str(optimum["layout_id"]),
            "layout_index": int(optimum["layout_index"]),
            "validation_accuracy": float(optimum_validation["accuracy"]),
            "validation_cost_ms": optimum_cost,
        },
        "paired_seed_count": runs,
        "seed_start": seed_start,
        "top_1_percent_rank_cutoff": top_one_percent_rank,
        "summary": {
            "fixed": _variant_summary(
                fixed_runs, top_one_percent_rank=top_one_percent_rank
            ),
            "annealed": _variant_summary(
                annealed_runs, top_one_percent_rank=top_one_percent_rank
            ),
            "paired_outcomes": comparison_counts,
            "paired_outcome_rates": outcome_rates,
            "paired_sign_test": {
                "test": "exact_two_sided_binomial_sign_test",
                "ties_excluded": comparison_counts["equal"],
                "null_hypothesis": (
                    "Annealed and fixed schedules are equally likely to win "
                    "conditional on a non-tied paired seed."
                ),
                "p_value": sign_test_p_value,
            },
            "annealed_minus_fixed_rank": _numeric_summary(rank_differences),
            "annealed_minus_fixed_cost_regret_ms": _numeric_summary(
                regret_differences
            ),
            "visited_layout_jaccard": _numeric_summary(overlap_values),
        },
        "paired_runs": pairs,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Completed exhaustive JSONL used only as a validation oracle.",
    )
    parser.add_argument("--runs", type=int, default=250)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Run and print the summary without writing a JSON report.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_benchmark(
        results_path=args.results,
        runs=args.runs,
        seed_start=args.seed_start,
    )
    if not args.no_output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
