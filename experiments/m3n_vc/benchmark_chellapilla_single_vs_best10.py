"""Compare one Chellapilla SA trajectory with best-of-ten on fixed layouts.

The default benchmark selects the established deterministic prefix of ten h24
layouts, then runs 100 independent seed blocks per layout.  Each block contains
ten 1,000-iteration continuous Gaussian SA trajectories.  Restart zero is the
single-run method and the feasibility-aware best trajectory in the same block
is the best-of-ten method, so no trajectory is redundantly recomputed.

Only validation outcomes are evaluated.  Results are append-only and resumable;
no holdout data are replayed and no figures are generated.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.benchmark_chellapilla_sa import (
    _classifier_ids,
    _compact,
    sample_layouts,
)
from experiments.m3n_vc.brute_force_k1_free_layouts import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_OUTCOMES,
    DEFAULT_SPLIT_STRATEGY,
    _cascade_payload,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    _file_sha256,
    _write_json_atomic,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga_with_k1 import (
    build_k1_layout_space,
    legal_layout_count,
)
from hierarchy_optimizer import HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from layout_search import cascade_from_genome, layout_id
from threshold_optimizer import (
    FixedLayoutThresholdEvaluator,
    _policy_key,
    optimize_fixed_layout_thresholds_chellapilla_sa,
    split_empirical_outcomes,
)


DEFAULT_OUTPUT_DIR = Path("checkpoints/chellapilla_single_vs_best10_h24")
DEFAULT_TARGET_ACCURACY = 0.9662
DEFAULT_LAYOUT_COUNT = 10
DEFAULT_TRIALS_PER_LAYOUT = 100
DEFAULT_ITERATIONS = 1_000
DEFAULT_LAYOUT_SEED = 20260818
DEFAULT_MINIMUM_CLASSIFIERS = 5
DEFAULT_WORKERS = 16
LAYOUT_SEED_STRIDE = 1_000_000
RESTARTS = 10

SETTINGS_SCHEMA_VERSION = "chellapilla-single-vs-best10-settings/v1"
RECORD_SCHEMA_VERSION = "chellapilla-single-vs-best10-trial/v1"
SUMMARY_SCHEMA_VERSION = "chellapilla-single-vs-best10-summary/v1"

_WORKER_EVALUATORS: tuple[FixedLayoutThresholdEvaluator, ...] = ()


def trial_base_seed(layout_index: int, trial_index: int) -> int:
    """Return the first seed in one disjoint ten-restart trial block."""

    return layout_index * LAYOUT_SEED_STRIDE + trial_index * RESTARTS


def _build_experiment(
    outcomes: Path,
    layout_count: int,
    layout_seed: int,
    minimum_classifiers: int,
) -> tuple[
    dict[str, object],
    dict[str, object],
    tuple[object, ...],
    tuple[object, ...],
    tuple[FixedLayoutThresholdEvaluator, ...],
]:
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
    genomes = sample_layouts(
        space,
        layout_count,
        seed=layout_seed,
        minimum_classifiers=minimum_classifiers,
    )
    cascades = tuple(cascade_from_genome(genome, space) for genome in genomes)
    evaluators = tuple(
        FixedLayoutThresholdEvaluator(optimizer, cascade) for cascade in cascades
    )
    return payload, split, genomes, cascades, evaluators


def _initialize_worker(
    outcomes: str,
    layout_count: int,
    layout_seed: int,
    minimum_classifiers: int,
) -> None:
    global _WORKER_EVALUATORS
    *_, _WORKER_EVALUATORS = _build_experiment(
        Path(outcomes), layout_count, layout_seed, minimum_classifiers
    )


def _best_restart_index(
    results: Sequence[Mapping[str, object]], target_accuracy: float
) -> int:
    if not results:
        raise ValueError("At least one restart result is required.")
    best_index = 0
    for index in range(1, len(results)):
        if _policy_key(results[index], target_accuracy) < _policy_key(
            results[best_index], target_accuracy
        ):
            best_index = index
    return best_index


def _best_of_ten_packet(
    results: Sequence[Mapping[str, object]],
    restart_seeds: Sequence[int],
    target_accuracy: float,
    iterations: int,
    group_elapsed_seconds: float,
) -> dict[str, object]:
    if len(results) != RESTARTS or len(restart_seeds) != RESTARTS:
        raise ValueError("A best-of-ten block must contain exactly ten restarts.")
    best_index = _best_restart_index(results, target_accuracy)
    winner = dict(results[best_index])
    winner.update(
        {
            "method": "best_of_10_chellapilla_continuous_gaussian_sa",
            "restart_count": RESTARTS,
            "iterations_per_restart": int(iterations),
            "total_requested_iterations": int(RESTARTS * iterations),
            "selected_restart_index": int(best_index),
            "selected_restart_seed": int(restart_seeds[best_index]),
            "restart_seeds": [int(seed) for seed in restart_seeds],
            "restart_costs_ms": [
                float(result["expected_cost"]) for result in results
            ],
            "restart_accuracies": [float(result["accuracy"]) for result in results],
            "evaluations": int(sum(int(result["evaluations"]) for result in results)),
            "elapsed_seconds": float(group_elapsed_seconds),
            "annealing_iterations": int(RESTARTS * iterations),
            "annealing_evaluations": int(
                sum(int(result["annealing_evaluations"]) for result in results)
            ),
            "annealing_elapsed_seconds": float(
                sum(float(result["annealing_elapsed_seconds"]) for result in results)
            ),
            "annealing_accepted_moves": int(
                sum(int(result["annealing_accepted_moves"]) for result in results)
            ),
            "infeasible_proposals_rejected": int(
                sum(int(result["infeasible_proposals_rejected"]) for result in results)
            ),
        }
    )
    packet = _compact(winner)
    packet.update(
        {
            "restart_feasible": [bool(result["feasible"]) for result in results],
            "restart_elapsed_seconds": [
                float(result["elapsed_seconds"]) for result in results
            ],
            "group_elapsed_seconds": float(group_elapsed_seconds),
        }
    )
    return packet


def _run_trial(
    task: tuple[int, int, float, int]
) -> tuple[
    int,
    int,
    int,
    list[int],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    float,
]:
    layout_index, trial_index, target_accuracy, iterations = task
    base_seed = trial_base_seed(layout_index, trial_index)
    restart_seeds = [base_seed + index for index in range(RESTARTS)]
    started = perf_counter()
    results = [
        optimize_fixed_layout_thresholds_chellapilla_sa(
            _WORKER_EVALUATORS[layout_index],
            target_accuracy,
            n_iterations=iterations,
            random_seed=seed,
            show_progress=False,
        )
        for seed in restart_seeds
    ]
    group_elapsed = perf_counter() - started
    restart_packets = []
    for restart_index, (seed, result) in enumerate(
        zip(restart_seeds, results, strict=True)
    ):
        packet = _compact(result)
        packet.update(
            {
                "restart_index": int(restart_index),
                "restart_seed": int(seed),
            }
        )
        restart_packets.append(packet)
    single = _compact(results[0])
    best_of_ten = _best_of_ten_packet(
        results,
        restart_seeds,
        target_accuracy,
        iterations,
        group_elapsed,
    )
    if _policy_key(best_of_ten, target_accuracy) > _policy_key(
        single, target_accuracy
    ):
        raise RuntimeError("Best-of-ten was worse than its paired restart zero.")
    return (
        layout_index,
        trial_index,
        base_seed,
        restart_seeds,
        restart_packets,
        single,
        best_of_ten,
        group_elapsed,
    )


def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in (
        "experiments/m3n_vc/benchmark_chellapilla_single_vs_best10.py",
        "experiments/m3n_vc/benchmark_chellapilla_sa.py",
        "experiments/m3n_vc/joint_optimize_hierarchy_ga_with_k1.py",
        "layout_search.py",
        "hierarchy_optimizer.py",
        "threshold_optimizer.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=float
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _records_sha256(records: Sequence[Mapping[str, object]]) -> str:
    """Hash records in canonical trial order, independent of worker completion."""

    digest = hashlib.sha256()
    for record in sorted(
        records,
        key=lambda item: (int(item["layout_index"]), int(item["trial_index"])),
    ):
        digest.update(
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), default=float
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _read_records(path: Path) -> list[dict[str, object]]:
    """Read JSONL and repair only an interrupted trailing record."""

    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    truncate_at: int | None = None
    needs_final_newline = False
    with path.open("rb") as handle:
        line_number = 0
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line_number += 1
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if handle.read().strip():
                    raise ValueError(
                        f"Malformed non-final JSONL line {line_number} in {path}."
                    ) from exc
                truncate_at = line_start
                print(f"Removing incomplete JSONL line {line_number} in {path}")
                break
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record {line_number} is not an object.")
            records.append(record)
            needs_final_newline = not raw_line.endswith(b"\n")
    if truncate_at is not None:
        with path.open("r+b") as handle:
            handle.truncate(truncate_at)
    elif needs_final_newline:
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def _layout_metadata(
    genomes: Sequence[object], cascades: Sequence[object], space: object
) -> list[dict[str, object]]:
    return [
        {
            "layout_index": index,
            "layout_id": layout_id(genome, space),
            "layout": _cascade_payload(cascade),
            "distinct_non_detector_classifiers": len(_classifier_ids(genome)),
        }
        for index, (genome, cascade) in enumerate(
            zip(genomes, cascades, strict=True)
        )
    ]


def _validate_method_result(
    result: object, label: str, target_accuracy: float
) -> Mapping[str, object]:
    if not isinstance(result, Mapping):
        raise ValueError(f"{label} is not an object.")
    for field in ("accuracy", "expected_cost", "feasible", "thresholds"):
        if field not in result:
            raise ValueError(f"{label} is missing {field}.")
    feasible = bool(float(result["accuracy"]) >= target_accuracy)
    if bool(result["feasible"]) != feasible:
        raise ValueError(f"{label} has an inconsistent feasibility flag.")
    return result


def _validate_records(
    records: Sequence[Mapping[str, object]],
    *,
    settings_sha256: str,
    layouts: Sequence[Mapping[str, object]],
    trials_per_layout: int,
    target_accuracy: float,
    iterations: int,
) -> set[tuple[int, int]]:
    completed: set[tuple[int, int]] = set()
    for position, record in enumerate(records, 1):
        label = f"record {position}"
        if record.get("schema_version") != RECORD_SCHEMA_VERSION:
            raise ValueError(f"{label} has the wrong schema version.")
        if record.get("settings_sha256") != settings_sha256:
            raise ValueError(f"{label} belongs to another experiment.")
        if record.get("dataset") != "m3n_vc/h24":
            raise ValueError(f"{label} has the wrong dataset.")
        if record.get("partition") != "validation":
            raise ValueError(f"{label} is not a validation record.")
        if record.get("holdout_usage") != "not_evaluated" or "holdout" in record:
            raise ValueError(f"{label} violates the validation-only contract.")
        layout_index = int(record.get("layout_index", -1))
        trial_index = int(record.get("trial_index", -1))
        if not 0 <= layout_index < len(layouts):
            raise ValueError(f"{label} has an invalid layout index.")
        if not 0 <= trial_index < trials_per_layout:
            raise ValueError(f"{label} has an invalid trial index.")
        key = (layout_index, trial_index)
        if key in completed:
            raise ValueError(f"Duplicate completed trial {key}.")
        completed.add(key)

        metadata = layouts[layout_index]
        for field in ("layout_id", "layout", "distinct_non_detector_classifiers"):
            if record.get(field) != metadata[field]:
                raise ValueError(f"{label} has mismatched layout metadata.")
        expected_base = trial_base_seed(layout_index, trial_index)
        expected_seeds = [expected_base + index for index in range(RESTARTS)]
        if int(record.get("base_seed", -1)) != expected_base:
            raise ValueError(f"{label} has the wrong base seed.")
        if record.get("restart_seeds") != expected_seeds:
            raise ValueError(f"{label} has the wrong restart seed block.")
        if int(record.get("iterations_per_restart", -1)) != iterations:
            raise ValueError(f"{label} has the wrong iteration count.")

        single = _validate_method_result(
            record.get("single"), f"{label}.single", target_accuracy
        )
        best = _validate_method_result(
            record.get("best_of_10"), f"{label}.best_of_10", target_accuracy
        )
        costs = best.get("restart_costs_ms")
        accuracies = best.get("restart_accuracies")
        feasible = best.get("restart_feasible")
        restart_elapsed = best.get("restart_elapsed_seconds")
        restart_packets = record.get("restarts")
        if not all(
            isinstance(values, list) and len(values) == RESTARTS
            for values in (
                costs,
                accuracies,
                feasible,
                restart_elapsed,
                restart_packets,
            )
        ):
            raise ValueError(f"{label} has incomplete restart results.")
        if best.get("method") != "best_of_10_chellapilla_continuous_gaussian_sa":
            raise ValueError(f"{label} has the wrong best-of-ten method.")
        if single.get("method") != "chellapilla_continuous_gaussian_sa":
            raise ValueError(f"{label} has the wrong single-run method.")
        if int(best.get("restart_count", -1)) != RESTARTS:
            raise ValueError(f"{label} has the wrong restart count.")
        if int(best.get("iterations_per_restart", -1)) != iterations:
            raise ValueError(f"{label} has the wrong best-of-ten iteration count.")
        if best.get("restart_seeds") != expected_seeds:
            raise ValueError(f"{label} has inconsistent nested restart seeds.")
        for restart_index, packet in enumerate(restart_packets):
            validated = _validate_method_result(
                packet, f"{label}.restarts[{restart_index}]", target_accuracy
            )
            if int(validated.get("restart_index", -1)) != restart_index:
                raise ValueError(f"{label} has the wrong restart index metadata.")
            if int(validated.get("restart_seed", -1)) != expected_seeds[restart_index]:
                raise ValueError(f"{label} has the wrong restart seed metadata.")
            if float(validated["expected_cost"]) != float(
                costs[restart_index]
            ) or float(validated["accuracy"]) != float(accuracies[restart_index]):
                raise ValueError(f"{label} restart arrays disagree with full packets.")
            if bool(validated["feasible"]) != bool(feasible[restart_index]):
                raise ValueError(f"{label} restart feasibility arrays disagree.")
            if float(validated["elapsed_seconds"]) != float(
                restart_elapsed[restart_index]
            ):
                raise ValueError(f"{label} restart elapsed arrays disagree.")
            if int(validated.get("annealing_iterations", -1)) != iterations:
                raise ValueError(f"{label} restart has the wrong iteration count.")
        selected_index = int(best.get("selected_restart_index", -1))
        expected_selected = _best_restart_index(restart_packets, target_accuracy)
        if selected_index != expected_selected:
            raise ValueError(f"{label} selected the wrong best restart.")
        if int(best.get("selected_restart_seed", -1)) != expected_seeds[selected_index]:
            raise ValueError(f"{label} selected the wrong restart seed.")
        if float(single["expected_cost"]) != float(costs[0]) or float(
            single["accuracy"]
        ) != float(accuracies[0]):
            raise ValueError(f"{label} single result is not restart zero.")
        restart_zero = {
            key: value
            for key, value in restart_packets[0].items()
            if key not in {"restart_index", "restart_seed"}
        }
        if dict(single) != restart_zero:
            raise ValueError(f"{label} single packet was not reused from restart zero.")
        if float(best["expected_cost"]) != float(costs[selected_index]) or float(
            best["accuracy"]
        ) != float(accuracies[selected_index]):
            raise ValueError(f"{label} winner does not match its selected restart.")
        aggregate_overrides = {
            "method",
            "evaluations",
            "elapsed_seconds",
            "annealing_iterations",
            "annealing_evaluations",
            "annealing_elapsed_seconds",
            "annealing_accepted_moves",
            "infeasible_proposals_rejected",
            "restart_index",
            "restart_seed",
        }
        selected_packet = restart_packets[selected_index]
        for field, value in selected_packet.items():
            if field not in aggregate_overrides and best.get(field) != value:
                raise ValueError(
                    f"{label} winner field {field!r} does not match its full restart packet."
                )
        if _policy_key(best, target_accuracy) > _policy_key(single, target_accuracy):
            raise ValueError(f"{label} best-of-ten is worse than restart zero.")
        if float(record.get("group_elapsed_seconds", -1.0)) < 0.0:
            raise ValueError(f"{label} has an invalid group elapsed time.")
        if float(best.get("group_elapsed_seconds", -1.0)) != float(
            record["group_elapsed_seconds"]
        ):
            raise ValueError(f"{label} has inconsistent group elapsed time.")
    return completed


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    maximum = float(np.max(array))
    return {
        "minimum": float(np.min(array)),
        "maximum": maximum,
        "highest": maximum,
        "mean": float(mean(array)),
        "median": float(median(array)),
        "standard_deviation": (
            float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        ),
    }


def _method_summary(
    records: Sequence[Mapping[str, object]], field: str
) -> dict[str, object]:
    results = [record[field] for record in records]
    return {
        "count": len(results),
        "feasible_count": sum(bool(result["feasible"]) for result in results),
        "feasible_rate": (
            sum(bool(result["feasible"]) for result in results) / len(results)
            if results
            else None
        ),
        "cost_ms": _distribution(
            [float(result["expected_cost"]) for result in results]
        ),
        "accuracy": _distribution([float(result["accuracy"]) for result in results]),
    }


def _paired_summary(
    records: Sequence[Mapping[str, object]], target_accuracy: float
) -> dict[str, object]:
    counts = {"best_of_10_better": 0, "tie": 0, "single_better": 0}
    cost_improvements: list[float] = []
    accuracy_changes: list[float] = []
    selected_indices: list[float] = []
    for record in records:
        single = record["single"]
        best = record["best_of_10"]
        single_key = _policy_key(single, target_accuracy)
        best_key = _policy_key(best, target_accuracy)
        if best_key < single_key:
            counts["best_of_10_better"] += 1
        elif single_key < best_key:
            counts["single_better"] += 1
        else:
            counts["tie"] += 1
        cost_improvements.append(
            float(single["expected_cost"]) - float(best["expected_cost"])
        )
        accuracy_changes.append(float(best["accuracy"]) - float(single["accuracy"]))
        selected_indices.append(float(best["selected_restart_index"]))
    total = len(records)
    return {
        "count": total,
        "comparison": counts,
        "rates": {
            key: (value / total if total else None) for key, value in counts.items()
        },
        "cost_improvement_ms_single_minus_best_of_10": _distribution(
            cost_improvements
        ),
        "accuracy_change_best_of_10_minus_single": _distribution(accuracy_changes),
        "selected_restart_index": _distribution(selected_indices),
        "restart_zero_selected_count": sum(index == 0.0 for index in selected_indices),
    }


def summarize_records(
    records: Sequence[Mapping[str, object]],
    layouts: Sequence[Mapping[str, object]],
    target_accuracy: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    layout_summaries: list[dict[str, object]] = []
    for layout in layouts:
        index = int(layout["layout_index"])
        subset = [record for record in records if int(record["layout_index"]) == index]
        layout_summaries.append(
            {
                **dict(layout),
                "completed_trials": len(subset),
                "methods": {
                    "single": _method_summary(subset, "single"),
                    "best_of_10": _method_summary(subset, "best_of_10"),
                },
                "paired": _paired_summary(subset, target_accuracy),
                "group_elapsed_seconds": _distribution(
                    [float(record["group_elapsed_seconds"]) for record in subset]
                ),
            }
        )
    pooled = {
        "methods": {
            "single": _method_summary(records, "single"),
            "best_of_10": _method_summary(records, "best_of_10"),
        },
        "paired": _paired_summary(records, target_accuracy),
        "group_elapsed_seconds": _distribution(
            [float(record["group_elapsed_seconds"]) for record in records]
        ),
    }
    return layout_summaries, pooled


def _summary_packet(
    *,
    records: Sequence[Mapping[str, object]],
    layouts: Sequence[Mapping[str, object]],
    settings: Mapping[str, object],
    split: Mapping[str, object],
    trials_per_layout: int,
    target_accuracy: float,
    status: str,
    wall_elapsed_seconds_this_invocation: float,
    settings_path: Path,
    records_path: Path,
) -> dict[str, object]:
    layout_summaries, pooled = summarize_records(records, layouts, target_accuracy)
    expected = len(layouts) * trials_per_layout
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": status,
        "settings": dict(settings),
        "settings_sha256": _payload_sha256(settings),
        "records_sha256": _records_sha256(records),
        "settings_path": str(settings_path.resolve()),
        "settings_file_sha256": _file_sha256(settings_path),
        "records_path": str(records_path.resolve()),
        "records_file_sha256": _file_sha256(records_path),
        "split": dict(split),
        "partition": "validation",
        "holdout_usage": "not_evaluated",
        "completed_trials": len(records),
        "expected_trials": expected,
        "completed_trajectories": len(records) * RESTARTS,
        "expected_trajectories": expected * RESTARTS,
        "completion_fraction": len(records) / expected,
        "wall_elapsed_seconds_this_invocation": float(
            wall_elapsed_seconds_this_invocation
        ),
        "layouts": layout_summaries,
        "pooled": pooled,
    }


def run_benchmark(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    layout_count: int = DEFAULT_LAYOUT_COUNT,
    trials_per_layout: int = DEFAULT_TRIALS_PER_LAYOUT,
    iterations: int = DEFAULT_ITERATIONS,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
    minimum_classifiers: int = DEFAULT_MINIMUM_CLASSIFIERS,
    workers: int = DEFAULT_WORKERS,
    overwrite: bool = False,
) -> dict[str, object]:
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1 inclusive.")
    if min(
        layout_count,
        trials_per_layout,
        iterations,
        minimum_classifiers,
        workers,
    ) < 1:
        raise ValueError("Counts, iterations, and workers must be positive.")
    if trials_per_layout * RESTARTS >= LAYOUT_SEED_STRIDE:
        raise ValueError("Trial seed blocks would overlap adjacent layout seed ranges.")

    payload, split, genomes, cascades, evaluators = _build_experiment(
        outcomes, layout_count, layout_seed, minimum_classifiers
    )
    space = build_k1_layout_space(payload)
    layouts = _layout_metadata(genomes, cascades, space)
    settings: dict[str, object] = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "dataset": "m3n_vc/h24",
        "partition": "validation",
        "holdout_usage": "not_evaluated",
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "implementation_sha256": _implementation_sha256(),
        "target_accuracy": float(target_accuracy),
        "target_accuracy_source": "explicit_cli_or_api_override",
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
        "split_seed": 0,
        "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "candidate_policy": "K0_K1_full_space",
        "removed_candidates": [],
        "legal_layout_space_size": legal_layout_count(space),
        "layout_sampling": (
            "established_exact_uniform_sample_layouts_prefix_conditioned_on_minimum"
        ),
        "layout_seed": int(layout_seed),
        "layout_count": int(layout_count),
        "minimum_distinct_non_detector_classifiers": int(minimum_classifiers),
        "layouts": layouts,
        "trials_per_layout": int(trials_per_layout),
        "iterations_per_restart": int(iterations),
        "restarts_per_best_of_10": RESTARTS,
        "trial_base_seed_rule": (
            f"layout_index * {LAYOUT_SEED_STRIDE} + trial_index * {RESTARTS}"
        ),
        "restart_seed_rule": "trial_base_seed + restart_index",
        "seed_blocks_disjoint_across_trials_and_layouts": True,
        "pairing": {
            "single": "restart_index_0",
            "best_of_10": "feasibility_aware_best_of_restart_indices_0_through_9",
            "single_is_member_of_best_of_10_block": True,
            "single_trajectory_recomputed": False,
            "all_full_restart_packets_persisted": True,
        },
        "optimizer": {
            "single_method": "chellapilla_continuous_gaussian_sa",
            "best_method": "best_of_10_chellapilla_continuous_gaussian_sa",
            "continuous_thresholds": True,
            "post_sa_polisher": False,
            "random_global_proposal": False,
            "selection": "repository_feasibility_aware_policy_key",
        },
        "workers": int(workers),
    }
    settings_sha256 = _payload_sha256(settings)

    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "settings.json"
    records_path = output_dir / "trial_packets.jsonl"
    summary_path = output_dir / "summary.json"
    if overwrite:
        for path in (settings_path, records_path, summary_path):
            path.unlink(missing_ok=True)
    if settings_path.exists():
        existing = json.loads(settings_path.read_text(encoding="utf-8"))
        if existing != settings:
            raise ValueError(f"{output_dir} has different settings; use --overwrite.")
    else:
        _write_json_atomic(settings_path, settings)

    records = _read_records(records_path)
    completed = _validate_records(
        records,
        settings_sha256=settings_sha256,
        layouts=layouts,
        trials_per_layout=trials_per_layout,
        target_accuracy=target_accuracy,
        iterations=iterations,
    )
    tasks = [
        (layout_index, trial_index, float(target_accuracy), int(iterations))
        for layout_index in range(layout_count)
        for trial_index in range(trials_per_layout)
        if (layout_index, trial_index) not in completed
    ]

    metadata_by_index = {int(layout["layout_index"]): layout for layout in layouts}
    started = perf_counter()

    def persist(
        worker_result: tuple[
            int,
            int,
            int,
            list[int],
            list[dict[str, object]],
            dict[str, object],
            dict[str, object],
            float,
        ]
    ) -> None:
        (
            layout_index,
            trial_index,
            base_seed,
            restart_seeds,
            restart_packets,
            single,
            best_of_ten,
            group_elapsed,
        ) = worker_result
        metadata = metadata_by_index[layout_index]
        record: dict[str, object] = {
            "schema_version": RECORD_SCHEMA_VERSION,
            "settings_sha256": settings_sha256,
            "dataset": "m3n_vc/h24",
            "partition": "validation",
            "holdout_usage": "not_evaluated",
            **dict(metadata),
            "trial_index": int(trial_index),
            "base_seed": int(base_seed),
            "restart_seeds": restart_seeds,
            "iterations_per_restart": int(iterations),
            "restarts": restart_packets,
            "single": single,
            "best_of_10": best_of_ten,
            "group_elapsed_seconds": float(group_elapsed),
        }
        records.append(record)
        with records_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")

    if tasks and workers == 1:
        global _WORKER_EVALUATORS
        _WORKER_EVALUATORS = evaluators
        for completed_now, task in enumerate(tasks, 1):
            persist(_run_trial(task))
            if completed_now % 10 == 0 or completed_now == len(tasks):
                elapsed = perf_counter() - started
                eta = elapsed / completed_now * (len(tasks) - completed_now)
                print(
                    f"Completed {completed_now:,}/{len(tasks):,} pending paired trials; "
                    f"ETA {eta / 60.0:.1f} min"
                )
                _write_json_atomic(
                    summary_path,
                    _summary_packet(
                        records=records,
                        layouts=layouts,
                        settings=settings,
                        split=split,
                        trials_per_layout=trials_per_layout,
                        target_accuracy=target_accuracy,
                        status="running",
                        wall_elapsed_seconds_this_invocation=elapsed,
                        settings_path=settings_path,
                        records_path=records_path,
                    ),
                )
    elif tasks:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(
                str(outcomes),
                layout_count,
                layout_seed,
                minimum_classifiers,
            ),
        ) as executor:
            futures = [executor.submit(_run_trial, task) for task in tasks]
            for completed_now, future in enumerate(as_completed(futures), 1):
                persist(future.result())
                if completed_now % 10 == 0 or completed_now == len(tasks):
                    elapsed = perf_counter() - started
                    eta = elapsed / completed_now * (len(tasks) - completed_now)
                    print(
                        f"Completed {completed_now:,}/{len(tasks):,} pending paired trials; "
                        f"ETA {eta / 60.0:.1f} min"
                    )
                    _write_json_atomic(
                        summary_path,
                        _summary_packet(
                            records=records,
                            layouts=layouts,
                            settings=settings,
                            split=split,
                            trials_per_layout=trials_per_layout,
                            target_accuracy=target_accuracy,
                            status="running",
                            wall_elapsed_seconds_this_invocation=elapsed,
                            settings_path=settings_path,
                            records_path=records_path,
                        ),
                    )

    elapsed = perf_counter() - started
    final_summary = _summary_packet(
        records=records,
        layouts=layouts,
        settings=settings,
        split=split,
        trials_per_layout=trials_per_layout,
        target_accuracy=target_accuracy,
        status=("complete" if len(records) == layout_count * trials_per_layout else "running"),
        wall_elapsed_seconds_this_invocation=elapsed,
        settings_path=settings_path,
        records_path=records_path,
    )
    _write_json_atomic(summary_path, final_summary)
    return final_summary


def dry_run_report(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    layout_count: int = DEFAULT_LAYOUT_COUNT,
    trials_per_layout: int = DEFAULT_TRIALS_PER_LAYOUT,
    iterations: int = DEFAULT_ITERATIONS,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
    minimum_classifiers: int = DEFAULT_MINIMUM_CLASSIFIERS,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, object]:
    payload, split, genomes, cascades, _ = _build_experiment(
        outcomes, layout_count, layout_seed, minimum_classifiers
    )
    space = build_k1_layout_space(payload)
    return {
        "dataset": "m3n_vc/h24",
        "partition": "validation",
        "holdout_usage": "not_evaluated",
        "target_accuracy": float(target_accuracy),
        "legal_layout_space_size": legal_layout_count(space),
        "layouts": _layout_metadata(genomes, cascades, space),
        "trials_per_layout": int(trials_per_layout),
        "paired_trial_count": int(layout_count * trials_per_layout),
        "trajectories_per_trial": RESTARTS,
        "total_unique_trajectories": int(
            layout_count * trials_per_layout * RESTARTS
        ),
        "iterations_per_trajectory": int(iterations),
        "workers": int(workers),
        "split": split,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY)
    parser.add_argument("--layout-count", type=int, default=DEFAULT_LAYOUT_COUNT)
    parser.add_argument(
        "--trials-per-layout", type=int, default=DEFAULT_TRIALS_PER_LAYOUT
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--layout-seed", type=int, default=DEFAULT_LAYOUT_SEED)
    parser.add_argument(
        "--minimum-classifiers", type=int, default=DEFAULT_MINIMUM_CLASSIFIERS
    )
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        report = dry_run_report(
            outcomes=args.outcomes,
            target_accuracy=args.target_accuracy,
            layout_count=args.layout_count,
            trials_per_layout=args.trials_per_layout,
            iterations=args.iterations,
            layout_seed=args.layout_seed,
            minimum_classifiers=args.minimum_classifiers,
            workers=args.workers,
        )
        print(json.dumps(report, indent=2, sort_keys=True, default=float))
        return
    summary = run_benchmark(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        target_accuracy=args.target_accuracy,
        layout_count=args.layout_count,
        trials_per_layout=args.trials_per_layout,
        iterations=args.iterations,
        layout_seed=args.layout_seed,
        minimum_classifiers=args.minimum_classifiers,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    pooled = summary["pooled"]
    print(json.dumps(pooled, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    # Required for Windows ProcessPoolExecutor spawn semantics.
    main()
