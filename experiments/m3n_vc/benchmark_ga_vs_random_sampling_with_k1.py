"""Paired K1-enabled GA-versus-random joint-search benchmark at p=0.90.

Ten trials vary only the outer layout-search seed.  The validation/holdout
split and the ten inner threshold-SA restart seeds are fixed, so both methods
see one deterministic fitness value for every topology.  Random search gets
the validation-search wall time reported by its paired 512-layout GA run.

There is no exhaustive oracle for the 11,589,085-layout K1-enabled space.
Consequently this benchmark reports paired validation outcomes, not ranks or
regret to the unknown global optimum.  The GA always receives the mandatory
K3 -> detector seed; exact-uniform random sampling receives no mandatory seed.
That intentional historical asymmetry is recorded in every result packet.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter, process_time
from typing import Callable, Mapping, Sequence

# Independent trials provide the parallelism.  Prevent each spawned process
# from silently creating another BLAS thread pool and distorting paired times.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np

from experiments.m3n_vc.brute_force_k1_free_layouts import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_OUTCOMES,
    DEFAULT_SPLIT_STRATEGY,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    _file_sha256,
    _settings_match,
    _write_json_atomic,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga_with_k1 import (
    DEFAULT_COMPONENT_RESAMPLE_RATE,
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_ELITE_COUNT,
    DEFAULT_EVALUATION_BUDGET,
    DEFAULT_GENERATIONS,
    DEFAULT_MUTATION_RATE,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_RANDOM_IMMIGRANT_RATE,
    DEFAULT_TOURNAMENT_SIZE,
    _implementation_sha256 as _ga_implementation_sha256,
    run_k1_search,
)
from experiments.m3n_vc.random_joint_optimize_hierarchy_with_k1 import (
    _implementation_sha256 as _random_implementation_sha256,
    run_k1_random_search,
)
from hierarchy_optimizer import PAPER_DETECTOR_COST_MS


SCHEMA_VERSION = "paired-ga-vs-random-with-k1/v1"
TARGET_ACCURACY = 0.90
LAYOUT_SPACE_SIZE = 11_589_085
TRIAL_SEEDS = tuple(range(10))
INNER_SEED = 0
SPLIT_SEED = 0
ITERATIONS_PER_RESTART = 1_000
RESTARTS = 10
QUANTILE_POINTS = 50
COMPARISON_TOLERANCE_MS = 1e-9
PRIOR_SINGLE_TRIAL_GA_SECONDS = 1_941.985601

DEFAULT_OUTPUT_DIR = Path(
    "checkpoints/ga_vs_random_with_k1_h24_target_090_paper_sa_10_trials"
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a list.")
    return value


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    return _mapping(value, label)


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    records: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{label} has invalid JSON on line {line_number}."
                ) from error
            records.append(_mapping(value, f"{label} line {line_number}"))
    return records


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} is {actual!r}; expected {expected!r}.")


def _require_close(actual: object, expected: float, label: str) -> None:
    try:
        number = float(actual)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric.") from error
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} is {number!r}; expected {expected!r}.")


def _canonical_split(value: object, label: str) -> dict[str, object]:
    split = _mapping(value, label)
    return {
        "strategy": split.get("strategy"),
        "random_seed": split.get("random_seed"),
        "holdout_fraction": split.get("holdout_fraction"),
        "validation_samples": split.get("validation_samples"),
        "holdout_samples": split.get("holdout_samples"),
        "per_run": split.get("per_run", split.get("per_group")),
    }


def _threshold_contract(settings: Mapping[str, object], label: str) -> None:
    optimizer = _mapping(settings.get("threshold_optimizer"), label)
    expected = {
        "method": "best_of_10_chellapilla_continuous_gaussian_sa",
        "iterations_per_restart": ITERATIONS_PER_RESTART,
        "restarts": RESTARTS,
        "restart_seeds": list(range(RESTARTS)),
        "continuous_thresholds": True,
        "quantile_points_used": False,
        "prune_stages_accepting_zero_validation_samples": True,
        "freeze_validation_active_slots_on_holdout": True,
    }
    for key, value in expected.items():
        _require_equal(optimizer.get(key), value, f"{label}.{key}")


def experiment_contract(outcomes: Path) -> dict[str, object]:
    """Return the immutable scientific contract saved beside all trials."""

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "m3n_vc/h24",
        "target_accuracy": TARGET_ACCURACY,
        "layout_grammar": "depth_one_K0_K1",
        "layout_space_size": LAYOUT_SPACE_SIZE,
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "removed_candidates": [],
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "split": {
            "strategy": DEFAULT_SPLIT_STRATEGY,
            "seed": SPLIT_SEED,
            "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
        },
        "threshold_optimizer": {
            "method": "best_of_10_chellapilla_continuous_gaussian_sa",
            "iterations_per_restart": ITERATIONS_PER_RESTART,
            "restarts": RESTARTS,
            "restart_seeds": list(range(RESTARTS)),
            "inner_seed_varies_across_trials": False,
            "quantile_points_recorded_but_unused": QUANTILE_POINTS,
            "prune_stages_accepting_zero_validation_samples": True,
            "freeze_validation_active_slots_on_holdout": True,
        },
        "implementation_fingerprints": {
            "genetic_algorithm": _ga_implementation_sha256(),
            "random_sampling": _random_implementation_sha256(),
        },
        "trial_seeds": list(TRIAL_SEEDS),
        "varied_seed_fields": {
            "genetic_algorithm": "outer_seed",
            "random_sampling": "sampling_seed",
        },
        "fixed_seed_fields": {
            "inner_seed": INNER_SEED,
            "split_seed": SPLIT_SEED,
            "threshold_restart_seeds": list(range(RESTARTS)),
        },
        "genetic_algorithm": {
            "algorithm": "dynamic_constrained_memetic_genetic_algorithm",
            "population_size": DEFAULT_POPULATION_SIZE,
            "generations": DEFAULT_GENERATIONS,
            "evaluation_budget": DEFAULT_EVALUATION_BUDGET,
            "elite_count": DEFAULT_ELITE_COUNT,
            "tournament_size": DEFAULT_TOURNAMENT_SIZE,
            "crossover_rate": DEFAULT_CROSSOVER_RATE,
            "mutation_rate": DEFAULT_MUTATION_RATE,
            "random_immigrant_rate": DEFAULT_RANDOM_IMMIGRANT_RATE,
            "component_resample_rate": DEFAULT_COMPONENT_RESAMPLE_RATE,
            "allow_cached_reentry": False,
            "internal_layout_workers": 1,
            "mandatory_initial_member": {
                "initial": ["K3", "detector"],
                "specialized": {},
            },
        },
        "random_sampling": {
            "algorithm": "uniform_random_implicit_layout_sampling_without_replacement",
            "sampler": "exact_uniform_over_all_legal_layouts",
            "minimum_classifier_condition": None,
            "mandatory_initial_member": None,
            "time_budget": "paired_ga_elapsed_seconds_this_invocation",
            "atomic_layout_overshoot_allowed": True,
        },
        "primary_comparison": {
            "scope": "validation_only",
            "selection_rule": "feasibility_aware_topology_selection",
            "cost_delta": "random_minus_ga_ms_positive_favors_ga",
        },
        "search_asymmetry": (
            "The GA is seeded with mandatory K3->detector while exact-uniform "
            "random sampling is unseeded. In addition, the GA representation's "
            "random initial members and immigrants sample component lengths "
            "uniformly, which overweights shorter layouts relative to exact-uniform "
            "sampling over the complete legal space. Results therefore compare the "
            "complete implemented search procedures, including both asymmetries."
        ),
        "oracle_limitation": (
            "No exhaustive K1-enabled oracle exists for the 11,589,085-layout "
            "space; global ranks, exact-optimum recovery, and optimum regret are "
            "therefore unavailable."
        ),
    }


def _validate_common_summary(
    summary: Mapping[str, object],
    *,
    outcomes_sha256: str,
    label: str,
) -> tuple[Mapping[str, object], Mapping[str, object], dict[str, object]]:
    settings = _mapping(summary.get("settings"), f"{label}.settings")
    expected = {
        "dataset": "m3n_vc/h24",
        "target_accuracy": TARGET_ACCURACY,
        "removed_candidates": [],
        "layout_grammar": "depth_one_K0_K1",
        "layout_space_size": LAYOUT_SPACE_SIZE,
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "outcomes_sha256": outcomes_sha256,
        "split_seed": SPLIT_SEED,
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
        "inner_seed": INNER_SEED,
    }
    for key, value in expected.items():
        if isinstance(value, float):
            _require_close(settings.get(key), value, f"{label}.settings.{key}")
        else:
            _require_equal(settings.get(key), value, f"{label}.settings.{key}")
    _require_close(
        settings.get("holdout_fraction"),
        DEFAULT_HOLDOUT_FRACTION,
        f"{label}.settings.holdout_fraction",
    )
    _require_equal(
        settings.get("quantile_points_compatibility_argument"),
        QUANTILE_POINTS,
        f"{label}.settings.quantile_points_compatibility_argument",
    )
    _threshold_contract(settings, f"{label}.settings.threshold_optimizer")

    split = _canonical_split(summary.get("split"), f"{label}.split")
    _require_equal(split["strategy"], DEFAULT_SPLIT_STRATEGY, f"{label}.split.strategy")
    _require_equal(split["random_seed"], SPLIT_SEED, f"{label}.split.random_seed")
    _require_close(
        split["holdout_fraction"],
        DEFAULT_HOLDOUT_FRACTION,
        f"{label}.split.holdout_fraction",
    )

    winner = _mapping(summary.get("winner"), f"{label}.winner")
    validation = _mapping(winner.get("validation"), f"{label}.winner.validation")
    holdout = _mapping(winner.get("holdout"), f"{label}.winner.holdout")
    for partition, metrics in (("validation", validation), ("holdout", holdout)):
        float(metrics["accuracy"])
        float(metrics["expected_cost"])
        if "feasible" not in metrics:
            raise ValueError(f"{label}.winner.{partition} has no feasibility flag.")
    _require_equal(
        summary.get("holdout_usage"),
        "winner_only_after_validation_search",
        f"{label}.holdout_usage",
    )
    return settings, winner, split


def validate_ga_summary(
    summary: Mapping[str, object], *, seed: int, outcomes_sha256: str
) -> Mapping[str, object]:
    settings, winner, _ = _validate_common_summary(
        summary, outcomes_sha256=outcomes_sha256, label=f"GA seed {seed}"
    )
    expected = {
        "algorithm": "dynamic_constrained_memetic_genetic_algorithm",
        "outer_seed": seed,
        "random_seed": seed,
        "population_size": DEFAULT_POPULATION_SIZE,
        "generations": DEFAULT_GENERATIONS,
        "evaluation_budget": DEFAULT_EVALUATION_BUDGET,
        "elite_count": DEFAULT_ELITE_COUNT,
        "tournament_size": DEFAULT_TOURNAMENT_SIZE,
        "crossover_rate": DEFAULT_CROSSOVER_RATE,
        "mutation_rate": DEFAULT_MUTATION_RATE,
        "random_immigrant_rate": DEFAULT_RANDOM_IMMIGRANT_RATE,
        "component_resample_rate": DEFAULT_COMPONENT_RESAMPLE_RATE,
        "allow_cached_reentry": False,
        "fitness_implementation_sha256": _ga_implementation_sha256(),
    }
    for key, value in expected.items():
        _require_equal(settings.get(key), value, f"GA seed {seed}.{key}")
    _require_equal(
        summary.get("layout_space_size"), LAYOUT_SPACE_SIZE, "GA layout space"
    )
    _require_equal(
        summary.get("unique_layouts_evaluated"),
        DEFAULT_EVALUATION_BUDGET,
        "GA evaluated layouts",
    )
    elapsed = float(summary.get("elapsed_seconds_this_invocation", -1.0))
    if elapsed <= 0.0:
        raise ValueError("GA reported elapsed time must be positive.")
    return winner


def validate_random_summary(
    summary: Mapping[str, object],
    *,
    seed: int,
    ga_elapsed_seconds: float,
    outcomes_sha256: str,
) -> Mapping[str, object]:
    settings, winner, _ = _validate_common_summary(
        summary, outcomes_sha256=outcomes_sha256, label=f"random seed {seed}"
    )
    expected = {
        "schema_version": "random-joint-layout-search-with-k1/v1",
        "algorithm": "uniform_random_implicit_layout_sampling_without_replacement",
        "sampling_seed": seed,
        "max_layouts": None,
        "fitness_implementation_sha256": _random_implementation_sha256(),
    }
    for key, value in expected.items():
        _require_equal(settings.get(key), value, f"random seed {seed}.{key}")
    sampler = _mapping(settings.get("layout_sampler"), "random layout sampler")
    sampler_expected = {
        "schema_version": "implicit-depth-one-layout-sampler/v1",
        "selection": "exact_uniform_without_replacement",
        "deduplication_key": "canonical_space_index",
        "materializes_full_space": False,
    }
    for key, value in sampler_expected.items():
        _require_equal(sampler.get(key), value, f"random sampler.{key}")
    _require_close(
        settings.get("time_budget_seconds"),
        ga_elapsed_seconds,
        f"random seed {seed} time budget",
    )
    count = int(summary.get("unique_layouts_evaluated", 0))
    if not 1 <= count <= LAYOUT_SPACE_SIZE:
        raise ValueError("Random evaluated-layout count is invalid.")
    if count < LAYOUT_SPACE_SIZE:
        _require_equal(
            summary.get("stop_reason"), "time_budget_reached", "random stop reason"
        )
    search_elapsed = float(summary.get("search_elapsed_seconds", -1.0))
    if search_elapsed < ga_elapsed_seconds:
        raise ValueError("Random search stopped before its paired GA time budget.")
    _require_close(
        summary.get("time_budget_overshoot_seconds"),
        max(0.0, search_elapsed - ga_elapsed_seconds),
        "random time-budget overshoot",
    )
    return winner


def _policy_key(validation: Mapping[str, object]) -> tuple[float, ...]:
    feasible = bool(validation.get("feasible"))
    accuracy = float(validation["accuracy"])
    cost = float(validation["expected_cost"])
    if feasible:
        return (0.0, cost, -accuracy)
    return (1.0, -accuracy, cost)


def _record_key(record: Mapping[str, object]) -> tuple[float, ...]:
    validation = _mapping(record.get("validation"), "record.validation")
    return (*_policy_key(validation), float(record.get("layout_index", 0)))


def paired_outcome(
    ga_validation: Mapping[str, object], random_validation: Mapping[str, object]
) -> str:
    ga_key = _policy_key(ga_validation)
    random_key = _policy_key(random_validation)
    if all(
        math.isclose(first, second, rel_tol=0.0, abs_tol=COMPARISON_TOLERANCE_MS)
        for first, second in zip(ga_key, random_key)
    ):
        return "tie"
    return "ga_better" if ga_key < random_key else "random_better"


def exact_two_sided_sign_test_p_value(ga_better: int, random_better: int) -> float:
    non_ties = ga_better + random_better
    if non_ties == 0:
        return 1.0
    lower_tail = min(ga_better, random_better)
    probability = sum(math.comb(non_ties, index) for index in range(lower_tail + 1))
    return min(1.0, 2.0 * probability / (1 << non_ties))


def _numeric_summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "population_stddev": float(pstdev(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _bootstrap_mean_interval(values: Sequence[float]) -> list[float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260824)
    resampled = rng.choice(array, size=(20_000, len(array)), replace=True)
    return [
        float(value)
        for value in np.quantile(resampled.mean(axis=1), (0.025, 0.975))
    ]


def _validate_evaluations_file(
    *, summary_path: Path, summary: Mapping[str, object], method: str
) -> tuple[Path, str, list[Mapping[str, object]]]:
    if method == "ga":
        path = summary_path.parent / "evaluations.jsonl"
    else:
        path = Path(str(summary.get("evaluations", "")))
        if not path.is_absolute():
            path = summary_path.parent / path
    expected_count = int(summary["unique_layouts_evaluated"])
    records = _read_jsonl(path, f"{method} evaluations")
    if len(records) != expected_count:
        raise ValueError(f"{method} evaluation count does not match its summary.")
    ids = {str(record.get("layout_id")) for record in records}
    if len(ids) != expected_count:
        raise ValueError(f"{method} evaluations contain duplicate layouts.")
    summary_settings = _mapping(summary.get("settings"), f"{method} settings")
    for record in records:
        if not _settings_match(record.get("settings"), summary_settings):
            raise ValueError(f"{method} evaluation settings do not match its summary.")
    cached_winner = min(records, key=_record_key)
    winner = _mapping(summary.get("winner"), f"{method} winner")
    _require_equal(
        cached_winner.get("layout_id"), winner.get("layout_id"), f"{method} winner"
    )
    digest = _file_sha256(path)
    if method == "random":
        _require_equal(summary.get("evaluations_sha256"), digest, "random evaluations hash")
        ranks = sorted(int(record.get("sample_rank", -1)) for record in records)
        _require_equal(ranks, list(range(expected_count)), "random sample ranks")
        space_indices = {int(record.get("space_index", -1)) for record in records}
        if len(space_indices) != expected_count or min(space_indices) < 0:
            raise ValueError("Random evaluations have invalid canonical space indices.")
        elapsed = [
            float(record.get("search_elapsed_seconds_at_completion", -1.0))
            for record in sorted(records, key=lambda item: int(item["sample_rank"]))
        ]
        if any(value < 0.0 for value in elapsed) or elapsed != sorted(elapsed):
            raise ValueError("Random evaluation completion times are invalid.")
    return path.resolve(), digest, records


def _source_packet(
    *, summary_path: Path, summary: Mapping[str, object], method: str
) -> tuple[dict[str, object], list[Mapping[str, object]]]:
    evaluations_path, evaluations_sha256, records = _validate_evaluations_file(
        summary_path=summary_path, summary=summary, method=method
    )
    return (
        {
            "summary_path": str(summary_path.resolve()),
            "summary_sha256": _file_sha256(summary_path),
            "evaluations_path": str(evaluations_path),
            "evaluations_sha256": evaluations_sha256,
        },
        records,
    )


def _method_packet(
    summary: Mapping[str, object], *, source: Mapping[str, object], method: str
) -> dict[str, object]:
    winner = _mapping(summary["winner"], "winner")
    validation = _mapping(winner["validation"], "winner.validation")
    holdout = _mapping(winner["holdout"], "winner.holdout")
    elapsed_field = (
        "elapsed_seconds_this_invocation"
        if method == "ga"
        else "search_elapsed_seconds"
    )
    return {
        "source": dict(source),
        "elapsed_seconds": float(summary[elapsed_field]),
        "elapsed_source_field": elapsed_field,
        "evaluated_layouts": int(summary["unique_layouts_evaluated"]),
        "layout_id": str(winner["layout_id"]),
        "layout_index": int(winner["layout_index"]),
        "layout": winner["layout"],
        "thresholds": validation.get("thresholds", {}),
        "active_slots": validation.get("active_slots", []),
        "pruned_slots": validation.get("pruned_slots", []),
        "validation": {
            "accuracy": float(validation["accuracy"]),
            "expected_cost_ms": float(validation["expected_cost"]),
            "feasible": bool(validation["feasible"]),
            "route_counts": validation.get("route_counts", {}),
        },
        "holdout": {
            "accuracy": float(holdout["accuracy"]),
            "expected_cost_ms": float(holdout["expected_cost"]),
            "feasible": bool(holdout["feasible"]),
            "route_counts": holdout.get("route_counts", {}),
        },
    }


def _deadline_censored_random_winner(
    records: Sequence[Mapping[str, object]], budget_seconds: float
) -> tuple[Mapping[str, object] | None, int, int]:
    included = [
        record
        for record in records
        if float(record.get("search_elapsed_seconds_at_completion", math.inf))
        <= budget_seconds
    ]
    winner = min(included, key=_record_key) if included else None
    return winner, len(included), len(records) - len(included)


def _phase_reference_path(method_dir: Path) -> Path:
    return method_dir / "harness_phase.json"


def _write_phase_reference(
    *,
    method_dir: Path,
    phase: str,
    seed: int,
    source_summary: Path,
    parallel_trials: int | None,
    phase_trial_count: int | None,
    wrapper_wall_seconds: float | None,
    wrapper_process_cpu_seconds: float | None,
    invocation_kind: str,
) -> Mapping[str, object]:
    packet: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "seed": seed,
        "source_summary": str(source_summary.resolve()),
        "source_summary_sha256": _file_sha256(source_summary),
        "parallel_trials_requested": parallel_trials,
        "phase_trial_count": phase_trial_count,
        "effective_phase_worker_limit": (
            min(parallel_trials, phase_trial_count)
            if parallel_trials is not None
            and phase_trial_count is not None
            and phase_trial_count > 0
            else None
        ),
        "internal_layout_workers": 1,
        "invocation_kind": invocation_kind,
        "wrapper_wall_seconds": wrapper_wall_seconds,
        "wrapper_process_cpu_seconds": wrapper_process_cpu_seconds,
        "timing_note": (
            "Wall time was measured while independent trials could execute concurrently."
            if parallel_trials is not None and parallel_trials > 1
            else "Wall time was measured without harness-level concurrency."
        ),
    }
    _write_json_atomic(_phase_reference_path(method_dir), packet)
    return packet


def _load_phase_reference(method_dir: Path, source_summary: Path) -> Mapping[str, object]:
    path = _phase_reference_path(method_dir)
    if path.exists():
        packet = _read_json(path, "harness phase reference")
        _require_equal(
            packet.get("source_summary_sha256"),
            _file_sha256(source_summary),
            "phase source summary hash",
        )
        return packet
    return _write_phase_reference(
        method_dir=method_dir,
        phase="recovered",
        seed=-1,
        source_summary=source_summary,
        parallel_trials=None,
        phase_trial_count=None,
        wrapper_wall_seconds=None,
        wrapper_process_cpu_seconds=None,
        invocation_kind="recovered_existing_complete_summary_context_unknown",
    )


def _archive_partial_ga(method_dir: Path) -> Path:
    suffix = 1
    while True:
        target = method_dir.with_name(f"{method_dir.name}_incomplete_attempt_{suffix:03d}")
        if not target.exists():
            method_dir.rename(target)
            return target
        suffix += 1


def _run_ga_worker(payload: Mapping[str, object]) -> dict[str, object]:
    seed = int(payload["seed"])
    output_dir = Path(str(payload["output_dir"]))
    outcomes = Path(str(payload["outcomes"]))
    captured = io.StringIO()
    wall_started = perf_counter()
    cpu_started = process_time()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            run_k1_search(
                outcomes=outcomes,
                output_dir=output_dir,
                target_accuracy=TARGET_ACCURACY,
                iterations=ITERATIONS_PER_RESTART,
                restarts=RESTARTS,
                quantile_points=QUANTILE_POINTS,
                inner_seed=INNER_SEED,
                split_seed=SPLIT_SEED,
                outer_seed=seed,
                holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
                split_strategy=DEFAULT_SPLIT_STRATEGY,
                population_size=DEFAULT_POPULATION_SIZE,
                generations=DEFAULT_GENERATIONS,
                evaluation_budget=DEFAULT_EVALUATION_BUDGET,
            )
    except Exception as error:
        tail = captured.getvalue()[-4_000:]
        raise RuntimeError(f"GA seed {seed} failed. Captured output:\n{tail}") from error
    wrapper_wall = perf_counter() - wall_started
    wrapper_cpu = process_time() - cpu_started
    summary_path = output_dir / "summary.json"
    _write_phase_reference(
        method_dir=output_dir,
        phase="genetic_algorithm",
        seed=seed,
        source_summary=summary_path,
        parallel_trials=int(payload["parallel_trials"]),
        phase_trial_count=int(payload["phase_trial_count"]),
        wrapper_wall_seconds=wrapper_wall,
        wrapper_process_cpu_seconds=wrapper_cpu,
        invocation_kind="fresh_uninterrupted_ga_required_for_time_matching",
    )
    return {"seed": seed, "summary_path": str(summary_path), "wall": wrapper_wall}


def _run_random_worker(payload: Mapping[str, object]) -> dict[str, object]:
    seed = int(payload["seed"])
    output_dir = Path(str(payload["output_dir"]))
    outcomes = Path(str(payload["outcomes"]))
    time_budget = float(payload["time_budget_seconds"])
    captured = io.StringIO()
    wall_started = perf_counter()
    cpu_started = process_time()
    try:
        with redirect_stdout(captured), redirect_stderr(captured):
            run_k1_random_search(
                outcomes=outcomes,
                output_dir=output_dir,
                target_accuracy=TARGET_ACCURACY,
                time_budget_seconds=time_budget,
                iterations=ITERATIONS_PER_RESTART,
                restarts=RESTARTS,
                quantile_points=QUANTILE_POINTS,
                inner_seed=INNER_SEED,
                sampling_seed=seed,
                split_seed=SPLIT_SEED,
                holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
                split_strategy=DEFAULT_SPLIT_STRATEGY,
                max_layouts=None,
            )
    except Exception as error:
        tail = captured.getvalue()[-4_000:]
        raise RuntimeError(
            f"Random-search seed {seed} failed. Captured output:\n{tail}"
        ) from error
    wrapper_wall = perf_counter() - wall_started
    wrapper_cpu = process_time() - cpu_started
    summary_path = output_dir / "summary.json"
    _write_phase_reference(
        method_dir=output_dir,
        phase="random_sampling",
        seed=seed,
        source_summary=summary_path,
        parallel_trials=int(payload["parallel_trials"]),
        phase_trial_count=int(payload["phase_trial_count"]),
        wrapper_wall_seconds=wrapper_wall,
        wrapper_process_cpu_seconds=wrapper_cpu,
        invocation_kind="fresh_or_cumulatively_resumed_random_search",
    )
    return {"seed": seed, "summary_path": str(summary_path), "wall": wrapper_wall}


def _execute_phase(
    *,
    name: str,
    tasks: Sequence[Mapping[str, object]],
    worker: Callable[[Mapping[str, object]], Mapping[str, object]],
    parallel_trials: int,
) -> None:
    if not tasks:
        print(f"{name}: no pending trials")
        return
    print(f"{name}: {len(tasks)} pending trial(s), parallelism={parallel_trials}")
    if parallel_trials == 1:
        for index, task in enumerate(tasks, start=1):
            result = worker(task)
            print(
                f"{name}: completed seed {result['seed']} "
                f"({index}/{len(tasks)}, wrapper wall {float(result['wall']):.1f}s)"
            )
        return
    with ProcessPoolExecutor(max_workers=parallel_trials) as executor:
        future_to_seed = {
            executor.submit(worker, task): int(task["seed"]) for task in tasks
        }
        completed = 0
        for future in as_completed(future_to_seed):
            result = future.result()
            completed += 1
            print(
                f"{name}: completed seed {result['seed']} "
                f"({completed}/{len(tasks)}, wrapper wall {float(result['wall']):.1f}s)"
            )


def _trial_summary(
    *,
    seed: int,
    contract: Mapping[str, object],
    ga_summary_path: Path,
    random_summary_path: Path,
    trial_dir: Path,
) -> Mapping[str, object]:
    outcomes_sha256 = str(contract["outcomes_sha256"])
    ga_summary = _read_json(ga_summary_path, f"GA seed {seed} summary")
    ga_winner = validate_ga_summary(
        ga_summary, seed=seed, outcomes_sha256=outcomes_sha256
    )
    ga_elapsed = float(ga_summary["elapsed_seconds_this_invocation"])
    random_summary = _read_json(random_summary_path, f"random seed {seed} summary")
    random_winner = validate_random_summary(
        random_summary,
        seed=seed,
        ga_elapsed_seconds=ga_elapsed,
        outcomes_sha256=outcomes_sha256,
    )
    ga_split = _canonical_split(ga_summary["split"], "GA split")
    random_split = _canonical_split(random_summary["split"], "random split")
    _require_equal(random_split, ga_split, "paired validation/holdout split")

    ga_source, _ = _source_packet(
        summary_path=ga_summary_path, summary=ga_summary, method="ga"
    )
    random_source, random_records = _source_packet(
        summary_path=random_summary_path, summary=random_summary, method="random"
    )
    ga_packet = _method_packet(ga_summary, source=ga_source, method="ga")
    random_packet = _method_packet(
        random_summary, source=random_source, method="random"
    )
    ga_validation = _mapping(ga_winner["validation"], "GA validation")
    random_validation = _mapping(random_winner["validation"], "random validation")
    ga_holdout = _mapping(ga_winner["holdout"], "GA holdout")
    random_holdout = _mapping(random_winner["holdout"], "random holdout")
    both_feasible = bool(ga_validation["feasible"]) and bool(
        random_validation["feasible"]
    )

    censored_winner, censored_count, overtime_count = (
        _deadline_censored_random_winner(random_records, ga_elapsed)
    )
    if censored_winner is None:
        censored_packet: dict[str, object] = {
            "available": False,
            "layouts_completed_by_deadline": 0,
            "overtime_layouts_excluded": overtime_count,
            "selection_outcome": None,
            "random_minus_ga_cost_ms": None,
        }
    else:
        censored_validation = _mapping(
            censored_winner["validation"], "deadline-censored random validation"
        )
        censored_both_feasible = bool(ga_validation["feasible"]) and bool(
            censored_validation["feasible"]
        )
        censored_packet = {
            "available": True,
            "layouts_completed_by_deadline": censored_count,
            "overtime_layouts_excluded": overtime_count,
            "layout_id": str(censored_winner["layout_id"]),
            "layout_index": int(censored_winner["layout_index"]),
            "validation": {
                "accuracy": float(censored_validation["accuracy"]),
                "expected_cost_ms": float(censored_validation["expected_cost"]),
                "feasible": bool(censored_validation["feasible"]),
            },
            "selection_outcome": paired_outcome(ga_validation, censored_validation),
            "both_feasible": censored_both_feasible,
            "random_minus_ga_cost_ms": (
                float(censored_validation["expected_cost"])
                - float(ga_validation["expected_cost"])
                if censored_both_feasible
                else None
            ),
        }

    packet: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "trial_seed": seed,
        "seed_plan": {
            "ga_outer_seed": seed,
            "random_sampling_seed": seed,
            "fixed_inner_seed": INNER_SEED,
            "fixed_restart_seeds": list(range(RESTARTS)),
            "fixed_split_seed": SPLIT_SEED,
        },
        "contract": dict(contract),
        "methods": {
            "genetic_algorithm": ga_packet,
            "random_sampling": random_packet,
        },
        "timing": {
            "budget_source": "ga.elapsed_seconds_this_invocation",
            "ga_reported_seconds": ga_elapsed,
            "random_budget_seconds": float(
                _mapping(random_summary["settings"], "random settings")[
                    "time_budget_seconds"
                ]
            ),
            "random_search_elapsed_seconds": float(
                random_summary["search_elapsed_seconds"]
            ),
            "random_atomic_overshoot_seconds": float(
                random_summary["time_budget_overshoot_seconds"]
            ),
            "ga_phase": dict(_load_phase_reference(trial_dir / "ga", ga_summary_path)),
            "random_phase": dict(
                _load_phase_reference(trial_dir / "random", random_summary_path)
            ),
            "fairness_note": (
                "GA elapsed excludes its winner-only holdout replay. Random receives "
                "that GA validation-search duration; its raw result may include one "
                "atomic layout completed just after the deadline."
            ),
        },
        "primary_validation_comparison": {
            "selection_outcome": paired_outcome(ga_validation, random_validation),
            "both_feasible": both_feasible,
            "random_minus_ga_cost_ms": (
                float(random_validation["expected_cost"])
                - float(ga_validation["expected_cost"])
                if both_feasible
                else None
            ),
            "raw_random_minus_ga_cost_ms_descriptive": (
                float(random_validation["expected_cost"])
                - float(ga_validation["expected_cost"])
            ),
            "random_minus_ga_accuracy": (
                float(random_validation["accuracy"])
                - float(ga_validation["accuracy"])
            ),
        },
        "deadline_censored_random_validation_secondary": {
            "method": "exclude_random_layouts_completed_after_paired_GA_budget",
            "holdout_used": False,
            **censored_packet,
        },
        "holdout_descriptive": {
            "not_used_for_selection": True,
            "random_minus_ga_cost_ms": (
                float(random_holdout["expected_cost"])
                - float(ga_holdout["expected_cost"])
            ),
            "random_minus_ga_accuracy": (
                float(random_holdout["accuracy"]) - float(ga_holdout["accuracy"])
            ),
        },
        "interpretation_limitations": [
            contract["search_asymmetry"],
            contract["oracle_limitation"],
            "Ten outer seeds provide limited power; holdout results are descriptive only.",
        ],
    }
    trial_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(trial_dir / "summary.json", packet)
    return packet


def summarize_trials(
    trials: Sequence[Mapping[str, object]],
    *,
    contract: Mapping[str, object],
    orchestration: Mapping[str, object],
) -> dict[str, object]:
    outcomes = {"ga_better": 0, "tie": 0, "random_better": 0}
    censored_outcomes = {"ga_better": 0, "tie": 0, "random_better": 0}
    feasible_deltas: list[float] = []
    raw_deltas: list[float] = []
    holdout_cost_deltas: list[float] = []
    holdout_accuracy_deltas: list[float] = []
    censored_deltas: list[float] = []
    overtime_counts: list[float] = []
    ga_costs: list[float] = []
    random_costs: list[float] = []
    ga_layout_counts: list[float] = []
    random_layout_counts: list[float] = []
    ga_elapsed: list[float] = []
    random_elapsed: list[float] = []
    for trial in trials:
        primary = _mapping(trial["primary_validation_comparison"], "primary")
        outcomes[str(primary["selection_outcome"])] += 1
        if primary.get("random_minus_ga_cost_ms") is not None:
            feasible_deltas.append(float(primary["random_minus_ga_cost_ms"]))
        raw_deltas.append(float(primary["raw_random_minus_ga_cost_ms_descriptive"]))
        methods = _mapping(trial["methods"], "methods")
        ga = _mapping(methods["genetic_algorithm"], "GA")
        random = _mapping(methods["random_sampling"], "random")
        ga_costs.append(float(_mapping(ga["validation"], "GA validation")["expected_cost_ms"]))
        random_costs.append(
            float(_mapping(random["validation"], "random validation")["expected_cost_ms"])
        )
        ga_layout_counts.append(float(ga["evaluated_layouts"]))
        random_layout_counts.append(float(random["evaluated_layouts"]))
        ga_elapsed.append(float(ga["elapsed_seconds"]))
        random_elapsed.append(float(random["elapsed_seconds"]))
        holdout = _mapping(trial["holdout_descriptive"], "holdout")
        holdout_cost_deltas.append(float(holdout["random_minus_ga_cost_ms"]))
        holdout_accuracy_deltas.append(float(holdout["random_minus_ga_accuracy"]))
        censored = _mapping(
            trial["deadline_censored_random_validation_secondary"], "censored"
        )
        overtime_counts.append(float(censored["overtime_layouts_excluded"]))
        if bool(censored["available"]):
            censored_outcomes[str(censored["selection_outcome"])] += 1
            if censored.get("random_minus_ga_cost_ms") is not None:
                censored_deltas.append(float(censored["random_minus_ga_cost_ms"]))

    trial_refs = [
        {
            "trial_seed": int(trial["trial_seed"]),
            "summary": f"trial_{int(trial['trial_seed']):02d}/summary.json",
        }
        for trial in trials
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": len(trials) == len(TRIAL_SEEDS),
        "completed_trials": len(trials),
        "expected_trials": len(TRIAL_SEEDS),
        "contract": dict(contract),
        "orchestration": dict(orchestration),
        "primary_validation_comparison": {
            "metric": "feasibility-aware selection; feasible cost delta is random-GA",
            "positive_feasible_cost_delta_favors": "genetic_algorithm",
            "paired_outcomes": outcomes,
            "exact_two_sided_sign_test": {
                "ties_excluded": outcomes["tie"],
                "non_tied_trials": outcomes["ga_better"] + outcomes["random_better"],
                "p_value": exact_two_sided_sign_test_p_value(
                    outcomes["ga_better"], outcomes["random_better"]
                ),
            },
            "feasible_random_minus_ga_cost_ms": _numeric_summary(feasible_deltas),
            "bootstrap_95pct_mean_feasible_delta_ms": _bootstrap_mean_interval(
                feasible_deltas
            ),
            "raw_random_minus_ga_cost_ms_descriptive": _numeric_summary(raw_deltas),
            "ga_validation_cost_ms": _numeric_summary(ga_costs),
            "random_validation_cost_ms": _numeric_summary(random_costs),
        },
        "search_effort": {
            "ga_evaluated_layouts": _numeric_summary(ga_layout_counts),
            "random_evaluated_layouts": _numeric_summary(random_layout_counts),
            "ga_elapsed_seconds": _numeric_summary(ga_elapsed),
            "random_elapsed_seconds": _numeric_summary(random_elapsed),
        },
        "deadline_censored_random_validation_secondary": {
            "method": "exclude_random_layouts_completed_after_paired_GA_budget",
            "available_trials": sum(censored_outcomes.values()),
            "paired_outcomes": censored_outcomes,
            "exact_two_sided_sign_test": {
                "ties_excluded": censored_outcomes["tie"],
                "p_value": exact_two_sided_sign_test_p_value(
                    censored_outcomes["ga_better"],
                    censored_outcomes["random_better"],
                ),
            },
            "feasible_random_minus_ga_cost_ms": _numeric_summary(censored_deltas),
            "overtime_layouts_excluded": _numeric_summary(overtime_counts),
        },
        "holdout_descriptive": {
            "not_used_for_selection": True,
            "random_minus_ga_cost_ms": _numeric_summary(holdout_cost_deltas),
            "random_minus_ga_accuracy": _numeric_summary(holdout_accuracy_deltas),
        },
        "oracle_comparison": {
            "available": False,
            "reason": contract["oracle_limitation"],
        },
        "assumptions_and_limitations": [
            contract["search_asymmetry"],
            contract["oracle_limitation"],
            "Trials vary outer layout-search seeds only; split and threshold-SA seeds are fixed.",
            "The exact sign test has limited power with ten trials.",
            "Concurrent independent trials can affect measured wall times.",
            "Raw random results may include one deadline-crossing atomic layout; a censored sensitivity is reported.",
            "Holdout outcomes are descriptive and never select a winner.",
        ],
        "trials": trial_refs,
    }


def run_benchmark(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    parallel_trials: int = 1,
) -> dict[str, object]:
    if parallel_trials < 1:
        raise ValueError("parallel_trials must be at least 1.")
    contract = experiment_contract(outcomes)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    if contract_path.exists():
        _require_equal(_read_json(contract_path, "existing contract"), contract, "contract")
    else:
        _write_json_atomic(contract_path, contract)

    outcomes_sha256 = str(contract["outcomes_sha256"])
    ga_tasks: list[dict[str, object]] = []
    for seed in TRIAL_SEEDS:
        trial_dir = output_dir / f"trial_{seed:02d}"
        ga_dir = trial_dir / "ga"
        trial_dir.mkdir(parents=True, exist_ok=True)
        summary_path = ga_dir / "summary.json"
        if summary_path.exists():
            validate_ga_summary(
                _read_json(summary_path, f"GA seed {seed}"),
                seed=seed,
                outcomes_sha256=outcomes_sha256,
            )
            continue
        if ga_dir.exists() and any(ga_dir.iterdir()):
            archived = _archive_partial_ga(ga_dir)
            print(f"GA seed {seed}: archived incomplete attempt to {archived}")
        ga_tasks.append(
            {
                "seed": seed,
                "output_dir": str(ga_dir),
                "outcomes": str(outcomes),
                "parallel_trials": parallel_trials,
                "phase_trial_count": 0,
            }
        )
    for task in ga_tasks:
        task["phase_trial_count"] = len(ga_tasks)
    _execute_phase(
        name="GA phase",
        tasks=ga_tasks,
        worker=_run_ga_worker,
        parallel_trials=min(parallel_trials, max(1, len(ga_tasks))),
    )

    random_tasks: list[dict[str, object]] = []
    for seed in TRIAL_SEEDS:
        trial_dir = output_dir / f"trial_{seed:02d}"
        ga_path = trial_dir / "ga" / "summary.json"
        ga_summary = _read_json(ga_path, f"GA seed {seed}")
        validate_ga_summary(
            ga_summary, seed=seed, outcomes_sha256=outcomes_sha256
        )
        ga_elapsed = float(ga_summary["elapsed_seconds_this_invocation"])
        random_dir = trial_dir / "random"
        random_path = random_dir / "summary.json"
        if random_path.exists():
            validate_random_summary(
                _read_json(random_path, f"random seed {seed}"),
                seed=seed,
                ga_elapsed_seconds=ga_elapsed,
                outcomes_sha256=outcomes_sha256,
            )
            continue
        random_tasks.append(
            {
                "seed": seed,
                "output_dir": str(random_dir),
                "outcomes": str(outcomes),
                "time_budget_seconds": ga_elapsed,
                "parallel_trials": parallel_trials,
                "phase_trial_count": 0,
            }
        )
    for task in random_tasks:
        task["phase_trial_count"] = len(random_tasks)
    _execute_phase(
        name="Random phase",
        tasks=random_tasks,
        worker=_run_random_worker,
        parallel_trials=min(parallel_trials, max(1, len(random_tasks))),
    )

    trial_packets = [
        _trial_summary(
            seed=seed,
            contract=contract,
            ga_summary_path=output_dir / f"trial_{seed:02d}" / "ga" / "summary.json",
            random_summary_path=(
                output_dir / f"trial_{seed:02d}" / "random" / "summary.json"
            ),
            trial_dir=output_dir / f"trial_{seed:02d}",
        )
        for seed in TRIAL_SEEDS
    ]
    orchestration = {
        "phase_barrier": "all_pending_GA_then_all_pending_random",
        "parallel_trials_requested": parallel_trials,
        "ga_phase_pending_trials": len(ga_tasks),
        "random_phase_pending_trials": len(random_tasks),
        "internal_layout_workers": 1,
        "worker_progress_output": "suppressed",
        "thread_limits": {
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "timing_interpretation": (
            "Wrapper wall and process CPU time are recorded per invocation. "
            "Random budgets use GA-reported validation-search wall time."
        ),
    }
    aggregate = summarize_trials(
        trial_packets, contract=contract, orchestration=orchestration
    )
    for reference in aggregate["trials"]:
        assert isinstance(reference, dict)
        summary_path = output_dir / str(reference["summary"])
        reference["summary_sha256"] = _file_sha256(summary_path)
    _write_json_atomic(output_dir / "summary.json", aggregate)
    return aggregate


def dry_run_report(*, output_dir: Path, parallel_trials: int) -> dict[str, object]:
    if parallel_trials < 1:
        raise ValueError("parallel_trials must be at least 1.")
    sequential_seconds = len(TRIAL_SEEDS) * 2.0 * PRIOR_SINGLE_TRIAL_GA_SECONDS
    idealized_seconds = (
        math.ceil(len(TRIAL_SEEDS) / parallel_trials)
        * 2.0
        * PRIOR_SINGLE_TRIAL_GA_SECONDS
    )
    return {
        "action": "dry_run_only_no_files_or_models_loaded",
        "output_dir": str(output_dir.resolve()),
        "trial_seeds": list(TRIAL_SEEDS),
        "parallel_trials": parallel_trials,
        "phase_order": ["all_GA", "all_random"],
        "conditions": {
            "target_accuracy": TARGET_ACCURACY,
            "K1_enabled": True,
            "layout_space_size": LAYOUT_SPACE_SIZE,
            "SA": "best-of-10 x 1,000 iterations, fixed restart seeds 0..9",
            "GA_layout_evaluations": DEFAULT_EVALUATION_BUDGET,
            "random_budget": "paired GA reported validation-search wall seconds",
        },
        "runtime_basis": {
            "historical_single_K1_GA_seconds": PRIOR_SINGLE_TRIAL_GA_SECONDS,
            "historical_artifact_is_provenance_incompatible_and_not_reused": True,
        },
        "estimated_sequential_hours": sequential_seconds / 3_600.0,
        "idealized_parallel_lower_bound_hours": idealized_seconds / 3_600.0,
        "runtime_warning": (
            "Concurrent CPU and memory contention can substantially exceed the idealized bound."
        ),
        "search_asymmetry": (
            "GA has a mandatory K3->detector seed and a component-length sampler "
            "that overweights shorter layouts; random is unseeded and exactly uniform."
        ),
        "oracle_limitation": "No exhaustive K1-enabled reference exists.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=1,
        help=(
            "Independent trial processes per phase. All GA trials finish before "
            "random trials; every optimizer remains internally single-worker."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.parallel_trials < 1:
        raise SystemExit("--parallel-trials must be at least 1")
    if args.dry_run:
        print(
            json.dumps(
                dry_run_report(
                    output_dir=args.output_dir, parallel_trials=args.parallel_trials
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    summary = run_benchmark(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        parallel_trials=args.parallel_trials,
    )
    primary = _mapping(summary["primary_validation_comparison"], "primary")
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Paired outcomes: {primary['paired_outcomes']}")
    print(
        "Exact sign-test p-value: "
        f"{_mapping(primary['exact_two_sided_sign_test'], 'sign test')['p_value']:.6g}"
    )


if __name__ == "__main__":  # Required for Windows ProcessPool spawn safety.
    main()
