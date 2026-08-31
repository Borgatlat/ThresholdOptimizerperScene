"""Paired GA-versus-random joint-search benchmark for h24 at p=0.9662.

This benchmark deliberately freezes the threshold-optimization fitness landscape:
all layouts use the same validation split and the same ten continuous-SA restart
seeds.  Trials 0 through 9 vary only the outer GA seed and the uniform random
layout-order seed.  The actual primary comparison gives random search the wall
time reported by its paired 512-layout GA run.  A completed exhaustive search is
used only after both winners are frozen, both to report ranks and to reconstruct
the validation-only random-search result at an equal 512-layout budget.

The comparison is not between equally seeded layout samplers.  The memetic GA's
starting population always contains K3 -> detector, while uniform random search
has no mandatory layout.  That a-priori seeding asymmetry is recorded explicitly
in every report.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from statistics import mean, median, pstdev
from time import perf_counter, process_time
from typing import Callable, Mapping, Sequence

import numpy as np

from experiments.m3n_vc.brute_force_k1_free_layouts import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_OUTCOMES,
    DEFAULT_SPLIT_STRATEGY,
    EXPECTED_LAYOUT_COUNT,
    REMOVED_CANDIDATES,
    IndexedLayout,
    _cascade_payload,
    _layout_selection_key,
    enumerate_k1_free_layouts,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    _file_sha256,
    _fitness_implementation_sha256,
    _settings_match,
    _write_json_atomic,
    run_joint_search,
)
from experiments.m3n_vc.random_joint_optimize_hierarchy import (
    _order_sha256,
    run_random_search,
    uniform_layout_order,
)
from hierarchy_optimizer import PAPER_DETECTOR_COST_MS
from threshold_optimizer import DEFAULT_SA_RESTARTS


SCHEMA_VERSION = "paired-ga-vs-random-k1-free/v1"
TARGET_ACCURACY = 0.9662
TRIAL_SEEDS = tuple(range(10))
INNER_SEED = 0
SPLIT_SEED = 0
ITERATIONS_PER_RESTART = 1_000
RESTARTS = 10
QUANTILE_POINTS = 50
RANDOM_EQUAL_EVALUATION_COUNT = 512
COMPARISON_TOLERANCE_MS = 1e-9

DEFAULT_OUTPUT_DIR = Path(
    "checkpoints/ga_vs_random_k1_free_h24_target_0962_paper_sa_10_trials"
)
DEFAULT_ORACLE_SUMMARY = Path(
    "checkpoints/k1_free_full_benchmark_h24_target_0962_paper_sa/summary.json"
)
DEFAULT_ORACLE_RESULTS = Path(
    "checkpoints/k1_free_full_benchmark_h24_target_0962_paper_sa/"
    "layout_results.jsonl"
)


@dataclass(frozen=True)
class Oracle:
    summary_path: Path
    results_path: Path
    summary_sha256: str
    results_sha256: str
    records: Mapping[str, Mapping[str, object]]
    rank_by_id: Mapping[str, int]
    optimum: Mapping[str, object]
    split: Mapping[str, object]
    outcomes_sha256: str


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


def _assert_k1_free(value: object, label: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _assert_k1_free(key, label)
            _assert_k1_free(nested, label)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _assert_k1_free(nested, label)
        return
    if isinstance(value, str) and (
        value == "K1"
        or value.startswith("K1@")
        or value.startswith("K1:")
        or "[K1:" in value
    ):
        raise ValueError(f"{label} unexpectedly contains removed classifier K1.")


def _canonical_split(value: object, label: str) -> dict[str, object]:
    split = _mapping(value, label)
    per_run = split.get("per_run", split.get("per_group"))
    return {
        "strategy": split.get("strategy"),
        "random_seed": split.get("random_seed"),
        "holdout_fraction": split.get("holdout_fraction"),
        "validation_samples": split.get("validation_samples"),
        "holdout_samples": split.get("holdout_samples"),
        "per_run": per_run,
    }


def _threshold_contract(settings: Mapping[str, object], label: str) -> None:
    optimizer = _mapping(settings.get("threshold_optimizer"), label)
    expected = {
        "method": "best_of_10_chellapilla_continuous_gaussian_sa",
        "iterations_per_restart": ITERATIONS_PER_RESTART,
        "restarts": RESTARTS,
        "restart_seeds": list(range(RESTARTS)),
        "prune_stages_accepting_zero_validation_samples": True,
        "freeze_validation_active_slots_on_holdout": True,
    }
    for key, value in expected.items():
        _require_equal(optimizer.get(key), value, f"{label}.{key}")
    if "continuous_thresholds" in optimizer:
        _require_equal(
            optimizer["continuous_thresholds"],
            True,
            f"{label}.continuous_thresholds",
        )
    if "quantile_points_used" in optimizer:
        _require_equal(
            optimizer["quantile_points_used"],
            False,
            f"{label}.quantile_points_used",
        )


def _experiment_contract(
    *, outcomes: Path, catalogue: Sequence[IndexedLayout]
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "m3n_vc/h24",
        "target_accuracy": TARGET_ACCURACY,
        "layout_space_size": EXPECTED_LAYOUT_COUNT,
        "layout_catalogue_sha256": _indexed_catalogue_sha256(catalogue),
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "removed_candidates": ["K1"],
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
            "fitness_implementation_sha256": _fitness_implementation_sha256(),
        },
        "trial_seeds": list(TRIAL_SEEDS),
        "varied_seed_fields": {
            "genetic_algorithm": "outer_seed",
            "random_sampling": "sampling_seed",
        },
        "fixed_seed_fields": {"inner_seed": INNER_SEED, "split_seed": SPLIT_SEED},
        "genetic_algorithm": {
            "algorithm": "constrained_memetic_genetic_algorithm",
            "population_size": 32,
            "generations": 24,
            "evaluation_budget": 512,
            "elite_count": 4,
            "tournament_size": 2,
            "crossover_rate": 0.8,
            "mutation_rate": 0.8,
            "random_immigrant_rate": 0.2,
            "component_resample_rate": 0.3,
            "stagnation_generations": 6,
            "max_restarts": 3,
            "outer_schedule": "fixed",
            "internal_workers": 1,
            "mandatory_initial_member": {
                "initial": ["K3", "detector"],
                "specialized": {},
            },
        },
        "random_sampling": {
            "algorithm": "uniform_random_layout_sampling_without_replacement",
            "mandatory_initial_member": None,
            "time_budget": "paired_ga_reported_elapsed_seconds_this_invocation",
            "atomic_layout_overshoot_allowed": True,
        },
        "primary_comparison": {
            "scope": "validation_only",
            "selection_rule": "feasibility_aware_topology_selection",
            "cost_delta": "random_minus_ga_ms_positive_favors_ga",
        },
        "equal_evaluation_secondary": {
            "scope": "validation_only_posthoc_exhaustive_oracle_replay",
            "random_layout_prefix": RANDOM_EQUAL_EVALUATION_COUNT,
        },
        "search_asymmetry": (
            "The GA is seeded with mandatory K3->detector while uniform random "
            "sampling is unseeded. Results therefore compare the implemented "
            "search procedures, including that a-priori GA advantage."
        ),
    }


def _indexed_catalogue_sha256(catalogue: Sequence[IndexedLayout]) -> str:
    digest = hashlib.sha256()
    for entry in catalogue:
        digest.update(f"{entry.index}:{entry.layout_id}\n".encode("ascii"))
    return digest.hexdigest()


def _validate_oracle_settings(
    summary: Mapping[str, object], outcomes_sha256: str
) -> Mapping[str, object]:
    settings = _mapping(summary.get("settings"), "oracle.settings")
    expected = {
        "schema_version": "k1-free-optimizer-benchmark/v2",
        "dataset": "m3n_vc/h24",
        "target_accuracy": TARGET_ACCURACY,
        "selected_layout_count": EXPECTED_LAYOUT_COUNT,
        "expected_layout_count": EXPECTED_LAYOUT_COUNT,
        "removed_candidates": ["K1"],
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "split_seed": SPLIT_SEED,
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
        "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
        "outcomes_sha256": outcomes_sha256,
    }
    for key, value in expected.items():
        if isinstance(value, float):
            _require_close(settings.get(key), value, f"oracle.settings.{key}")
        else:
            _require_equal(settings.get(key), value, f"oracle.settings.{key}")
    _threshold_contract(settings, "oracle.settings.threshold_optimizer")
    _require_equal(summary.get("target_accuracy"), TARGET_ACCURACY, "oracle target")
    return settings


def load_and_validate_oracle(
    *,
    summary_path: Path,
    results_path: Path,
    outcomes: Path,
    catalogue: Sequence[IndexedLayout],
) -> Oracle:
    """Load the completed exhaustive validation oracle and verify its provenance."""

    outcomes_sha256 = _file_sha256(outcomes)
    summary = _read_json(summary_path, "oracle summary")
    _validate_oracle_settings(summary, outcomes_sha256)
    split = _canonical_split(summary.get("split"), "oracle.split")
    expected_split = {
        "strategy": DEFAULT_SPLIT_STRATEGY,
        "random_seed": SPLIT_SEED,
        "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
    }
    for key, expected in expected_split.items():
        _require_equal(split.get(key), expected, f"oracle.split.{key}")

    methods = _mapping(summary.get("methods"), "oracle.methods")
    exhaustive = _mapping(methods.get("exhaustive_joint"), "oracle exhaustive")
    _require_equal(
        exhaustive.get("completed_layouts"),
        EXPECTED_LAYOUT_COUNT,
        "oracle completed layouts",
    )
    results_sha256 = _file_sha256(results_path)
    _require_equal(
        exhaustive.get("layout_results_sha256"),
        results_sha256,
        "oracle results SHA-256",
    )

    raw_records = _read_jsonl(results_path, "oracle layout results")
    if len(raw_records) != EXPECTED_LAYOUT_COUNT:
        raise ValueError(
            f"Oracle has {len(raw_records):,} layouts; expected {EXPECTED_LAYOUT_COUNT:,}."
        )
    indexed_by_id = {item.layout_id: item for item in catalogue}
    records: dict[str, Mapping[str, object]] = {}
    seen_indices: set[int] = set()
    for record in raw_records:
        layout_id = str(record.get("layout_id"))
        layout_index = int(record.get("layout_index", -1))
        if layout_id in records or layout_index in seen_indices:
            raise ValueError("Oracle contains duplicate layout IDs or indices.")
        indexed = indexed_by_id.get(layout_id)
        if indexed is None or indexed.index != layout_index:
            raise ValueError("Oracle layout identity does not match the catalogue.")
        _require_equal(record.get("layout"), _cascade_payload(indexed.cascade), "oracle layout")
        validation = _mapping(record.get("validation"), "oracle validation")
        _require_close(
            validation.get("target_accuracy"), TARGET_ACCURACY, "oracle target"
        )
        _assert_k1_free(record, f"oracle layout {layout_id}")
        records[layout_id] = record
        seen_indices.add(layout_index)
    if seen_indices != set(range(EXPECTED_LAYOUT_COUNT)):
        raise ValueError("Oracle layout indices are not complete.")

    ranked = sorted(records.values(), key=_layout_selection_key)
    rank_by_id = {
        str(record["layout_id"]): rank
        for rank, record in enumerate(ranked, start=1)
    }
    optimum = ranked[0]
    _require_equal(
        exhaustive.get("best_layout_id"), optimum.get("layout_id"), "oracle winner"
    )
    _require_equal(
        exhaustive.get("best_layout_index"),
        optimum.get("layout_index"),
        "oracle winner index",
    )
    return Oracle(
        summary_path=summary_path.resolve(),
        results_path=results_path.resolve(),
        summary_sha256=_file_sha256(summary_path),
        results_sha256=results_sha256,
        records=records,
        rank_by_id=rank_by_id,
        optimum=optimum,
        split=split,
        outcomes_sha256=outcomes_sha256,
    )


def _validate_common_summary(
    summary: Mapping[str, object], *, oracle: Oracle, label: str
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    settings = _mapping(summary.get("settings"), f"{label}.settings")
    expected = {
        "target_accuracy": TARGET_ACCURACY,
        "removed_candidates": ["K1"],
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "outcomes_sha256": oracle.outcomes_sha256,
        "split_seed": SPLIT_SEED,
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
    }
    for key, value in expected.items():
        if isinstance(value, float):
            _require_close(settings.get(key), value, f"{label}.settings.{key}")
        else:
            _require_equal(settings.get(key), value, f"{label}.settings.{key}")
    _threshold_contract(settings, f"{label}.settings.threshold_optimizer")
    _require_equal(
        settings.get("fitness_implementation_sha256"),
        _fitness_implementation_sha256(),
        f"{label} fitness implementation",
    )
    split = _canonical_split(summary.get("split"), f"{label}.split")
    _require_equal(split, dict(oracle.split), f"{label}.split")
    winner = _mapping(summary.get("winner"), f"{label}.winner")
    _assert_k1_free(winner.get("layout"), f"{label}.winner.layout")
    layout_id = str(winner.get("layout_id"))
    oracle_record = oracle.records.get(layout_id)
    if oracle_record is None:
        raise ValueError(f"{label} winner is not in the exhaustive oracle.")
    _require_equal(winner.get("layout_index"), oracle_record.get("layout_index"), label)
    _require_equal(winner.get("layout"), oracle_record.get("layout"), label)
    validation = _mapping(winner.get("validation"), f"{label}.winner.validation")
    _assert_k1_free(validation, f"{label}.winner.validation")
    oracle_validation = _mapping(
        oracle_record.get("validation"), "oracle winner validation"
    )
    for key in ("accuracy", "expected_cost"):
        _require_close(
            validation.get(key),
            float(oracle_validation[key]),
            f"{label}.winner.validation.{key}",
        )
    _require_equal(
        bool(validation.get("feasible")),
        bool(oracle_validation.get("feasible")),
        f"{label}.winner.validation.feasible",
    )
    _mapping(winner.get("holdout"), f"{label}.winner.holdout")
    _require_equal(
        summary.get("holdout_usage"),
        "winner_only_after_validation_search",
        f"{label}.holdout_usage",
    )
    return settings, winner


def validate_ga_summary(
    summary: Mapping[str, object], *, seed: int, oracle: Oracle
) -> Mapping[str, object]:
    settings, winner = _validate_common_summary(
        summary, oracle=oracle, label=f"GA seed {seed}"
    )
    expected = {
        "algorithm": "constrained_memetic_genetic_algorithm",
        "iterations": ITERATIONS_PER_RESTART,
        "quantile_points": QUANTILE_POINTS,
        "inner_seed": INNER_SEED,
        "outer_seed": seed,
        "population_size": 32,
        "generations": 24,
        "evaluation_budget": 512,
        "elite_count": 4,
        "tournament_size": 2,
        "crossover_rate": 0.8,
        "mutation_rate": 0.8,
        "random_immigrant_rate": 0.2,
        "component_resample_rate": 0.3,
        "stagnation_generations": 6,
        "max_restarts": 3,
        "outer_parameter_schedule": "fixed",
        "annealed_outer_schedule": None,
        "layout_catalogue_sha256": _indexed_catalogue_sha256(
            tuple(enumerate_k1_free_layouts())
        ),
    }
    for key, value in expected.items():
        _require_equal(settings.get(key), value, f"GA seed {seed}.{key}")
    _require_equal(
        summary.get("unique_layouts_evaluated"), 512, "GA evaluated layouts"
    )
    _require_equal(summary.get("layout_space_size"), 5545, "GA layout space")
    _require_equal(summary.get("stop_reason"), "evaluation_budget", "GA stop")
    elapsed = float(summary.get("elapsed_seconds_this_invocation", -1.0))
    if elapsed <= 0.0:
        raise ValueError("GA reported elapsed time must be positive.")
    return winner


def validate_random_summary(
    summary: Mapping[str, object],
    *,
    seed: int,
    ga_elapsed_seconds: float,
    oracle: Oracle,
    catalogue: Sequence[IndexedLayout],
) -> Mapping[str, object]:
    settings, winner = _validate_common_summary(
        summary, oracle=oracle, label=f"random seed {seed}"
    )
    expected = {
        "schema_version": "random-joint-layout-search/v1",
        "algorithm": "uniform_random_layout_sampling_without_replacement",
        "dataset": "m3n_vc/h24",
        "layout_space_size": 5545,
        "sampling_seed": seed,
        "layout_order_sha256": _order_sha256(
            uniform_layout_order(catalogue, seed)
        ),
        "max_layouts": None,
    }
    for key, value in expected.items():
        _require_equal(settings.get(key), value, f"random seed {seed}.{key}")
    _require_close(
        settings.get("time_budget_seconds"),
        ga_elapsed_seconds,
        f"random seed {seed} time budget",
    )
    count = int(summary.get("unique_layouts_evaluated", 0))
    if not 1 <= count <= EXPECTED_LAYOUT_COUNT:
        raise ValueError("Random evaluated-layout count is invalid.")
    if count < EXPECTED_LAYOUT_COUNT:
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


def _validate_evaluations_file(
    *,
    summary_path: Path,
    summary: Mapping[str, object],
    method: str,
) -> tuple[Path, str]:
    if method == "ga":
        path = summary_path.parent / "evaluations.jsonl"
        expected_count = int(summary["unique_layouts_evaluated"])
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
    summary_settings = _mapping(summary.get("settings"), f"{method} summary settings")
    for record in records:
        if not _settings_match(record.get("settings"), summary_settings):
            raise ValueError(f"{method} evaluation settings do not match its summary.")
        _assert_k1_free(record.get("layout"), f"{method} evaluation layout")
        _assert_k1_free(
            record.get("validation"), f"{method} evaluation validation"
        )
    winner = _mapping(summary.get("winner"), f"{method} winner")
    best = min(records, key=_layout_selection_key)
    _require_equal(
        best.get("layout_id"), winner.get("layout_id"), f"{method} cached winner"
    )
    digest = _file_sha256(path)
    if method == "random":
        _require_equal(summary.get("evaluations_sha256"), digest, "random evaluations hash")
        ranks = sorted(int(record.get("sample_rank", -1)) for record in records)
        _require_equal(ranks, list(range(expected_count)), "random sample ranks")
        seed = int(summary_settings["sampling_seed"])
        order = uniform_layout_order(tuple(enumerate_k1_free_layouts()), seed)
        by_rank = {int(record["sample_rank"]): record for record in records}
        for rank, indexed in enumerate(order[:expected_count]):
            _require_equal(
                by_rank[rank].get("layout_id"),
                indexed.layout_id,
                f"random sample rank {rank}",
            )
    return path.resolve(), digest


def _deadline_censored_random_winner(
    *, evaluations_path: Path, budget_seconds: float
) -> tuple[Mapping[str, object] | None, int, int]:
    """Select from layouts completed by the deadline, excluding overtime work."""

    records = _read_jsonl(evaluations_path, "random evaluations")
    included = [
        record
        for record in records
        if float(record.get("search_elapsed_seconds_at_completion", math.inf))
        <= budget_seconds
    ]
    winner = min(included, key=_layout_selection_key) if included else None
    return winner, len(included), len(records) - len(included)


def _source_packet(
    *,
    summary_path: Path,
    summary: Mapping[str, object],
    method: str,
    mode: str,
) -> dict[str, object]:
    evaluations_path, evaluations_sha256 = _validate_evaluations_file(
        summary_path=summary_path, summary=summary, method=method
    )
    return {
        "mode": mode,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _file_sha256(summary_path),
        "evaluations_path": str(evaluations_path),
        "evaluations_sha256": evaluations_sha256,
    }


def _policy_key(validation: Mapping[str, object]) -> tuple[float, ...]:
    feasible = bool(validation.get("feasible"))
    accuracy = float(validation["accuracy"])
    cost = float(validation["expected_cost"])
    return (
        0.0 if feasible else 1.0,
        cost if feasible else -accuracy,
        -accuracy if feasible else cost,
    )


def paired_outcome(
    ga_validation: Mapping[str, object], random_validation: Mapping[str, object]
) -> str:
    ga_key = _policy_key(ga_validation)
    random_key = _policy_key(random_validation)
    if ga_key < random_key:
        return "ga_better"
    if random_key < ga_key:
        return "random_better"
    return "tie"


def exact_two_sided_sign_test_p_value(ga_better: int, random_better: int) -> float:
    non_ties = ga_better + random_better
    if non_ties == 0:
        return 1.0
    lower_tail = min(ga_better, random_better)
    probability = sum(math.comb(non_ties, k) for k in range(lower_tail + 1))
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
    return [float(value) for value in np.quantile(resampled.mean(axis=1), (0.025, 0.975))]


def oracle_random_prefix_winner(
    *,
    oracle_records: Mapping[str, Mapping[str, object]],
    catalogue: Sequence[IndexedLayout],
    seed: int,
    count: int = RANDOM_EQUAL_EVALUATION_COUNT,
) -> Mapping[str, object]:
    if not 1 <= count <= len(catalogue):
        raise ValueError("Random oracle-prefix count is invalid.")
    order = uniform_layout_order(catalogue, seed)
    prefix = [oracle_records[indexed.layout_id] for indexed in order[:count]]
    return min(prefix, key=_layout_selection_key)


def _method_packet(
    summary: Mapping[str, object], *, source: Mapping[str, object], oracle: Oracle
) -> dict[str, object]:
    winner = _mapping(summary["winner"], "winner")
    validation = _mapping(winner["validation"], "winner.validation")
    holdout = _mapping(winner["holdout"], "winner.holdout")
    layout_id = str(winner["layout_id"])
    elapsed_field = (
        "elapsed_seconds_this_invocation"
        if "elapsed_seconds_this_invocation" in summary
        else "search_elapsed_seconds"
    )
    optimum_validation = _mapping(oracle.optimum["validation"], "oracle optimum")
    return {
        "source": dict(source),
        "elapsed_seconds": float(summary[elapsed_field]),
        "elapsed_source_field": elapsed_field,
        "evaluated_layouts": int(summary["unique_layouts_evaluated"]),
        "layout_id": layout_id,
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
            "feasible": bool(holdout.get("feasible", False)),
            "route_counts": holdout.get("route_counts", {}),
        },
        "exhaustive_rank": int(oracle.rank_by_id[layout_id]),
        "validation_regret_ms": (
            float(validation["expected_cost"])
            - float(optimum_validation["expected_cost"])
            if bool(validation["feasible"])
            else None
        ),
        "exact_optimum_recovered": layout_id == str(oracle.optimum["layout_id"]),
    }


def _phase_reference_path(method_dir: Path) -> Path:
    return method_dir / "harness_phase.json"


def _write_phase_reference(
    *,
    method_dir: Path,
    phase: str,
    seed: int,
    source_summary: Path,
    source_mode: str,
    parallel_trials: int | None,
    phase_trial_count: int | None,
    wrapper_wall_seconds: float | None,
    wrapper_process_cpu_seconds: float | None,
    invocation_kind: str,
) -> Mapping[str, object]:
    method_dir.mkdir(parents=True, exist_ok=True)
    packet: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "seed": seed,
        "source_mode": source_mode,
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
            "Reported search wall time was measured while independent trials "
            "could be executing concurrently in the same phase."
            if parallel_trials is not None and parallel_trials > 1
            else "Reported search wall time was measured without harness-level concurrency."
        ),
    }
    _write_json_atomic(_phase_reference_path(method_dir), packet)
    return packet


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
            run_joint_search(
                outcomes=outcomes,
                output_dir=output_dir,
                target_accuracy=TARGET_ACCURACY,
                iterations=ITERATIONS_PER_RESTART,
                quantile_points=QUANTILE_POINTS,
                inner_seed=INNER_SEED,
                split_seed=SPLIT_SEED,
                outer_seed=seed,
                holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
                split_strategy=DEFAULT_SPLIT_STRATEGY,
                population_size=32,
                generations=24,
                evaluation_budget=512,
                elite_count=4,
                tournament_size=2,
                crossover_rate=0.8,
                mutation_rate=0.8,
                random_immigrant_rate=0.2,
                component_resample_rate=0.3,
                stagnation_generations=6,
                max_restarts=3,
                annealed_outer_schedule=False,
                workers=1,
                brute_force_summary=None,
                brute_force_results=None,
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
        source_mode="generated",
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
            run_random_search(
                outcomes=outcomes,
                output_dir=output_dir,
                target_accuracy=TARGET_ACCURACY,
                time_budget_seconds=time_budget,
                iterations=ITERATIONS_PER_RESTART,
                restarts=RESTARTS,
                inner_seed=INNER_SEED,
                sampling_seed=seed,
                split_seed=SPLIT_SEED,
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
        source_mode="generated",
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
        source_mode="generated",
        parallel_trials=None,
        phase_trial_count=None,
        wrapper_wall_seconds=None,
        wrapper_process_cpu_seconds=None,
        invocation_kind="recovered_existing_complete_summary_context_unknown",
    )


def _trial_summary(
    *,
    seed: int,
    contract: Mapping[str, object],
    oracle: Oracle,
    catalogue: Sequence[IndexedLayout],
    ga_summary_path: Path,
    random_summary_path: Path,
    ga_source_mode: str,
    random_source_mode: str,
    trial_dir: Path,
) -> Mapping[str, object]:
    ga_summary = _read_json(ga_summary_path, f"GA seed {seed} summary")
    ga_winner = validate_ga_summary(ga_summary, seed=seed, oracle=oracle)
    ga_elapsed = float(ga_summary["elapsed_seconds_this_invocation"])
    random_summary = _read_json(random_summary_path, f"random seed {seed} summary")
    random_winner = validate_random_summary(
        random_summary,
        seed=seed,
        ga_elapsed_seconds=ga_elapsed,
        oracle=oracle,
        catalogue=catalogue,
    )
    ga_source = _source_packet(
        summary_path=ga_summary_path,
        summary=ga_summary,
        method="ga",
        mode=ga_source_mode,
    )
    random_source = _source_packet(
        summary_path=random_summary_path,
        summary=random_summary,
        method="random",
        mode=random_source_mode,
    )
    ga_packet = _method_packet(ga_summary, source=ga_source, oracle=oracle)
    random_packet = _method_packet(random_summary, source=random_source, oracle=oracle)
    ga_validation = _mapping(ga_winner["validation"], "GA validation")
    random_validation = _mapping(random_winner["validation"], "random validation")
    ga_holdout = _mapping(ga_winner["holdout"], "GA holdout")
    random_holdout = _mapping(random_winner["holdout"], "random holdout")
    both_validation_feasible = bool(ga_validation["feasible"]) and bool(
        random_validation["feasible"]
    )
    random_512 = oracle_random_prefix_winner(
        oracle_records=oracle.records,
        catalogue=catalogue,
        seed=seed,
    )
    random_512_validation = _mapping(random_512["validation"], "random-512")
    ga_oracle_record = oracle.records[str(ga_winner["layout_id"])]
    ga_oracle_validation = _mapping(ga_oracle_record["validation"], "GA oracle")
    censored_winner, censored_count, overtime_count = (
        _deadline_censored_random_winner(
            evaluations_path=Path(str(random_source["evaluations_path"])),
            budget_seconds=ga_elapsed,
        )
    )
    censored_packet: dict[str, object]
    if censored_winner is None:
        censored_packet = {
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
            "selection_outcome": paired_outcome(
                ga_validation, censored_validation
            ),
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
        "oracle": {
            "summary_path": str(oracle.summary_path),
            "summary_sha256": oracle.summary_sha256,
            "results_path": str(oracle.results_path),
            "results_sha256": oracle.results_sha256,
        },
        "methods": {"genetic_algorithm": ga_packet, "random_sampling": random_packet},
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
            "ga_phase": dict(
                _load_phase_reference(trial_dir / "ga", ga_summary_path)
            ),
            "random_phase": dict(
                _load_phase_reference(trial_dir / "random", random_summary_path)
            ),
            "conservative_asymmetry_favoring_random": (
                "GA-reported elapsed time includes its winner holdout replay; "
                "random receives that entire duration as validation-search time. "
                "The raw random implementation winner also includes the final "
                "atomic layout that crosses the deadline."
            ),
        },
        "primary_validation_comparison": {
            "selection_outcome": paired_outcome(ga_validation, random_validation),
            "both_feasible": both_validation_feasible,
            "random_minus_ga_cost_ms": (
                float(random_validation["expected_cost"])
                - float(ga_validation["expected_cost"])
                if both_validation_feasible
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
        "deadline_censored_random_validation_secondary": {
            "method": "exclude_random_layouts_completed_after_paired_GA_budget",
            "holdout_used": False,
            **censored_packet,
        },
        "equal_512_validation_oracle_secondary": {
            "method": "posthoc_exhaustive_oracle_replay",
            "holdout_used": False,
            "evaluated_layouts_each": 512,
            "ga_layout_id": str(ga_oracle_record["layout_id"]),
            "ga_exhaustive_rank": int(
                oracle.rank_by_id[str(ga_oracle_record["layout_id"])]
            ),
            "random_layout_id": str(random_512["layout_id"]),
            "random_exhaustive_rank": int(
                oracle.rank_by_id[str(random_512["layout_id"])]
            ),
            "selection_outcome": paired_outcome(
                ga_oracle_validation, random_512_validation
            ),
            "both_feasible": bool(ga_oracle_validation["feasible"])
            and bool(random_512_validation["feasible"]),
            "random_minus_ga_cost_ms": (
                float(random_512_validation["expected_cost"])
                - float(ga_oracle_validation["expected_cost"])
                if bool(ga_oracle_validation["feasible"])
                and bool(random_512_validation["feasible"])
                else None
            ),
        },
        "interpretation_limitations": [
            contract["search_asymmetry"],
            "Ten outer seeds are a small sample; holdout outcomes are descriptive only.",
            (
                "The historical time-matching convention conservatively favors random: "
                "its budget includes GA holdout time and its raw winner includes one "
                "deadline-crossing layout. A censored sensitivity result is reported."
            ),
        ],
    }
    trial_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(trial_dir / "summary.json", packet)
    return packet


def summarize_trials(
    trials: Sequence[Mapping[str, object]],
    *,
    contract: Mapping[str, object],
    oracle: Oracle,
    orchestration: Mapping[str, object],
) -> dict[str, object]:
    outcomes = {"ga_better": 0, "tie": 0, "random_better": 0}
    censored_outcomes = {"ga_better": 0, "tie": 0, "random_better": 0}
    equal_512_outcomes = {"ga_better": 0, "tie": 0, "random_better": 0}
    feasible_cost_deltas: list[float] = []
    raw_cost_deltas: list[float] = []
    holdout_cost_deltas: list[float] = []
    equal_512_deltas: list[float] = []
    censored_deltas: list[float] = []
    overtime_layout_counts: list[float] = []
    ga_ranks: list[float] = []
    random_ranks: list[float] = []
    for trial in trials:
        primary = _mapping(trial["primary_validation_comparison"], "primary")
        outcome = str(primary["selection_outcome"])
        outcomes[outcome] += 1
        if primary.get("random_minus_ga_cost_ms") is not None:
            feasible_cost_deltas.append(float(primary["random_minus_ga_cost_ms"]))
        raw_cost_deltas.append(float(primary["raw_random_minus_ga_cost_ms_descriptive"]))
        holdout = _mapping(trial["holdout_descriptive"], "holdout")
        holdout_cost_deltas.append(float(holdout["random_minus_ga_cost_ms"]))
        censored = _mapping(
            trial["deadline_censored_random_validation_secondary"], "censored"
        )
        overtime_layout_counts.append(float(censored["overtime_layouts_excluded"]))
        if bool(censored["available"]):
            censored_outcomes[str(censored["selection_outcome"])] += 1
            if censored.get("random_minus_ga_cost_ms") is not None:
                censored_deltas.append(float(censored["random_minus_ga_cost_ms"]))
        secondary = _mapping(
            trial["equal_512_validation_oracle_secondary"], "equal-512"
        )
        equal_512_outcomes[str(secondary["selection_outcome"])] += 1
        if secondary.get("random_minus_ga_cost_ms") is not None:
            equal_512_deltas.append(float(secondary["random_minus_ga_cost_ms"]))
        methods = _mapping(trial["methods"], "methods")
        ga_ranks.append(float(_mapping(methods["genetic_algorithm"], "GA")["exhaustive_rank"]))
        random_ranks.append(float(_mapping(methods["random_sampling"], "random")["exhaustive_rank"]))

    trial_refs = []
    for trial in trials:
        seed = int(trial["trial_seed"])
        trial_refs.append({"trial_seed": seed, "summary": f"trial_{seed:02d}/summary.json"})
    return {
        "schema_version": SCHEMA_VERSION,
        "complete": len(trials) == len(TRIAL_SEEDS),
        "completed_trials": len(trials),
        "expected_trials": len(TRIAL_SEEDS),
        "contract": dict(contract),
        "orchestration": dict(orchestration),
        "oracle": {
            "summary_path": str(oracle.summary_path),
            "summary_sha256": oracle.summary_sha256,
            "results_path": str(oracle.results_path),
            "results_sha256": oracle.results_sha256,
            "optimum_layout_id": str(oracle.optimum["layout_id"]),
        },
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
            "feasible_random_minus_ga_cost_ms": _numeric_summary(
                feasible_cost_deltas
            ),
            "bootstrap_95pct_mean_feasible_delta_ms": _bootstrap_mean_interval(
                feasible_cost_deltas
            ),
            "raw_random_minus_ga_cost_ms_descriptive": _numeric_summary(
                raw_cost_deltas
            ),
            "ga_exhaustive_rank": _numeric_summary(ga_ranks),
            "random_exhaustive_rank": _numeric_summary(random_ranks),
        },
        "equal_512_validation_oracle_secondary": {
            "method": "posthoc_exhaustive_oracle_replay",
            "paired_outcomes": equal_512_outcomes,
            "exact_two_sided_sign_test": {
                "ties_excluded": equal_512_outcomes["tie"],
                "p_value": exact_two_sided_sign_test_p_value(
                    equal_512_outcomes["ga_better"],
                    equal_512_outcomes["random_better"],
                ),
            },
            "feasible_random_minus_ga_cost_ms": _numeric_summary(equal_512_deltas),
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
            "overtime_layouts_excluded": _numeric_summary(overtime_layout_counts),
            "remaining_timing_asymmetry": (
                "Censoring removes deadline-crossing layouts, but random still "
                "receives GA winner-holdout time as validation-search budget."
            ),
        },
        "holdout_descriptive": {
            "not_used_for_selection": True,
            "random_minus_ga_cost_ms": _numeric_summary(holdout_cost_deltas),
        },
        "assumptions_and_limitations": [
            contract["search_asymmetry"],
            "Trials vary outer layout-search seeds only; split and threshold-SA seeds are fixed.",
            "The exact sign test has limited power with ten trials.",
            "Reported per-method wall times may be affected by concurrent independent trials.",
            (
                "The primary historical time convention favors random because the GA "
                "budget includes holdout replay and raw random includes its final "
                "deadline-crossing layout; censored and equal-512 sensitivities are reported."
            ),
            "Holdout results are descriptive and never select a winner.",
        ],
        "trials": trial_refs,
    }


def _import_reference(
    *, method_dir: Path, phase: str, seed: int, source_summary: Path
) -> None:
    _write_phase_reference(
        method_dir=method_dir,
        phase=phase,
        seed=seed,
        source_summary=source_summary,
        source_mode="imported_reference_not_copied",
        parallel_trials=None,
        phase_trial_count=None,
        wrapper_wall_seconds=None,
        wrapper_process_cpu_seconds=None,
        invocation_kind="external_completed_run_timing_context_not_controlled_by_harness",
    )


def run_benchmark(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    oracle_summary_path: Path = DEFAULT_ORACLE_SUMMARY,
    oracle_results_path: Path = DEFAULT_ORACLE_RESULTS,
    parallel_trials: int = 1,
    seed_zero_ga_summary: Path | None = None,
    seed_zero_random_summary: Path | None = None,
) -> dict[str, object]:
    if parallel_trials < 1:
        raise ValueError("parallel_trials must be at least 1.")
    if DEFAULT_SA_RESTARTS != RESTARTS:
        raise RuntimeError(
            "The current GA inner-fitness restart default no longer matches "
            "the frozen best-of-10 benchmark contract."
        )
    if (seed_zero_ga_summary is None) != (seed_zero_random_summary is None):
        raise ValueError("Seed-zero GA and random imports must be provided together.")
    catalogue = tuple(enumerate_k1_free_layouts())
    if len(catalogue) != EXPECTED_LAYOUT_COUNT:
        raise RuntimeError("The K1-free layout catalogue is incomplete.")
    contract = _experiment_contract(outcomes=outcomes, catalogue=catalogue)
    oracle = load_and_validate_oracle(
        summary_path=oracle_summary_path,
        results_path=oracle_results_path,
        outcomes=outcomes,
        catalogue=catalogue,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "contract.json", contract)

    imported = seed_zero_ga_summary is not None
    source_paths: dict[int, dict[str, Path]] = {}
    source_modes: dict[int, dict[str, str]] = {}
    ga_tasks: list[dict[str, object]] = []
    for seed in TRIAL_SEEDS:
        trial_dir = output_dir / f"trial_{seed:02d}"
        ga_dir = trial_dir / "ga"
        random_dir = trial_dir / "random"
        trial_dir.mkdir(parents=True, exist_ok=True)
        if seed == 0 and imported:
            assert seed_zero_ga_summary is not None
            assert seed_zero_random_summary is not None
            ga_path = seed_zero_ga_summary.resolve()
            random_path = seed_zero_random_summary.resolve()
            _import_reference(
                method_dir=ga_dir,
                phase="genetic_algorithm",
                seed=seed,
                source_summary=ga_path,
            )
            _import_reference(
                method_dir=random_dir,
                phase="random_sampling",
                seed=seed,
                source_summary=random_path,
            )
            source_paths[seed] = {"ga": ga_path, "random": random_path}
            source_modes[seed] = {"ga": "imported", "random": "imported"}
            continue
        ga_path = ga_dir / "summary.json"
        random_path = random_dir / "summary.json"
        source_paths[seed] = {"ga": ga_path, "random": random_path}
        source_modes[seed] = {"ga": "generated", "random": "generated"}
        if ga_path.exists():
            validate_ga_summary(
                _read_json(ga_path, f"GA seed {seed}"), seed=seed, oracle=oracle
            )
        else:
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
        ga_path = source_paths[seed]["ga"]
        ga_summary = _read_json(ga_path, f"GA seed {seed}")
        validate_ga_summary(ga_summary, seed=seed, oracle=oracle)
        ga_elapsed = float(ga_summary["elapsed_seconds_this_invocation"])
        random_path = source_paths[seed]["random"]
        if source_modes[seed]["random"] == "imported":
            random_summary = _read_json(random_path, "imported random seed 0")
            validate_random_summary(
                random_summary,
                seed=seed,
                ga_elapsed_seconds=ga_elapsed,
                oracle=oracle,
                catalogue=catalogue,
            )
            continue
        if random_path.exists():
            validate_random_summary(
                _read_json(random_path, f"random seed {seed}"),
                seed=seed,
                ga_elapsed_seconds=ga_elapsed,
                oracle=oracle,
                catalogue=catalogue,
            )
            continue
        random_tasks.append(
            {
                "seed": seed,
                "output_dir": str(random_path.parent),
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

    trial_packets: list[Mapping[str, object]] = []
    for seed in TRIAL_SEEDS:
        trial_packets.append(
            _trial_summary(
                seed=seed,
                contract=contract,
                oracle=oracle,
                catalogue=catalogue,
                ga_summary_path=source_paths[seed]["ga"],
                random_summary_path=source_paths[seed]["random"],
                ga_source_mode=source_modes[seed]["ga"],
                random_source_mode=source_modes[seed]["random"],
                trial_dir=output_dir / f"trial_{seed:02d}",
            )
        )

    orchestration = {
        "phase_barrier": "all_pending_GA_then_all_pending_random",
        "parallel_trials_requested": parallel_trials,
        "ga_phase_pending_trials": len(ga_tasks),
        "random_phase_pending_trials": len(random_tasks),
        "internal_layout_workers": 1,
        "worker_progress_output": "suppressed",
        "threading_environment_expectation": (
            "Set OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS="
            "NUMEXPR_NUM_THREADS=1 before Python starts."
        ),
        "timing_interpretation": (
            "Wrapper wall and process CPU seconds are recorded per invocation. "
            "Primary random budgets use GA-reported elapsed wall seconds."
        ),
        "seed_zero_imported": imported,
    }
    aggregate = summarize_trials(
        trial_packets,
        contract=contract,
        oracle=oracle,
        orchestration=orchestration,
    )
    for reference in aggregate["trials"]:
        assert isinstance(reference, dict)
        summary_path = output_dir / str(reference["summary"])
        reference["summary_sha256"] = _file_sha256(summary_path)
    _write_json_atomic(output_dir / "summary.json", aggregate)
    return aggregate


def dry_run_report(*, output_dir: Path, parallel_trials: int) -> dict[str, object]:
    observed_ga_seconds = 1626.6618771
    sequential_seconds = len(TRIAL_SEEDS) * 2.0 * observed_ga_seconds
    idealized_phase_seconds = (
        math.ceil(len(TRIAL_SEEDS) / parallel_trials)
        * 2.0
        * observed_ga_seconds
    )
    return {
        "action": "dry_run_only_no_files_or_models_loaded",
        "output_dir": str(output_dir.resolve()),
        "trial_seeds": list(TRIAL_SEEDS),
        "parallel_trials": parallel_trials,
        "phase_order": ["all_GA", "all_random"],
        "conditions": {
            "target_accuracy": TARGET_ACCURACY,
            "K1_free": True,
            "SA": "best-of-10 x 1,000 iterations, fixed restart seeds 0..9",
            "GA_layout_evaluations": 512,
            "random_budget": "paired GA reported elapsed wall seconds",
        },
        "estimated_sequential_hours": sequential_seconds / 3600.0,
        "idealized_parallel_lower_bound_hours": idealized_phase_seconds / 3600.0,
        "runtime_warning": "Concurrent CPU contention can substantially exceed the idealized bound.",
        "search_asymmetry": (
            "GA has mandatory K3->detector seed; uniform random has no mandatory seed."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--oracle-summary", type=Path, default=DEFAULT_ORACLE_SUMMARY
    )
    parser.add_argument(
        "--oracle-results", type=Path, default=DEFAULT_ORACLE_RESULTS
    )
    parser.add_argument(
        "--parallel-trials",
        type=int,
        default=1,
        help=(
            "Independent trial processes per phase. All GA trials finish before "
            "the random-search phase; each optimizer still uses one internal worker."
        ),
    )
    parser.add_argument(
        "--seed-zero-ga-summary",
        type=Path,
        help="Optional provenance-validated external seed-0 GA summary.",
    )
    parser.add_argument(
        "--seed-zero-random-summary",
        type=Path,
        help="Optional provenance-validated external seed-0 random summary.",
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
                    output_dir=args.output_dir,
                    parallel_trials=args.parallel_trials,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    summary = run_benchmark(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        oracle_summary_path=args.oracle_summary,
        oracle_results_path=args.oracle_results,
        parallel_trials=args.parallel_trials,
        seed_zero_ga_summary=args.seed_zero_ga_summary,
        seed_zero_random_summary=args.seed_zero_random_summary,
    )
    primary = _mapping(
        summary["primary_validation_comparison"], "primary comparison"
    )
    print(f"Wrote {args.output_dir / 'summary.json'}")
    print(f"Paired outcomes: {primary['paired_outcomes']}")
    print(
        "Exact sign-test p-value: "
        f"{_mapping(primary['exact_two_sided_sign_test'], 'sign test')['p_value']:.6g}"
    )


if __name__ == "__main__":  # Required for Windows ProcessPool spawn safety.
    main()
