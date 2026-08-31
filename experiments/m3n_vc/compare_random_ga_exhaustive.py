"""Validate and compare K1-free GA, random, and exhaustive summaries.

The utility deliberately refuses to compare summaries whose validation
target, empirical outcomes, split, detector, layout space, or threshold-SA
contract differ.  It reads only the three frozen summary packets and writes
one standardized JSON packet; no evaluation or plotting is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from experiments.m3n_vc.brute_force_k1_free_layouts import (
    IndexedLayout,
    _cascade_payload,
    enumerate_k1_free_layouts,
)
from experiments.m3n_vc.random_joint_optimize_hierarchy import (
    _order_sha256,
    uniform_layout_order,
)


SCHEMA_VERSION = "k1-free-random-ga-exhaustive-comparison/v1"
METHOD_NAMES = ("genetic_algorithm", "random_sampling", "exhaustive_search")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _integer(value: object, label: str) -> int:
    result = _number(value, label)
    if not result.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return int(result)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean.")
    return value


def _required(mapping: Mapping[str, object], key: str, label: str) -> object:
    if key not in mapping:
        raise ValueError(f"{label} is missing required field {key!r}.")
    return mapping[key]


def _canonical_split(
    summary: Mapping[str, object], settings: Mapping[str, object], label: str
) -> dict[str, object]:
    split = _mapping(_required(summary, "split", label), f"{label}.split")
    strategy = str(_required(split, "strategy", f"{label}.split"))
    seed = _integer(
        _required(split, "random_seed", f"{label}.split"),
        f"{label}.split.random_seed",
    )
    holdout_fraction = _number(
        _required(split, "holdout_fraction", f"{label}.split"),
        f"{label}.split.holdout_fraction",
    )
    if strategy != str(_required(settings, "split_strategy", f"{label}.settings")):
        raise ValueError(f"{label} has inconsistent split strategies.")
    if seed != _integer(
        _required(settings, "split_seed", f"{label}.settings"),
        f"{label}.settings.split_seed",
    ):
        raise ValueError(f"{label} has inconsistent split seeds.")
    settings_fraction = _number(
        _required(settings, "holdout_fraction", f"{label}.settings"),
        f"{label}.settings.holdout_fraction",
    )
    if not math.isclose(holdout_fraction, settings_fraction, abs_tol=1e-12):
        raise ValueError(f"{label} has inconsistent holdout fractions.")

    per_run = split.get("per_run", split.get("per_group"))
    if not isinstance(per_run, Mapping):
        raise ValueError(f"{label}.split must provide per_run or per_group counts.")
    return {
        "strategy": strategy,
        "seed": seed,
        "holdout_fraction": holdout_fraction,
        "validation_samples": _integer(
            _required(split, "validation_samples", f"{label}.split"),
            f"{label}.split.validation_samples",
        ),
        "holdout_samples": _integer(
            _required(split, "holdout_samples", f"{label}.split"),
            f"{label}.split.holdout_samples",
        ),
        "per_run": json.loads(json.dumps(per_run, sort_keys=True)),
    }


def _canonical_sa(settings: Mapping[str, object], label: str) -> dict[str, object]:
    sa = _mapping(
        _required(settings, "threshold_optimizer", f"{label}.settings"),
        f"{label}.settings.threshold_optimizer",
    )
    method = str(_required(sa, "method", f"{label}.threshold_optimizer"))
    seeds_value = _required(sa, "restart_seeds", f"{label}.threshold_optimizer")
    if not isinstance(seeds_value, Sequence) or isinstance(seeds_value, (str, bytes)):
        raise ValueError(f"{label}.threshold_optimizer.restart_seeds must be a list.")
    seeds = [
        _integer(seed, f"{label}.threshold_optimizer.restart_seeds[{index}]")
        for index, seed in enumerate(seeds_value)
    ]
    restarts = _integer(
        _required(sa, "restarts", f"{label}.threshold_optimizer"),
        f"{label}.threshold_optimizer.restarts",
    )
    if len(seeds) != restarts:
        raise ValueError(f"{label} restart count and restart seed count differ.")
    return {
        "method": method,
        "iterations_per_restart": _integer(
            _required(
                sa, "iterations_per_restart", f"{label}.threshold_optimizer"
            ),
            f"{label}.threshold_optimizer.iterations_per_restart",
        ),
        "restarts": restarts,
        "restart_seeds": seeds,
        "prune_stages_accepting_zero_validation_samples": _boolean(
            _required(
                sa,
                "prune_stages_accepting_zero_validation_samples",
                f"{label}.threshold_optimizer",
            ),
            f"{label}.threshold_optimizer."
            "prune_stages_accepting_zero_validation_samples",
        ),
        "freeze_validation_active_slots_on_holdout": _boolean(
            _required(
                sa,
                "freeze_validation_active_slots_on_holdout",
                f"{label}.threshold_optimizer",
            ),
            f"{label}.threshold_optimizer."
            "freeze_validation_active_slots_on_holdout",
        ),
    }


def _layout_space_size(
    summary: Mapping[str, object], settings: Mapping[str, object], label: str
) -> int:
    candidates = (
        summary.get("layout_space_size"),
        settings.get("layout_space_size"),
        settings.get("expected_layout_count"),
    )
    for candidate in candidates:
        if candidate is not None:
            size = _integer(candidate, f"{label}.layout_space_size")
            if size < 1:
                raise ValueError(f"{label}.layout_space_size must be positive.")
            return size
    raise ValueError(f"{label} does not declare its layout-space size.")


def _contract(summary: Mapping[str, object], label: str) -> dict[str, object]:
    settings = _mapping(
        _required(summary, "settings", label), f"{label}.settings"
    )
    target = _number(
        _required(settings, "target_accuracy", f"{label}.settings"),
        f"{label}.settings.target_accuracy",
    )
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"{label} target accuracy must be in [0, 1].")
    summary_target = summary.get("target_accuracy")
    if summary_target is not None and not math.isclose(
        target, _number(summary_target, f"{label}.target_accuracy"), abs_tol=1e-12
    ):
        raise ValueError(f"{label} has inconsistent target accuracies.")

    outcomes_hash = str(
        _required(settings, "outcomes_sha256", f"{label}.settings")
    )
    if len(outcomes_hash) != 64:
        raise ValueError(f"{label}.settings.outcomes_sha256 is not a SHA-256 hash.")
    try:
        int(outcomes_hash, 16)
    except ValueError as error:
        raise ValueError(
            f"{label}.settings.outcomes_sha256 is not hexadecimal."
        ) from error
    removed = settings.get("removed_candidates")
    if not isinstance(removed, Sequence) or isinstance(removed, (str, bytes)):
        raise ValueError(f"{label}.settings.removed_candidates must be a list.")
    return {
        "target_accuracy": target,
        "outcomes_sha256": outcomes_hash,
        "removed_candidates": sorted(str(item) for item in removed),
        "detector_mode": str(
            _required(settings, "detector_mode", f"{label}.settings")
        ),
        "detector_cost_ms": _number(
            _required(settings, "detector_cost_ms", f"{label}.settings"),
            f"{label}.settings.detector_cost_ms",
        ),
        "layout_space_size": _layout_space_size(summary, settings, label),
        "split": _canonical_split(summary, settings, label),
        "threshold_optimizer": _canonical_sa(settings, label),
    }


def _route_counts(value: object, label: str) -> dict[str, int]:
    routes = _mapping(value, label)
    result = {
        str(route): _integer(count, f"{label}.{route}")
        for route, count in routes.items()
    }
    if any(count < 0 for count in result.values()):
        raise ValueError(f"{label} cannot contain negative counts.")
    return result


def _thresholds(value: object, label: str) -> dict[str, float]:
    thresholds = _mapping(value, label)
    result = {
        str(slot): _number(threshold, f"{label}.{slot}")
        for slot, threshold in thresholds.items()
    }
    if any(not 0.0 <= threshold <= 1.0 for threshold in result.values()):
        raise ValueError(f"{label} values must be in [0, 1].")
    return result


def _partition(
    metrics_value: object, label: str, target_accuracy: float
) -> tuple[dict[str, object], dict[str, float]]:
    metrics = _mapping(metrics_value, label)
    accuracy = _number(_required(metrics, "accuracy", label), f"{label}.accuracy")
    cost = _number(
        _required(metrics, "expected_cost", label), f"{label}.expected_cost"
    )
    if not 0.0 <= accuracy <= 1.0 or cost < 0.0:
        raise ValueError(f"{label} has an invalid accuracy or expected cost.")
    routes = _route_counts(
        _required(metrics, "route_counts", label), f"{label}.route_counts"
    )
    total = metrics.get("total")
    if total is not None and sum(routes.values()) != _integer(total, f"{label}.total"):
        raise ValueError(f"{label} route counts do not sum to total samples.")
    return (
        {
            "expected_cost_ms": cost,
            "accuracy": accuracy,
            "meets_target_accuracy": accuracy >= target_accuracy,
            "route_counts": routes,
        },
        _thresholds(
            _required(metrics, "thresholds", label), f"{label}.thresholds"
        ),
    )


def _method_packet(
    record_value: object,
    *,
    label: str,
    target_accuracy: float,
    elapsed_seconds: object,
    elapsed_source_field: str,
    evaluated_layouts: object,
) -> dict[str, object]:
    record = _mapping(record_value, label)
    layout = _mapping(_required(record, "layout", label), f"{label}.layout")
    serialized_layout = json.dumps(layout, sort_keys=True)
    if "K1" in serialized_layout:
        raise ValueError(f"{label}.layout contains removed candidate K1.")
    validation, thresholds = _partition(
        _required(record, "validation", label), f"{label}.validation", target_accuracy
    )
    holdout, holdout_thresholds = _partition(
        _required(record, "holdout", label), f"{label}.holdout", target_accuracy
    )
    if holdout_thresholds != thresholds:
        raise ValueError(f"{label} holdout did not replay validation thresholds.")
    elapsed = _number(elapsed_seconds, f"{label}.{elapsed_source_field}")
    count = _integer(evaluated_layouts, f"{label}.evaluated_layouts")
    if elapsed < 0.0 or count < 1:
        raise ValueError(f"{label} has an invalid elapsed time or layout count.")
    packet: dict[str, object] = {
        "layout": json.loads(serialized_layout),
        "thresholds": thresholds,
        "validation": validation,
        "holdout": holdout,
        "elapsed_seconds": elapsed,
        "elapsed_source_field": elapsed_source_field,
        "evaluated_layouts": count,
    }
    for source, destination in (
        ("layout_id", "layout_id"),
        ("layout_index", "layout_index"),
        ("best_layout_id", "layout_id"),
        ("best_layout_index", "layout_index"),
    ):
        if source in record and destination not in packet:
            packet[destination] = (
                _integer(record[source], f"{label}.{source}")
                if destination == "layout_index"
                else str(record[source])
            )
    return packet


def _same_contract(
    contracts: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    reference_name = METHOD_NAMES[0]
    reference = dict(contracts[reference_name])
    for name in METHOD_NAMES[1:]:
        candidate = dict(contracts[name])
        if candidate != reference:
            differing = sorted(
                key
                for key in set(reference) | set(candidate)
                if reference.get(key) != candidate.get(key)
            )
            raise ValueError(
                f"{name} does not match {reference_name}; differing contract "
                f"fields: {', '.join(differing)}."
            )
    if reference["removed_candidates"] != ["K1"]:
        raise ValueError("Comparison requires the K1-free candidate set.")
    return reference


def _validation_selection_key(
    record: Mapping[str, object], target_accuracy: float
) -> tuple[float, ...]:
    validation = _mapping(
        _required(record, "validation", "layout result"),
        "layout result.validation",
    )
    accuracy = _number(
        _required(validation, "accuracy", "layout result.validation"),
        "layout result.validation.accuracy",
    )
    cost = _number(
        _required(validation, "expected_cost", "layout result.validation"),
        "layout result.validation.expected_cost",
    )
    feasible = accuracy >= target_accuracy
    index = _integer(
        _required(record, "layout_index", "layout result"),
        "layout result.layout_index",
    )
    return (
        0.0 if feasible else 1.0,
        cost if feasible else -accuracy,
        -accuracy if feasible else cost,
        float(index),
    )


def _prefix_winner(
    records: Sequence[Mapping[str, object]], target_accuracy: float
) -> Mapping[str, object]:
    if not records:
        raise ValueError("A sampling prefix cannot be empty.")
    return min(
        records,
        key=lambda record: _validation_selection_key(record, target_accuracy),
    )


def _prefix_packet(
    records: Sequence[Mapping[str, object]], target_accuracy: float
) -> dict[str, object]:
    winner = _prefix_winner(records, target_accuracy)
    validation = _mapping(winner["validation"], "prefix winner.validation")
    return {
        "evaluated_layouts": len(records),
        "best_layout_id": str(winner["layout_id"]),
        "best_layout_index": int(winner["layout_index"]),
        "best_validation_cost_ms": float(validation["expected_cost"]),
        "best_validation_accuracy": float(validation["accuracy"]),
    }


def _sampling_analysis(
    *,
    random_summary: Mapping[str, object],
    method_packets: Mapping[str, Mapping[str, object]],
    layout_results: Sequence[Mapping[str, object]],
    catalogue: Sequence[IndexedLayout],
    target_accuracy: float,
) -> dict[str, object]:
    if not catalogue:
        raise ValueError("The authoritative layout catalogue is empty.")
    random_settings = _mapping(
        _required(random_summary, "settings", "random_sampling"),
        "random_sampling.settings",
    )
    sampling_seed = _integer(
        _required(random_settings, "sampling_seed", "random_sampling.settings"),
        "random_sampling.settings.sampling_seed",
    )
    if sampling_seed != 0:
        raise ValueError("The equal-evaluation analysis requires sampling seed 0.")
    order = uniform_layout_order(catalogue, sampling_seed)
    actual_order_hash = _order_sha256(order)
    saved_order_hash = str(
        _required(
            random_settings, "layout_order_sha256", "random_sampling.settings"
        )
    )
    if actual_order_hash != saved_order_hash:
        raise ValueError(
            "The random summary's layout-order hash does not match the "
            "authoritative seed-0 catalogue permutation."
        )

    catalogue_by_id = {entry.layout_id: entry for entry in catalogue}
    if len(catalogue_by_id) != len(catalogue):
        raise ValueError("The authoritative catalogue contains duplicate layout IDs.")
    records_by_id: dict[str, Mapping[str, object]] = {}
    seen_indices: set[int] = set()
    for line_number, record_value in enumerate(layout_results, 1):
        record = _mapping(record_value, f"layout_results line {line_number}")
        layout_id = str(
            _required(record, "layout_id", f"layout_results line {line_number}")
        )
        layout_index = _integer(
            _required(record, "layout_index", f"layout_results line {line_number}"),
            f"layout_results line {line_number}.layout_index",
        )
        indexed = catalogue_by_id.get(layout_id)
        if indexed is None or indexed.index != layout_index:
            raise ValueError(
                f"Layout result {layout_id} does not match the authoritative catalogue."
            )
        saved_layout = _mapping(
            _required(record, "layout", f"layout_results line {line_number}"),
            f"layout_results line {line_number}.layout",
        )
        if dict(saved_layout) != _cascade_payload(indexed.cascade):
            raise ValueError(f"Layout result {layout_id} has a different cascade payload.")
        if layout_id in records_by_id or layout_index in seen_indices:
            raise ValueError("The exhaustive layout-results file contains duplicates.")
        # Validate the fitness packet now, rather than trusting it only if the
        # layout happens to enter one of the analyzed prefixes.
        validation_value = _mapping(
            _required(record, "validation", f"layout_results line {line_number}"),
            f"layout_results line {line_number}.validation",
        )
        validation_for_check = dict(validation_value)
        if indexed.cascade.initial == [indexed.cascade.detector]:
            validation_for_check.setdefault("thresholds", {})
        _partition(
            validation_for_check,
            f"layout_results line {line_number}.validation",
            target_accuracy,
        )
        records_by_id[layout_id] = record
        seen_indices.add(layout_index)

    expected_indices = set(range(len(catalogue)))
    if len(records_by_id) != len(catalogue) or seen_indices != expected_indices:
        raise ValueError("The exhaustive layout-results file is incomplete.")
    ordered_records = [records_by_id[indexed.layout_id] for indexed in order]

    actual_count = int(method_packets["random_sampling"]["evaluated_layouts"])
    ga_count = int(method_packets["genetic_algorithm"]["evaluated_layouts"])
    if not 1 <= actual_count <= len(ordered_records):
        raise ValueError("The random evaluation count is outside the layout space.")
    if not 1 <= ga_count <= len(ordered_records):
        raise ValueError("The GA evaluation count is outside the layout space.")
    actual_prefix = ordered_records[:actual_count]
    equal_ga_prefix = ordered_records[:ga_count]
    actual_prefix_winner = _prefix_winner(actual_prefix, target_accuracy)
    random_packet = method_packets["random_sampling"]
    random_validation = random_packet["validation"]
    saved_random_id = str(random_packet.get("layout_id", ""))
    prefix_validation = _mapping(
        actual_prefix_winner["validation"], "actual prefix winner.validation"
    )
    if (
        str(actual_prefix_winner["layout_id"]) != saved_random_id
        or not math.isclose(
            float(prefix_validation["expected_cost"]),
            float(random_validation["expected_cost_ms"]),
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(prefix_validation["accuracy"]),
            float(random_validation["accuracy"]),
            abs_tol=1e-12,
        )
        or dict(prefix_validation["thresholds"]) != dict(random_packet["thresholds"])
        or dict(prefix_validation["route_counts"])
        != dict(random_validation["route_counts"])
    ):
        raise ValueError(
            "The exhaustive seed-0 prefix does not reproduce the saved random winner."
        )

    feasible_records = [
        record
        for record in layout_results
        if float(_mapping(record["validation"], "validation")["accuracy"])
        >= target_accuracy
    ]
    exhaustive_winner = _prefix_winner(feasible_records, target_accuracy)
    exhaustive_packet = method_packets["exhaustive_search"]
    exhaustive_validation = _mapping(
        exhaustive_winner["validation"], "exhaustive winner.validation"
    )
    global_cost = float(exhaustive_validation["expected_cost"])
    if (
        str(exhaustive_winner["layout_id"])
        != str(exhaustive_packet.get("layout_id", ""))
        or not math.isclose(
            global_cost,
            float(exhaustive_packet["validation"]["expected_cost_ms"]),
            abs_tol=1e-9,
        )
    ):
        raise ValueError(
            "The exhaustive JSONL winner does not reproduce the benchmark summary."
        )

    tolerance = 1e-9
    globally_optimal_ids = {
        str(record["layout_id"])
        for record in feasible_records
        if math.isclose(
            float(_mapping(record["validation"], "validation")["expected_cost"]),
            global_cost,
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    }
    first_optimum_rank = min(
        rank
        for rank, indexed in enumerate(order)
        if indexed.layout_id in globally_optimal_ids
    )
    random_cost = float(random_validation["expected_cost_ms"])
    strictly_better = sum(
        float(_mapping(record["validation"], "validation")["expected_cost"])
        < random_cost - tolerance
        for record in feasible_records
    )
    at_or_better = sum(
        float(_mapping(record["validation"], "validation")["expected_cost"])
        <= random_cost + tolerance
        for record in feasible_records
    )
    return {
        "sampling_seed": sampling_seed,
        "layout_order_sha256": actual_order_hash,
        "actual_equal_time_prefix": _prefix_packet(
            actual_prefix, target_accuracy
        ),
        "equal_ga_evaluation_prefix": _prefix_packet(
            equal_ga_prefix, target_accuracy
        ),
        "global_optimum": {
            "validation_cost_ms": global_cost,
            "nominal_layout_count": len(globally_optimal_ids),
            "first_sample_rank_zero_based": first_optimum_rank,
            "first_sample_rank_one_based": first_optimum_rank + 1,
        },
        "exhaustive_layout_counts_relative_to_random": {
            "strictly_better": strictly_better,
            "at_or_better": at_or_better,
        },
        "actual_prefix_reproduces_random_winner": True,
    }


def build_comparison(
    ga_summary: Mapping[str, object],
    random_summary: Mapping[str, object],
    benchmark_summary: Mapping[str, object],
    *,
    layout_results: Sequence[Mapping[str, object]] | None = None,
    catalogue: Sequence[IndexedLayout] | None = None,
) -> dict[str, object]:
    """Build a validated, standardized comparison packet in memory."""

    summaries = {
        "genetic_algorithm": ga_summary,
        "random_sampling": random_summary,
        "exhaustive_search": benchmark_summary,
    }
    contracts = {
        name: _contract(summary, name) for name, summary in summaries.items()
    }
    contract = _same_contract(contracts)
    target = float(contract["target_accuracy"])

    ga = _method_packet(
        _required(ga_summary, "winner", "genetic_algorithm"),
        label="genetic_algorithm.winner",
        target_accuracy=target,
        elapsed_seconds=_required(
            ga_summary, "elapsed_seconds_this_invocation", "genetic_algorithm"
        ),
        elapsed_source_field="elapsed_seconds_this_invocation",
        evaluated_layouts=_required(
            ga_summary, "unique_layouts_evaluated", "genetic_algorithm"
        ),
    )
    random_elapsed_field = (
        "search_elapsed_seconds"
        if "search_elapsed_seconds" in random_summary
        else "elapsed_seconds_this_invocation"
    )
    random = _method_packet(
        _required(random_summary, "winner", "random_sampling"),
        label="random_sampling.winner",
        target_accuracy=target,
        elapsed_seconds=_required(
            random_summary, random_elapsed_field, "random_sampling"
        ),
        elapsed_source_field=random_elapsed_field,
        evaluated_layouts=_required(
            random_summary, "unique_layouts_evaluated", "random_sampling"
        ),
    )
    methods = _mapping(
        _required(benchmark_summary, "methods", "exhaustive_search"),
        "exhaustive_search.methods",
    )
    exhaustive_record = _mapping(
        _required(methods, "exhaustive_joint", "exhaustive_search.methods"),
        "exhaustive_search.methods.exhaustive_joint",
    )
    exhaustive = _method_packet(
        exhaustive_record,
        label="exhaustive_search.methods.exhaustive_joint",
        target_accuracy=target,
        elapsed_seconds=_required(
            exhaustive_record,
            "completion_seconds",
            "exhaustive_search.methods.exhaustive_joint",
        ),
        elapsed_source_field="completion_seconds",
        evaluated_layouts=_required(
            exhaustive_record,
            "completed_layouts",
            "exhaustive_search.methods.exhaustive_joint",
        ),
    )
    if exhaustive["evaluated_layouts"] != contract["layout_space_size"]:
        raise ValueError("The benchmark summary is not a complete exhaustive search.")

    method_packets = {
        "genetic_algorithm": ga,
        "random_sampling": random,
        "exhaustive_search": exhaustive,
    }
    layout_space_size = int(contract["layout_space_size"])
    for name, packet in method_packets.items():
        if int(packet["evaluated_layouts"]) > layout_space_size:
            raise ValueError(f"{name} evaluated more than the declared layout space.")
        layout_index = packet.get("layout_index")
        if layout_index is not None and not 0 <= int(layout_index) < layout_space_size:
            raise ValueError(f"{name}.layout_index is outside the layout space.")
    infeasible = [
        name
        for name, packet in method_packets.items()
        if not packet["validation"]["meets_target_accuracy"]
    ]
    if infeasible:
        raise ValueError(
            "Validation-cost regret is undefined for infeasible winners: "
            + ", ".join(infeasible)
        )

    ga_cost = float(ga["validation"]["expected_cost_ms"])
    random_cost = float(random["validation"]["expected_cost_ms"])
    exhaustive_cost = float(exhaustive["validation"]["expected_cost_ms"])
    ga_regret = ga_cost - exhaustive_cost
    random_regret = random_cost - exhaustive_cost
    tolerance = 1e-9
    if ga_regret < -tolerance or random_regret < -tolerance:
        raise ValueError(
            "An approximate optimizer is cheaper than the exhaustive result "
            "under an identical deterministic inner-SA contract."
        )

    comparison: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_contract": contract,
        "methods": method_packets,
        "pairwise_validation_cost_ms": {
            "random_minus_ga": random_cost - ga_cost,
            "ga_minus_exhaustive": ga_regret,
            "random_minus_exhaustive": random_regret,
        },
        "validation_regret_to_exhaustive_ms": {
            "genetic_algorithm": max(0.0, ga_regret),
            "random_sampling": max(0.0, random_regret),
            "exhaustive_search": 0.0,
        },
    }
    if layout_results is not None:
        authoritative_catalogue = (
            tuple(enumerate_k1_free_layouts())
            if catalogue is None
            else tuple(catalogue)
        )
        if len(authoritative_catalogue) != int(contract["layout_space_size"]):
            raise ValueError(
                "The authoritative catalogue size does not match the summaries."
            )
        comparison["sampling_analysis"] = _sampling_analysis(
            random_summary=random_summary,
            method_packets=method_packets,
            layout_results=layout_results,
            catalogue=authoritative_catalogue,
            target_accuracy=target,
        )
    return comparison


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_summary(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} summary does not exist: {path}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} summary is not valid JSON: {path}") from error


def _read_layout_results(path: Path) -> list[Mapping[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Exhaustive layout results do not exist: {path}")
    records: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid exhaustive JSONL record on line {line_number}: {path}"
                ) from error
            records.append(
                _mapping(record, f"exhaustive layout-results line {line_number}")
            )
    return records


def _recorded_layout_results(
    benchmark_summary: Mapping[str, object], benchmark_summary_path: Path
) -> tuple[Path | None, str]:
    methods = _mapping(
        _required(benchmark_summary, "methods", "exhaustive_search"),
        "exhaustive_search.methods",
    )
    exhaustive = _mapping(
        _required(methods, "exhaustive_joint", "exhaustive_search.methods"),
        "exhaustive_search.methods.exhaustive_joint",
    )
    expected_hash = str(
        _required(
            exhaustive,
            "layout_results_sha256",
            "exhaustive_search.methods.exhaustive_joint",
        )
    )
    if len(expected_hash) != 64:
        raise ValueError("The benchmark layout-results SHA-256 is invalid.")
    recorded_value = exhaustive.get("layout_results")
    if recorded_value is None:
        return None, expected_hash
    recorded = Path(str(recorded_value))
    if recorded.is_absolute():
        return recorded.resolve(), expected_hash

    candidates = {
        recorded.resolve(),
        (benchmark_summary_path.parent / recorded).resolve(),
    }
    existing = [candidate for candidate in candidates if candidate.is_file()]
    if len(existing) > 1:
        raise ValueError(
            "The benchmark's relative layout-results path resolves ambiguously."
        )
    return (existing[0] if existing else recorded.resolve()), expected_hash


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_comparison(
    *,
    ga_summary_path: Path,
    random_summary_path: Path,
    benchmark_summary_path: Path,
    output_path: Path,
    layout_results_path: Path | None = None,
    checkpoints_root: Path = Path("checkpoints"),
    catalogue: Sequence[IndexedLayout] | None = None,
) -> dict[str, object]:
    """Read summaries, validate them, and atomically write the comparison."""

    resolved_output = output_path.resolve()
    resolved_root = checkpoints_root.resolve()
    if resolved_output != resolved_root and resolved_root not in resolved_output.parents:
        raise ValueError("--output must be located under the checkpoints directory.")
    input_paths = {
        "genetic_algorithm": ga_summary_path.resolve(),
        "random_sampling": random_summary_path.resolve(),
        "exhaustive_search": benchmark_summary_path.resolve(),
    }
    if resolved_output in input_paths.values():
        raise ValueError("--output cannot overwrite an input summary.")
    summaries = {
        name: _read_summary(path, name) for name, path in input_paths.items()
    }
    recorded_results_path, expected_results_hash = _recorded_layout_results(
        summaries["exhaustive_search"], input_paths["exhaustive_search"]
    )
    resolved_results_path = (
        layout_results_path.resolve()
        if layout_results_path is not None
        else recorded_results_path
    )
    if resolved_results_path is None:
        raise ValueError(
            "The benchmark does not record layout_results; pass --layout-results."
        )
    if resolved_output == resolved_results_path:
        raise ValueError("--output cannot overwrite the exhaustive layout results.")
    actual_results_hash = _sha256(resolved_results_path)
    if actual_results_hash != expected_results_hash:
        raise ValueError(
            "The exhaustive layout-results SHA-256 does not match the benchmark summary."
        )
    layout_results = _read_layout_results(resolved_results_path)
    comparison = build_comparison(
        summaries["genetic_algorithm"],
        summaries["random_sampling"],
        summaries["exhaustive_search"],
        layout_results=layout_results,
        catalogue=catalogue,
    )
    comparison["inputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in input_paths.items()
    }
    comparison["inputs"]["exhaustive_layout_results"] = {
        "path": str(resolved_results_path),
        "sha256": actual_results_hash,
    }
    _write_json_atomic(resolved_output, comparison)
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-summary", type=Path, required=True)
    parser.add_argument("--random-summary", type=Path, required=True)
    parser.add_argument("--benchmark-summary", type=Path, required=True)
    parser.add_argument(
        "--layout-results",
        type=Path,
        help=(
            "Exhaustive layout-results JSONL. Defaults to the path recorded "
            "in the benchmark summary; its SHA-256 is always verified."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    comparison = run_comparison(
        ga_summary_path=args.ga_summary,
        random_summary_path=args.random_summary,
        benchmark_summary_path=args.benchmark_summary,
        layout_results_path=args.layout_results,
        output_path=args.output,
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
