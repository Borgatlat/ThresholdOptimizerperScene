"""Uniform random K1-free layout search with canonical threshold SA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.brute_force_k1_free_layouts import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_ITERATIONS,
    DEFAULT_OUTCOMES,
    DEFAULT_SEED,
    DEFAULT_SPLIT_STRATEGY,
    EXPECTED_LAYOUT_COUNT,
    REMOVED_CANDIDATES,
    IndexedLayout,
    _compact_optimization,
    _direct_detector_metrics,
    _layout_selection_key,
    _without_candidates,
    enumerate_k1_free_layouts,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    InnerAnnealingFitness,
    _file_sha256,
    _fitness_implementation_sha256,
    _load_jsonl,
    _settings_match,
    _write_json_atomic,
)
from hierarchy_optimizer import HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    DEFAULT_SA_RESTARTS,
    FixedLayoutThresholdEvaluator,
    split_empirical_outcomes,
)


DEFAULT_OUTPUT_DIR = Path("checkpoints/random_joint_k1_free_h24_target_0962")
DEFAULT_TARGET_ACCURACY = 0.9662
DEFAULT_TIME_BUDGET_SECONDS = 1_500.0


def uniform_layout_order(
    layouts: Sequence[IndexedLayout], random_seed: int
) -> tuple[IndexedLayout, ...]:
    """Return a deterministic uniform permutation without replacement."""

    permutation = np.random.default_rng(random_seed).permutation(len(layouts))
    return tuple(layouts[int(index)] for index in permutation)


def _order_sha256(order: Sequence[IndexedLayout]) -> str:
    digest = hashlib.sha256()
    for sampled in order:
        digest.update(f"{sampled.index}:{sampled.layout_id}\n".encode("utf-8"))
    return digest.hexdigest()


def _validate_cached_records(
    records: Mapping[str, Mapping[str, object]],
    order: Sequence[IndexedLayout],
    settings: Mapping[str, object],
) -> float:
    """Validate the append-only prefix and return its recorded elapsed time."""

    ranked: list[tuple[int, float]] = []
    seen_ranks: set[int] = set()
    for layout_id, record in records.items():
        if not _settings_match(record.get("settings"), settings):
            raise ValueError("An evaluation belongs to a different experiment.")
        rank = int(record.get("sample_rank", -1))
        if rank < 0 or rank >= len(order) or rank in seen_ranks:
            raise ValueError("Cached random-search sample ranks are invalid.")
        expected = order[rank]
        if layout_id != expected.layout_id or int(record["layout_index"]) != expected.index:
            raise ValueError("A cached evaluation does not match the saved sample order.")
        elapsed = float(record.get("search_elapsed_seconds_at_completion", -1.0))
        if elapsed < 0.0:
            raise ValueError("A cached evaluation has no valid elapsed timestamp.")
        ranked.append((rank, elapsed))
        seen_ranks.add(rank)

    ranked.sort()
    if [rank for rank, _ in ranked] != list(range(len(ranked))):
        raise ValueError("Cached random-search evaluations are not a prefix of the order.")
    if any(later < earlier for (_, earlier), (_, later) in zip(ranked, ranked[1:])):
        raise ValueError("Cached random-search elapsed timestamps are not monotonic.")
    return ranked[-1][1] if ranked else 0.0


def _holdout_metrics(
    winner: Mapping[str, object],
    indexed: IndexedLayout,
    holdout_optimizer: HierarchyOptimizer,
    target_accuracy: float,
) -> dict[str, object]:
    validation = winner["validation"]
    if not isinstance(validation, Mapping):
        raise ValueError("Winning record has no validation metrics.")
    cascade = indexed.cascade
    if cascade.initial == [cascade.detector]:
        metrics = _direct_detector_metrics(
            holdout_optimizer, cascade, target_accuracy
        )
    else:
        thresholds = validation.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("Winning validation policy has no thresholds.")
        options: dict[str, object] = {"strict_thresholds": True}
        if "active_slots" in validation:
            options["active_slots"] = validation["active_slots"]
        metrics = FixedLayoutThresholdEvaluator(
            holdout_optimizer, cascade
        ).evaluate(thresholds, **options)
    metrics = dict(metrics)
    metrics.update(
        {
            "feasible": bool(float(metrics["accuracy"]) >= target_accuracy),
            "target_accuracy": float(target_accuracy),
            "method": (
                "direct_detector_holdout"
                if cascade.initial == [cascade.detector]
                else "validation_pruned_policy_holdout_replay"
            ),
        }
    )
    return _compact_optimization(metrics)


def run_random_search(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    iterations: int = DEFAULT_ITERATIONS,
    restarts: int = DEFAULT_SA_RESTARTS,
    inner_seed: int = DEFAULT_SEED,
    sampling_seed: int = DEFAULT_SEED,
    split_seed: int = DEFAULT_SEED,
    max_layouts: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1 inclusive.")
    if time_budget_seconds <= 0.0:
        raise ValueError("time_budget_seconds must be positive.")
    if iterations < 1 or restarts < 1:
        raise ValueError("iterations and restarts must both be positive.")
    if max_layouts is not None and max_layouts < 1:
        raise ValueError("max_layouts must be positive when provided.")

    layouts = tuple(enumerate_k1_free_layouts())
    if len(layouts) != EXPECTED_LAYOUT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_LAYOUT_COUNT:,} layouts, got {len(layouts):,}."
        )
    order = uniform_layout_order(layouts, sampling_seed)
    order_hash = _order_sha256(order)
    settings: dict[str, object] = {
        "schema_version": "random-joint-layout-search/v1",
        "algorithm": "uniform_random_layout_sampling_without_replacement",
        "dataset": "m3n_vc/h24",
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "removed_candidates": sorted(REMOVED_CANDIDATES),
        "layout_space_size": EXPECTED_LAYOUT_COUNT,
        "sampling_seed": int(sampling_seed),
        "layout_order_sha256": order_hash,
        "fitness_implementation_sha256": _fitness_implementation_sha256(),
        "target_accuracy": float(target_accuracy),
        "time_budget_seconds": float(time_budget_seconds),
        "max_layouts": max_layouts,
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
        "split_seed": int(split_seed),
        "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "threshold_optimizer": {
            "method": f"best_of_{restarts}_chellapilla_continuous_gaussian_sa",
            "iterations_per_restart": int(iterations),
            "restarts": int(restarts),
            "restart_seeds": [inner_seed + index for index in range(restarts)],
            "continuous_thresholds": True,
            "quantile_points_used": False,
            "prune_stages_accepting_zero_validation_samples": True,
            "freeze_validation_active_slots_on_holdout": True,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "settings.json"
    results_path = output_dir / "evaluations.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    summary_path = output_dir / "summary.json"
    if overwrite:
        for path in (settings_path, results_path, checkpoint_path, summary_path):
            path.unlink(missing_ok=True)
    if settings_path.exists():
        existing_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if not _settings_match(existing_settings, settings):
            raise ValueError("Existing random-search checkpoint has different settings.")
    else:
        _write_json_atomic(settings_path, settings)

    records = _load_jsonl(results_path)
    recorded_elapsed = _validate_cached_records(records, order, settings)
    completed_ids = set(records)
    prior_elapsed = recorded_elapsed
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not _settings_match(checkpoint.get("settings"), settings):
            raise ValueError("Existing random-search checkpoint has different settings.")
        checkpoint_elapsed = float(checkpoint.get("search_elapsed_seconds", 0.0))
        if checkpoint_elapsed < recorded_elapsed:
            raise ValueError("Checkpoint elapsed time predates its cached evaluations.")
        prior_elapsed = checkpoint_elapsed
        if checkpoint.get("status") == "complete" and summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))

    payload = _without_candidates(load_empirical_outcomes(outcomes), REMOVED_CANDIDATES)
    validation_payload, holdout_payload, split = split_empirical_outcomes(
        payload,
        holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
        split_strategy=DEFAULT_SPLIT_STRATEGY,
        random_seed=split_seed,
    )
    validation_optimizer = HierarchyOptimizer(
        validation_payload,
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )
    fitness = InnerAnnealingFitness(
        validation_optimizer,
        target_accuracy=target_accuracy,
        quantile_points=DEFAULT_QUANTILE_POINTS,
        iterations=iterations,
        inner_seed=inner_seed,
        settings=settings,
        restarts=restarts,
    )

    started = perf_counter()
    new_evaluations = 0
    stop_reason = "layout_space_exhausted"
    for sample_rank, indexed in enumerate(order):
        if indexed.layout_id in completed_ids:
            continue
        elapsed = prior_elapsed + perf_counter() - started
        if elapsed >= time_budget_seconds:
            stop_reason = "time_budget_reached"
            break
        if max_layouts is not None and len(records) >= max_layouts:
            stop_reason = "max_layouts_reached"
            break

        evaluation_started = perf_counter()
        record = fitness(indexed)
        record["sample_rank"] = int(sample_rank)
        record["evaluation_wall_seconds"] = perf_counter() - evaluation_started
        record["search_elapsed_seconds_at_completion"] = (
            prior_elapsed + perf_counter() - started
        )
        with results_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
        records[indexed.layout_id] = record
        completed_ids.add(indexed.layout_id)
        new_evaluations += 1

        elapsed = prior_elapsed + perf_counter() - started
        _write_json_atomic(
            checkpoint_path,
            {
                "status": "running",
                "settings": settings,
                "evaluated_layouts": len(records),
                "search_elapsed_seconds": elapsed,
            },
        )

        if new_evaluations % 16 == 0:
            best = min(records.values(), key=_layout_selection_key)
            validation = best["validation"]
            assert isinstance(validation, Mapping)
            elapsed = prior_elapsed + perf_counter() - started
            print(
                f"Random search: {len(records):,} layouts; "
                f"best={float(validation['expected_cost']):.3f} ms; "
                f"elapsed={elapsed:.1f}/{time_budget_seconds:.1f} s"
            )

    search_elapsed = prior_elapsed + perf_counter() - started
    if not records:
        raise RuntimeError("The time budget ended before any layout was evaluated.")
    winner = dict(min(records.values(), key=_layout_selection_key))
    winner_indexed = next(
        indexed for indexed in layouts if indexed.layout_id == winner["layout_id"]
    )
    holdout_optimizer = HierarchyOptimizer(
        holdout_payload,
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )
    winner["holdout"] = _holdout_metrics(
        winner,
        winner_indexed,
        holdout_optimizer,
        target_accuracy,
    )
    total_elapsed = prior_elapsed + perf_counter() - started
    summary: dict[str, object] = {
        "settings": settings,
        "split": split,
        "stop_reason": stop_reason,
        "search_elapsed_seconds": search_elapsed,
        "total_elapsed_seconds_including_holdout": total_elapsed,
        "time_budget_overshoot_seconds": max(0.0, search_elapsed - time_budget_seconds),
        "unique_layouts_evaluated": len(records),
        "new_evaluations_this_invocation": new_evaluations,
        "fraction_of_layout_space": len(records) / EXPECTED_LAYOUT_COUNT,
        "evaluations": str(results_path.resolve()),
        "evaluations_sha256": _file_sha256(results_path),
        "winner": winner,
        "holdout_usage": "winner_only_after_validation_search",
    }
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(
        checkpoint_path,
        {
            "status": "complete",
            "settings": settings,
            "evaluated_layouts": len(records),
            "search_elapsed_seconds": search_elapsed,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY)
    parser.add_argument(
        "--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--restarts", type=int, default=DEFAULT_SA_RESTARTS)
    parser.add_argument("--inner-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-layouts", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_random_search(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        target_accuracy=args.target_accuracy,
        time_budget_seconds=args.time_budget_seconds,
        iterations=args.iterations,
        restarts=args.restarts,
        inner_seed=args.inner_seed,
        sampling_seed=args.sampling_seed,
        split_seed=args.split_seed,
        max_layouts=args.max_layouts,
        overwrite=args.overwrite,
    )
    winner = summary["winner"]
    validation = winner["validation"]
    holdout = winner["holdout"]
    print(
        f"Winner {winner['layout_id']}: validation "
        f"{validation['accuracy']:.6f} / {validation['expected_cost']:.3f} ms; "
        f"holdout {holdout['accuracy']:.6f} / {holdout['expected_cost']:.3f} ms"
    )


if __name__ == "__main__":
    main()
