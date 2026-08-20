"""Brute-force K1-free hierarchy layouts and anneal each layout's thresholds.

The outer search is exact over the paper's legal layout grammar after removing
router K1:

* the initial cascade is an ordered subset of K0, K2, and K3;
* if K0 occurs, each of its SUV and coupe branches is an ordered subset of
  the group specialist(s) and globals that did not precede K0; and
* every initial/branch cascade terminates at the paper Kdet.

This produces 5,545 layouts. The detector-only layout has no thresholds and is
scored directly; the other 5,544 layouts receive the best of ten independent
1,000-step continuous Gaussian SA runs based on Chellapilla et al. (DAS 2006).

The defaults intentionally reproduce the settings behind
``fig1_layouts_accuracy_cost.png``:

* h24 empirical outcomes;
* paper Kdet;
* blocked-per-run 80/20 validation/holdout split;
* ten continuous SA restarts of 1,000 iterations and seed 0; and
* the fixed K3 -> Kdet validation accuracy target (0.9833763718528082).

Examples
--------
Inspect the count without running optimization::

    python brute_force_k1_free_layouts.py --dry-run

Run locally, resuming an interrupted result file automatically::

    python brute_force_k1_free_layouts.py

Split across 64 scheduler array tasks::

    python brute_force_k1_free_layouts.py --num-shards 64 --shard-index 0

Merge completed shard files::

    python brute_force_k1_free_layouts.py --output-dir checkpoints/brute_force_k1_free_h24 --merge-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path
from time import perf_counter
from typing import Iterable, Iterator, Mapping, Sequence

from tqdm import tqdm

from empirical_outcomes import load_empirical_outcomes
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


DEFAULT_OUTCOMES = Path("checkpoints/empirical_outcomes.pkl")
DEFAULT_OUTPUT_DIR = Path("checkpoints/brute_force_k1_free_h24")
DEFAULT_ITERATIONS = 1_000
DEFAULT_HOLDOUT_FRACTION = 0.20
DEFAULT_SPLIT_STRATEGY = "blocked_per_run"
DEFAULT_SEED = 0
REMOVED_CANDIDATES = frozenset({"K1"})

# Fig. 1's K3 -> paper-Kdet baseline validation accuracy. Its corresponding
# holdout result is 0.9851612903225806 accuracy at 1561.0626697763914 ms.
FIG1_K3_TARGET_ACCURACY = 0.9833763718528082
FIG1_K3_HOLDOUT_ACCURACY = 0.9851612903225806
FIG1_K3_HOLDOUT_COST_MS = 1561.0626697763914

# Measured on the development machine for the seven-slot
# k0_k2_k3_hierarchy layout, which is close to the 7.18-slot average here.
REFERENCE_SECONDS_PER_LAYOUT = 3.70
EXPECTED_LAYOUT_COUNT = 5_545


@dataclass(frozen=True)
class IndexedLayout:
    """One stable, serializable layout in the K1-free search."""

    index: int
    layout_id: str
    cascade: Cascade


def ordered_subsets(items: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Yield every ordered subset, including the empty subset."""

    for length in range(len(items) + 1):
        yield from permutations(items, length)


def _cascade_payload(cascade: Cascade) -> dict[str, object]:
    return {
        "initial": list(cascade.initial),
        "specialized": {
            f"{router_id}:{group}": list(chain)
            for (router_id, group), chain in sorted(cascade.specialized.items())
        },
    }


def layout_id(cascade: Cascade) -> str:
    """Return a short stable identifier based only on cascade structure."""

    canonical = json.dumps(
        _cascade_payload(cascade),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def enumerate_k1_free_layouts(
    *,
    global_ids: Sequence[str] = ("K2", "K3"),
    router_id: str = "K0",
    specialized_by_group: Mapping[str, Sequence[str]] | None = None,
    detector_id: str = "detector",
) -> Iterator[IndexedLayout]:
    """Enumerate the complete legal layout space after removing K1.

    A global that precedes K0 has already returned IDK before routing, so the
    paper grammar excludes it from K0's branches. A global after K0 may also
    appear in a branch because the two occurrences are on mutually exclusive
    execution paths.
    """

    if specialized_by_group is None:
        specialized_by_group = {
            "coupe": ("K5", "K6"),
            "suv": ("K4",),
        }
    groups = tuple(sorted(specialized_by_group))
    initial_candidates = (*global_ids, router_id)
    next_index = 0

    for initial in ordered_subsets(initial_candidates):
        if router_id not in initial:
            cascade = Cascade(
                expected_cost=0.0,
                initial=[*initial, detector_id],
                specialized={},
                detector=detector_id,
            )
            yield IndexedLayout(next_index, layout_id(cascade), cascade)
            next_index += 1
            continue

        router_position = initial.index(router_id)
        rejected_globals = set(initial[:router_position]) & set(global_ids)
        remaining_globals = tuple(
            candidate_id
            for candidate_id in global_ids
            if candidate_id not in rejected_globals
        )
        branch_options = []
        for group in groups:
            candidates = (
                *remaining_globals,
                *specialized_by_group[group],
            )
            branch_options.append(tuple(ordered_subsets(candidates)))

        for selected_branches in product(*branch_options):
            specialized = {
                (router_id, group): [*chain, detector_id]
                for group, chain in zip(groups, selected_branches, strict=True)
            }
            cascade = Cascade(
                expected_cost=0.0,
                initial=[*initial, detector_id],
                specialized=specialized,
                detector=detector_id,
            )
            yield IndexedLayout(next_index, layout_id(cascade), cascade)
            next_index += 1


def _without_candidates(payload: dict, excluded: Iterable[str]) -> dict:
    """Return an empirical payload with excluded candidates removed."""

    excluded_set = set(excluded)
    filtered = dict(payload)
    filtered.update({
        "labels": payload["labels"].copy(),
        "candidates": payload["candidates"].loc[
            ~payload["candidates"]["id"].isin(excluded_set)
        ].copy(),
        "detector": dict(payload["detector"]),
        "outcomes": payload["outcomes"].loc[
            ~payload["outcomes"]["candidate_id"].isin(excluded_set)
        ].copy(),
    })
    return filtered


def _direct_detector_metrics(
    optimizer: HierarchyOptimizer,
    cascade: Cascade,
    target_accuracy: float,
) -> dict:
    """Score the one zero-threshold layout without invoking the annealer."""

    metrics = optimizer.evaluate_cascade(cascade)
    metrics.update(
        {
            "thresholds": {},
            "feasible": bool(metrics["accuracy"] >= target_accuracy),
            "target_accuracy": float(target_accuracy),
            "method": "direct_no_thresholds",
            "evaluations": 1,
            "elapsed_seconds": 0.0,
            "annealing_iterations": 0,
        }
    )
    return metrics


def _compact_optimization(metrics: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "accuracy",
        "expected_cost",
        "correct",
        "total",
        "thresholds",
        "route_counts",
        "macro_accuracy",
        "worst_class_accuracy",
        "per_class_accuracy",
        "feasible",
        "target_accuracy",
        "method",
        "evaluations",
        "elapsed_seconds",
        "annealing_iterations",
        "annealing_evaluations",
        "annealing_elapsed_seconds",
        "annealing_accepted_moves",
        "random_proposal_rate",
        "coordinate_descent_evaluations",
        "coordinate_descent_elapsed_seconds",
        "coordinate_descent_passes",
        "restart_count",
        "iterations_per_restart",
        "total_requested_iterations",
        "selected_restart_index",
        "selected_restart_seed",
        "restart_seeds",
        "restart_costs_ms",
        "restart_accuracies",
        "infeasible_proposals_rejected",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _layout_selection_key(record: Mapping[str, object]) -> tuple[float, ...]:
    validation = record["validation"]
    assert isinstance(validation, Mapping)
    feasible = bool(validation["feasible"])
    accuracy = float(validation["accuracy"])
    cost = float(validation["expected_cost"])
    return (
        0.0 if feasible else 1.0,
        cost if feasible else -accuracy,
        -accuracy if feasible else cost,
        float(record["layout_index"]),
    )


def _settings(
    *,
    outcomes: Path,
    target_accuracy: float,
    iterations: int,
    quantile_points: int,
    seed: int,
    holdout_fraction: float,
    split_strategy: str,
    num_shards: int,
    shard_index: int,
) -> dict[str, object]:
    return {
        "outcomes": str(outcomes.resolve()),
        "removed_candidates": sorted(REMOVED_CANDIDATES),
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "target_accuracy": float(target_accuracy),
        "target_accuracy_source": "fig1_K3_validation_baseline",
        "iterations": int(iterations),
        "quantile_points": int(quantile_points),
        "seed": int(seed),
        "holdout_fraction": float(holdout_fraction),
        "split_strategy": split_strategy,
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
    }


def _record_settings_match(
    record: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    actual = record.get("settings")
    return isinstance(actual, Mapping) and dict(actual) == dict(expected)


def _load_jsonl(path: Path) -> dict[int, dict]:
    records: dict[int, dict] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A killed process can leave one partial trailing line. Earlier
                # complete records remain valid and resumable.
                print(f"Ignoring incomplete JSONL line {line_number} in {path}")
                continue
            records[int(record["layout_index"])] = record
    return records


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _summary(
    records: Mapping[int, Mapping[str, object]],
    *,
    settings: Mapping[str, object],
    assigned_layouts: int,
    elapsed_seconds: float,
) -> dict[str, object]:
    best = min(records.values(), key=_layout_selection_key) if records else None
    return {
        "settings": dict(settings),
        "expected_total_layouts": EXPECTED_LAYOUT_COUNT,
        "assigned_layouts": int(assigned_layouts),
        "completed_layouts": int(len(records)),
        "elapsed_seconds_this_invocation": float(elapsed_seconds),
        "best": best,
        "fig1_reference": {
            "layout": {
                "initial": ["K3", "detector"],
                "specialized": {},
            },
            "validation_target_accuracy": FIG1_K3_TARGET_ACCURACY,
            "holdout_accuracy": FIG1_K3_HOLDOUT_ACCURACY,
            "holdout_cost_ms": FIG1_K3_HOLDOUT_COST_MS,
        },
    }


def run_search(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_accuracy: float = FIG1_K3_TARGET_ACCURACY,
    iterations: int = DEFAULT_ITERATIONS,
    quantile_points: int = DEFAULT_QUANTILE_POINTS,
    seed: int = DEFAULT_SEED,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    num_shards: int = 1,
    shard_index: int = 0,
    max_layouts: int | None = None,
    checkpoint_every: int = 25,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run or resume one deterministic shard of the K1-free search."""

    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards).")
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if quantile_points < 1:
        raise ValueError("quantile_points must be at least 1.")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1.")

    layouts = list(enumerate_k1_free_layouts())
    if len(layouts) != EXPECTED_LAYOUT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_LAYOUT_COUNT:,} K1-free layouts, "
            f"enumerated {len(layouts):,}."
        )
    assigned = [
        layout for layout in layouts if layout.index % num_shards == shard_index
    ]
    if max_layouts is not None:
        assigned = assigned[:max_layouts]

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{shard_index:05d}_of_{num_shards:05d}"
    results_path = output_dir / f"results_{stem}.jsonl"
    summary_path = output_dir / f"summary_{stem}.json"
    if overwrite:
        results_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)

    settings = _settings(
        outcomes=outcomes,
        target_accuracy=target_accuracy,
        iterations=iterations,
        quantile_points=quantile_points,
        seed=seed,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        num_shards=num_shards,
        shard_index=shard_index,
    )
    completed = _load_jsonl(results_path)
    if completed:
        first_record = next(iter(completed.values()))
        if not _record_settings_match(first_record, settings):
            raise ValueError(
                f"{results_path} contains results with different settings. "
                "Use a different output directory or pass --overwrite."
            )

    pending = [layout for layout in assigned if layout.index not in completed]
    estimated_hours = len(pending) * REFERENCE_SECONDS_PER_LAYOUT / 3600.0
    print(
        f"K1-free layouts: {len(layouts):,}; shard layouts: {len(assigned):,}; "
        f"already complete: {len(assigned) - len(pending):,}; pending: {len(pending):,}"
    )
    print(
        f"Estimated pending time for one worker: {estimated_hours:.2f} hours "
        f"at {REFERENCE_SECONDS_PER_LAYOUT:.2f} seconds/layout."
    )
    if not pending:
        summary = _summary(
            completed,
            settings=settings,
            assigned_layouts=len(assigned),
            elapsed_seconds=0.0,
        )
        _write_json_atomic(summary_path, summary)
        return summary

    payload = _without_candidates(
        load_empirical_outcomes(outcomes),
        REMOVED_CANDIDATES,
    )
    validation_payload, holdout_payload, split = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=seed,
    )
    validation_optimizer = HierarchyOptimizer(
        validation_payload,
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )
    holdout_optimizer = HierarchyOptimizer(
        holdout_payload,
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )

    started = perf_counter()
    with results_path.open("a", encoding="utf-8", buffering=1) as results_file:
        progress = tqdm(pending, desc=f"K1-free layouts ({stem})")
        for completed_this_run, indexed in enumerate(progress, start=1):
            cascade = indexed.cascade
            if cascade.initial == [cascade.detector]:
                validation_metrics = _direct_detector_metrics(
                    validation_optimizer,
                    cascade,
                    target_accuracy,
                )
                holdout_metrics = _direct_detector_metrics(
                    holdout_optimizer,
                    cascade,
                    target_accuracy,
                )
            else:
                validation_evaluator = FixedLayoutThresholdEvaluator(
                    validation_optimizer,
                    cascade,
                )
                holdout_evaluator = FixedLayoutThresholdEvaluator(
                    holdout_optimizer,
                    cascade,
                )
                validation_metrics = (
                    optimize_fixed_layout_thresholds_simulated_annealing(
                        validation_evaluator,
                        target_accuracy,
                        quantile_points=quantile_points,
                        n_iterations=iterations,
                        random_seed=seed,
                        show_progress=False,
                    )
                )
                holdout_metrics = holdout_evaluator.evaluate(
                    validation_metrics["thresholds"]
                )

            validation_metrics = dict(validation_metrics)
            holdout_metrics = dict(holdout_metrics)
            validation_metrics["feasible"] = bool(
                validation_metrics["accuracy"] >= target_accuracy
            )
            holdout_metrics["feasible"] = bool(
                holdout_metrics["accuracy"] >= target_accuracy
            )
            record = {
                "layout_index": indexed.index,
                "layout_id": indexed.layout_id,
                "layout": _cascade_payload(cascade),
                "settings": settings,
                "split": split,
                "validation": _compact_optimization(validation_metrics),
                "holdout": _compact_optimization(holdout_metrics),
                "holdout_feasible": bool(holdout_metrics["feasible"]),
            }
            results_file.write(
                json.dumps(record, sort_keys=True, default=float) + "\n"
            )
            completed[indexed.index] = record

            best = min(completed.values(), key=_layout_selection_key)
            best_validation = best["validation"]
            assert isinstance(best_validation, Mapping)
            progress.set_postfix(
                best_cost=f"{float(best_validation['expected_cost']):.1f}",
                best_acc=f"{float(best_validation['accuracy']):.4f}",
            )
            if completed_this_run % checkpoint_every == 0:
                _write_json_atomic(
                    summary_path,
                    _summary(
                        completed,
                        settings=settings,
                        assigned_layouts=len(assigned),
                        elapsed_seconds=perf_counter() - started,
                    ),
                )

    summary = _summary(
        completed,
        settings=settings,
        assigned_layouts=len(assigned),
        elapsed_seconds=perf_counter() - started,
    )
    _write_json_atomic(summary_path, summary)
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    return summary


def merge_results(output_dir: Path) -> dict[str, object]:
    """Merge all shard JSONL files in an output directory."""

    records: dict[int, dict] = {}
    paths = sorted(output_dir.glob("results_shard_*_of_*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No shard result files found under {output_dir}")
    for path in paths:
        records.update(_load_jsonl(path))
    best = min(records.values(), key=_layout_selection_key)
    payload = {
        "source_files": [str(path) for path in paths],
        "expected_total_layouts": EXPECTED_LAYOUT_COUNT,
        "completed_unique_layouts": len(records),
        "complete": len(records) == EXPECTED_LAYOUT_COUNT,
        "best": best,
        "fig1_reference": {
            "layout": {
                "initial": ["K3", "detector"],
                "specialized": {},
            },
            "validation_target_accuracy": FIG1_K3_TARGET_ACCURACY,
            "holdout_accuracy": FIG1_K3_HOLDOUT_ACCURACY,
            "holdout_cost_ms": FIG1_K3_HOLDOUT_COST_MS,
        },
    }
    destination = output_dir / "merged_summary.json"
    _write_json_atomic(destination, payload)
    print(f"Wrote {destination}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate all 5,545 legal K1-free hierarchy layouts and anneal "
            "each layout's per-occurrence thresholds."
        )
    )
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=FIG1_K3_TARGET_ACCURACY,
        help="Fixed validation accuracy constraint shared by every layout.",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--quantile-points",
        type=int,
        default=DEFAULT_QUANTILE_POINTS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
    )
    parser.add_argument(
        "--split-strategy",
        choices=("blocked_per_run", "random_per_run"),
        default=DEFAULT_SPLIT_STRATEGY,
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--max-layouts",
        type=int,
        default=None,
        help="Only process this many assigned layouts; useful for smoke tests.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard this shard's existing results before starting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print layout counts and estimated runtime without loading outcomes.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing shard result files without running layouts.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.merge_only:
        merge_results(args.output_dir)
        return

    layouts = list(enumerate_k1_free_layouts())
    assigned_count = sum(
        layout.index % args.num_shards == args.shard_index
        for layout in layouts
    )
    if args.max_layouts is not None:
        assigned_count = min(assigned_count, args.max_layouts)
    if args.dry_run:
        hours = assigned_count * REFERENCE_SECONDS_PER_LAYOUT / 3600.0
        print(f"Total K1-free layouts: {len(layouts):,}")
        print(
            f"Shard {args.shard_index}/{args.num_shards}: "
            f"{assigned_count:,} layouts"
        )
        print(
            f"Estimated one-worker time: {hours:.2f} hours "
            f"({REFERENCE_SECONDS_PER_LAYOUT:.2f} seconds/layout)"
        )
        return

    summary = run_search(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        target_accuracy=args.target_accuracy,
        iterations=args.iterations,
        quantile_points=args.quantile_points,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        split_strategy=args.split_strategy,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_layouts=args.max_layouts,
        checkpoint_every=args.checkpoint_every,
        overwrite=args.overwrite,
    )
    best = summary.get("best")
    if best is not None:
        print("Best layout in this shard:")
        print(json.dumps(best, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
