"""Benchmark three K1-free h24 hierarchy-optimization strategies.

The comparison uses one fixed blocked validation/holdout split:

1. dynamic-programming layout optimization at registry thresholds;
2. best-of-ten 8,000-iteration Chellapilla SA on that DP layout; and
3. exhaustive enumeration of all 5,545 legal layouts, with the same restarted
    SA threshold optimizer applied independently to every layout.

Every layout uses the identical ten random seeds. Reject-all stages are pruned
on validation following the paper and that active-stage mask is frozen for the
holdout replay. All reports and figures are written beneath the selected
checkpoint directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.brute_force_k1_free_layouts import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_OUTCOMES,
    DEFAULT_SPLIT_STRATEGY,
    EXPECTED_LAYOUT_COUNT,
    REMOVED_CANDIDATES,
    IndexedLayout,
    _cascade_payload,
    _without_candidates,
    enumerate_k1_free_layouts,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    _file_sha256,
    _write_json_atomic,
)
from hierarchy_optimizer import HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


DEFAULT_OUTPUT_DIR = Path("checkpoints/k1_free_full_benchmark_h24_8k_common_seeds")
DEFAULT_ITERATIONS_PER_RESTART = 8_000
DEFAULT_RESTARTS = 10
DEFAULT_SEED = 0

_WORKER_OPTIMIZER: HierarchyOptimizer | None = None
_WORKER_TARGET_ACCURACY = 0.0
_WORKER_ITERATIONS = DEFAULT_ITERATIONS_PER_RESTART
_WORKER_RESTARTS = DEFAULT_RESTARTS
_WORKER_BASE_SEED = DEFAULT_SEED


def _compact(metrics: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "accuracy",
        "expected_cost",
        "correct",
        "total",
        "thresholds",
        "active_slots",
        "pruned_slots",
        "route_counts",
        "macro_accuracy",
        "worst_class_accuracy",
        "per_class_accuracy",
        "feasible",
        "target_accuracy",
        "method",
        "evaluations",
        "elapsed_seconds",
        "restart_count",
        "iterations_per_restart",
        "total_requested_iterations",
        "selected_restart_index",
        "selected_restart_seed",
        "restart_seeds",
        "restart_costs_ms",
        "restart_accuracies",
        "annealing_iterations",
        "annealing_evaluations",
        "annealing_elapsed_seconds",
        "annealing_accepted_moves",
        "infeasible_proposals_rejected",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def _with_constraint(
    metrics: Mapping[str, object], target_accuracy: float, method: str
) -> dict[str, object]:
    result = dict(metrics)
    result.update(
        {
            "feasible": bool(float(result["accuracy"]) >= target_accuracy),
            "target_accuracy": float(target_accuracy),
            "method": method,
            "evaluations": 1,
            "elapsed_seconds": 0.0,
        }
    )
    return result


def _selection_key(record: Mapping[str, object]) -> tuple[float, ...]:
    validation = record["validation"]
    feasible = bool(validation["feasible"])
    accuracy = float(validation["accuracy"])
    cost = float(validation["expected_cost"])
    return (
        0.0 if feasible else 1.0,
        cost if feasible else -accuracy,
        -accuracy if feasible else cost,
        float(record["layout_index"]),
    )


def _read_jsonl(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    records: dict[int, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            if line_number == len(path.read_text(encoding="utf-8").splitlines()):
                print(f"Ignoring incomplete trailing line {line_number} in {path}")
                break
            raise ValueError(f"Invalid JSONL line {line_number} in {path}") from error
        records[int(record["layout_index"])] = record
    return records


def _initialize_worker(
    outcomes: str,
    target_accuracy: float,
    iterations: int,
    restarts: int,
    base_seed: int,
) -> None:
    global _WORKER_OPTIMIZER
    global _WORKER_TARGET_ACCURACY
    global _WORKER_ITERATIONS
    global _WORKER_RESTARTS
    global _WORKER_BASE_SEED
    payload = _without_candidates(
        load_empirical_outcomes(Path(outcomes)), REMOVED_CANDIDATES
    )
    validation_payload, _, _ = split_empirical_outcomes(
        payload,
        holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
        split_strategy=DEFAULT_SPLIT_STRATEGY,
        random_seed=0,
    )
    _WORKER_OPTIMIZER = HierarchyOptimizer(
        validation_payload,
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )
    _WORKER_TARGET_ACCURACY = target_accuracy
    _WORKER_ITERATIONS = iterations
    _WORKER_RESTARTS = restarts
    _WORKER_BASE_SEED = base_seed


def _optimize_layout(indexed: IndexedLayout) -> dict[str, object]:
    if _WORKER_OPTIMIZER is None:
        raise RuntimeError("Layout worker was not initialized.")
    started = perf_counter()
    cascade = indexed.cascade
    if cascade.initial == [cascade.detector]:
        validation = _with_constraint(
            _WORKER_OPTIMIZER.evaluate_cascade(cascade),
            _WORKER_TARGET_ACCURACY,
            "direct_no_thresholds",
        )
    else:
        evaluator = FixedLayoutThresholdEvaluator(_WORKER_OPTIMIZER, cascade)
        validation = optimize_fixed_layout_thresholds_simulated_annealing(
            evaluator,
            _WORKER_TARGET_ACCURACY,
            n_iterations=_WORKER_ITERATIONS,
            restarts=_WORKER_RESTARTS,
            random_seed=_WORKER_BASE_SEED,
            show_progress=False,
        )
    return {
        "schema_version": "k1-free-layout-result/v2",
        "layout_index": indexed.index,
        "layout_id": indexed.layout_id,
        "layout": _cascade_payload(cascade),
        "validation": _compact(validation),
        "worker_completion_seconds": perf_counter() - started,
    }


def _plot_comparison(summary: Mapping[str, object], output_dir: Path) -> tuple[Path, Path]:
    methods = summary["methods"]
    keys = ("dp_fixed_thresholds", "sa_on_dp_layout", "exhaustive_joint")
    labels = ("DP layout\nfixed thresholds", "SA thresholds\non DP layout", "Exhaustive layouts\n+ SA thresholds")
    colors = ("#607D8B", "#2A9D8F", "#E76F51")
    x = np.arange(len(keys))
    validation_costs = [float(methods[key]["validation"]["expected_cost"]) for key in keys]
    holdout_costs = [float(methods[key]["holdout"]["expected_cost"]) for key in keys]
    validation_accuracies = [100.0 * float(methods[key]["validation"]["accuracy"]) for key in keys]
    holdout_accuracies = [100.0 * float(methods[key]["holdout"]["accuracy"]) for key in keys]
    completion_times = [float(methods[key]["completion_seconds"]) for key in keys]

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), layout="constrained")
    width = 0.36
    axes[0].bar(x - width / 2, validation_costs, width, label="Validation", color="#457B9D")
    axes[0].bar(x + width / 2, holdout_costs, width, label="Holdout", color="#A8DADC")
    axes[0].set_ylabel("Expected cost (ms)")
    axes[0].set_title("Expected cascade cost")
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, validation_accuracies, width, label="Validation", color="#457B9D")
    axes[1].bar(x + width / 2, holdout_accuracies, width, label="Holdout", color="#A8DADC")
    axes[1].axhline(100.0 * float(summary["target_accuracy"]), color="#333333", linestyle="--", linewidth=1.1, label="Target")
    lower = min(validation_accuracies + holdout_accuracies) - 0.15
    upper = max(validation_accuracies + holdout_accuracies) + 0.15
    axes[1].set_ylim(lower, upper)
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("End-to-end accuracy")
    axes[1].legend(frameon=False)

    bars = axes[2].bar(x, completion_times, color=colors)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Measured completion time (s, log scale)")
    axes[2].set_title("Optimizer completion time")
    axes[2].bar_label(bars, labels=[f"{value:.3g}s" for value in completion_times], padding=3, fontsize=8)

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("K1-free h24 optimization benchmark", fontsize=15)
    png_path = output_dir / "optimizer_comparison_bar_chart.png"
    pdf_path = output_dir / "optimizer_comparison_bar_chart.pdf"
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def run_benchmark(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    iterations: int = DEFAULT_ITERATIONS_PER_RESTART,
    restarts: int = DEFAULT_RESTARTS,
    seed: int = DEFAULT_SEED,
    workers: int | None = None,
    checkpoint_every: int = 25,
    overwrite: bool = False,
    max_layouts: int | None = None,
    target_accuracy: float | None = None,
) -> dict[str, object]:
    if min(iterations, restarts, checkpoint_every) < 1:
        raise ValueError("iterations, restarts, and checkpoint_every must be positive.")
    if workers is None:
        workers = min(16, max(1, (os.cpu_count() or 2) - 1))
    if workers < 1:
        raise ValueError("workers must be positive.")
    if target_accuracy is not None and not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1 inclusive.")

    layouts = list(enumerate_k1_free_layouts())
    if len(layouts) != EXPECTED_LAYOUT_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_LAYOUT_COUNT} layouts, got {len(layouts)}.")
    selected_layouts = layouts if max_layouts is None else layouts[:max_layouts]
    payload = _without_candidates(load_empirical_outcomes(outcomes), REMOVED_CANDIDATES)
    validation_payload, holdout_payload, split = split_empirical_outcomes(
        payload,
        holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
        split_strategy=DEFAULT_SPLIT_STRATEGY,
        random_seed=0,
    )
    validation_optimizer = HierarchyOptimizer(
        validation_payload, detector_mode="paper", detector_cost_ms=PAPER_DETECTOR_COST_MS
    )
    holdout_optimizer = HierarchyOptimizer(
        holdout_payload, detector_mode="paper", detector_cost_ms=PAPER_DETECTOR_COST_MS
    )

    dp_started = perf_counter()
    dp_layout = validation_optimizer.synthesize()
    dp_layout_payload = _cascade_payload(dp_layout)
    try:
        dp_indexed_layout = next(
            indexed
            for indexed in layouts
            if _cascade_payload(indexed.cascade) == dp_layout_payload
        )
    except StopIteration as error:
        raise RuntimeError("The K1-free DP layout is absent from the exhaustive space.") from error
    dp_validation_evaluator = FixedLayoutThresholdEvaluator(validation_optimizer, dp_layout)
    dp_holdout_evaluator = FixedLayoutThresholdEvaluator(holdout_optimizer, dp_layout)
    dp_validation = dp_validation_evaluator.evaluate(
        prune_reject_all_stages=True,
        strict_thresholds=True,
    )
    dp_fixed_validation_accuracy = float(dp_validation["accuracy"])
    if target_accuracy is None:
        target_accuracy = dp_fixed_validation_accuracy
        target_accuracy_source = "k1_free_dp_fixed_threshold_validation_accuracy"
    else:
        target_accuracy = float(target_accuracy)
        target_accuracy_source = "explicit_cli_or_api_override"
    dp_validation = _with_constraint(dp_validation, target_accuracy, "dp_fixed_thresholds")
    dp_holdout = _with_constraint(
        dp_holdout_evaluator.evaluate(
            strict_thresholds=True,
            active_slots=dp_validation["active_slots"],
        ),
        target_accuracy,
        "validation_pruned_policy_holdout_replay",
    )
    dp_completion = perf_counter() - dp_started
    if "K1" in dp_layout.initial or any(
        "K1" in chain for chain in dp_layout.specialized.values()
    ):
        raise AssertionError("K1 appeared in the supposedly K1-free DP layout.")

    sa_started = perf_counter()
    sa_validation = optimize_fixed_layout_thresholds_simulated_annealing(
        dp_validation_evaluator,
        target_accuracy,
        n_iterations=iterations,
        restarts=restarts,
        random_seed=seed,
        show_progress=False,
    )
    sa_completion = perf_counter() - sa_started
    sa_holdout = _with_constraint(
        dp_holdout_evaluator.evaluate(
            sa_validation["thresholds"],
            strict_thresholds=True,
            active_slots=sa_validation["active_slots"],
        ),
        target_accuracy,
        "validation_pruned_policy_holdout_replay",
    )

    settings = {
        "schema_version": "k1-free-optimizer-benchmark/v2",
        "dataset": "m3n_vc/h24",
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "removed_candidates": sorted(REMOVED_CANDIDATES),
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
        "split_seed": 0,
        "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
        "target_accuracy": target_accuracy,
        "target_accuracy_source": target_accuracy_source,
        "dp_fixed_validation_accuracy": dp_fixed_validation_accuracy,
        "dp_layout_index_in_exhaustive_space": dp_indexed_layout.index,
        "threshold_optimizer": {
            "method": f"best_of_{restarts}_chellapilla_continuous_gaussian_sa",
            "iterations_per_restart": iterations,
            "restarts": restarts,
            "restart_seeds": [seed + index for index in range(restarts)],
            "seed_strategy": "same_restart_seeds_for_every_layout",
            "strict_confidence_comparison": True,
            "prune_stages_accepting_zero_validation_samples": True,
            "freeze_validation_active_slots_on_holdout": True,
        },
        "expected_layout_count": EXPECTED_LAYOUT_COUNT,
        "selected_layout_count": len(selected_layouts),
        "base_seed": seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "settings.json"
    results_path = output_dir / "layout_results.jsonl"
    summary_path = output_dir / "summary.json"
    if overwrite:
        settings_path.unlink(missing_ok=True)
        results_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if settings_path.exists():
        existing = json.loads(settings_path.read_text(encoding="utf-8"))
        if existing != settings:
            raise ValueError(f"{output_dir} has different settings; use --overwrite.")
    else:
        _write_json_atomic(settings_path, settings)

    records = _read_jsonl(results_path)
    pending = [layout for layout in selected_layouts if layout.index not in records]
    brute_started = perf_counter()
    print(
        f"K1-free exhaustive search: {len(selected_layouts):,} layouts; "
        f"complete={len(records):,}; pending={len(pending):,}; workers={workers}"
    )

    def persist(record: dict[str, object]) -> None:
        with results_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
        records[int(record["layout_index"])] = record

    if pending and workers == 1:
        _initialize_worker(str(outcomes), target_accuracy, iterations, restarts, seed)
        for completed_now, indexed in enumerate(pending, 1):
            persist(_optimize_layout(indexed))
            if completed_now % checkpoint_every == 0 or completed_now == len(pending):
                elapsed = perf_counter() - brute_started
                eta = elapsed / completed_now * (len(pending) - completed_now)
                print(f"Completed {completed_now:,}/{len(pending):,}; ETA={eta / 60:.1f} min")
    elif pending:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(str(outcomes), target_accuracy, iterations, restarts, seed),
        ) as executor:
            futures = {executor.submit(_optimize_layout, layout): layout.index for layout in pending}
            for completed_now, future in enumerate(as_completed(futures), 1):
                persist(future.result())
                if completed_now % checkpoint_every == 0 or completed_now == len(pending):
                    elapsed = perf_counter() - brute_started
                    eta = elapsed / completed_now * (len(pending) - completed_now)
                    best = min(records.values(), key=_selection_key)
                    print(
                        f"Completed {completed_now:,}/{len(pending):,}; "
                        f"best={float(best['validation']['expected_cost']):.3f} ms; "
                        f"ETA={eta / 60:.1f} min"
                    )
    brute_completion = perf_counter() - brute_started

    expected_restart_seeds = [seed + index for index in range(restarts)]
    for record in records.values():
        validation = record["validation"]
        if validation.get("restart_seeds") not in (None, expected_restart_seeds):
            raise RuntimeError(
                f"Layout {record['layout_index']} did not use the common restart seeds."
            )
    if dp_indexed_layout.index in records:
        dp_layout_validation = records[dp_indexed_layout.index]["validation"]
        if (
            float(dp_layout_validation["expected_cost"])
            != float(sa_validation["expected_cost"])
            or float(dp_layout_validation["accuracy"])
            != float(sa_validation["accuracy"])
            or dp_layout_validation["thresholds"] != sa_validation["thresholds"]
        ):
            raise RuntimeError(
                "The exhaustive search did not reproduce the standalone DP-layout "
                "SA policy under the shared seeds."
            )

    best = min(records.values(), key=_selection_key)
    best_index = int(best["layout_index"])
    best_layout = layouts[best_index].cascade
    brute_holdout_evaluator = FixedLayoutThresholdEvaluator(
        holdout_optimizer, best_layout
    )
    brute_holdout = _with_constraint(
        brute_holdout_evaluator.evaluate(
            best["validation"]["thresholds"],
            strict_thresholds=True,
            active_slots=best["validation"]["active_slots"],
        ),
        target_accuracy,
        "validation_pruned_policy_holdout_replay",
    )
    summed_worker_seconds = sum(
        float(record["worker_completion_seconds"]) for record in records.values()
    )
    summary = {
        "settings": settings,
        "split": split,
        "target_accuracy": target_accuracy,
        "methods": {
            "dp_fixed_thresholds": {
                "completion_seconds": dp_completion,
                "layout": _cascade_payload(dp_layout),
                "validation": _compact(dp_validation),
                "holdout": _compact(dp_holdout),
            },
            "sa_on_dp_layout": {
                "completion_seconds": sa_completion,
                "layout": _cascade_payload(dp_layout),
                "validation": _compact(sa_validation),
                "holdout": _compact(sa_holdout),
            },
            "exhaustive_joint": {
                "completion_seconds": brute_completion,
                "summed_worker_seconds": summed_worker_seconds,
                "workers": workers,
                "completed_layouts": len(records),
                "layout_results": str(results_path.resolve()),
                "layout_results_sha256": _file_sha256(results_path),
                "best_layout_index": best_index,
                "best_layout_id": best["layout_id"],
                "layout": best["layout"],
                "validation": best["validation"],
                "holdout": _compact(brute_holdout),
            },
        },
    }
    _write_json_atomic(summary_path, summary)
    _plot_comparison(summary, output_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS_PER_RESTART)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--max-layouts", type=int)
    parser.add_argument(
        "--target-accuracy",
        type=float,
        help="Required validation accuracy in [0, 1]. Defaults to the fixed-threshold DP accuracy.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run_benchmark(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        iterations=args.iterations,
        restarts=args.restarts,
        seed=args.seed,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        overwrite=args.overwrite,
        max_layouts=args.max_layouts,
        target_accuracy=args.target_accuracy,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
