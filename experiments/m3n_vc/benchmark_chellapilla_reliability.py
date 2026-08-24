"""Measure the run-to-run reliability of the DAS 2006 threshold annealer.

The default experiment evaluates five deterministic random h24 layouts with
at least five distinct non-detector classifiers.  Each layout receives 1,000
independent 1,000-iteration runs of the continuous Gaussian simulated
annealer.  Results are append-only/resumable and plots are derived solely
from the saved trial packets.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from statistics import mean, median
from textwrap import fill
from time import perf_counter
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.benchmark_chellapilla_sa import (
    PAPER_URL,
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
    DEFAULT_TARGET_ACCURACY,
    build_k1_layout_space,
    legal_layout_count,
)
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from layout_search import cascade_from_genome, layout_id
from threshold_optimizer import (
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_chellapilla_sa,
    split_empirical_outcomes,
)


DEFAULT_OUTPUT_DIR = Path("checkpoints/chellapilla_sa_reliability_h24")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/chellapilla_sa_reliability_h24")
DEFAULT_LAYOUT_COUNT = 5
DEFAULT_TRIAL_COUNT = 1_000
DEFAULT_ITERATIONS = 1_000
DEFAULT_LAYOUT_SEED = 20260818
DEFAULT_MINIMUM_CLASSIFIERS = 5
TRIAL_SEED_STRIDE = 1_000_000

_WORKER_EVALUATORS: tuple[FixedLayoutThresholdEvaluator, ...] = ()


def _build_experiment(
    outcomes: Path,
    layout_count: int,
    layout_seed: int,
    minimum_classifiers: int,
):
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
        space, layout_count, layout_seed, minimum_classifiers
    )
    cascades = tuple(cascade_from_genome(genome, space) for genome in genomes)
    evaluators = tuple(
        FixedLayoutThresholdEvaluator(optimizer, cascade) for cascade in cascades
    )
    return payload, split, space, genomes, cascades, evaluators


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


def _run_trial(task: tuple[int, int, int, float, int]) -> tuple[int, int, int, dict]:
    layout_index, trial_index, trial_seed, target_accuracy, iterations = task
    result = optimize_fixed_layout_thresholds_chellapilla_sa(
        _WORKER_EVALUATORS[layout_index],
        target_accuracy,
        n_iterations=iterations,
        random_seed=trial_seed,
        show_progress=False,
    )
    return layout_index, trial_index, trial_seed, _compact(result)


def _read_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL record {line_number} in {path}") from error
    return records


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    return {
        label: float(np.quantile(values, quantile))
        for label, quantile in (
            ("p01", 0.01),
            ("p05", 0.05),
            ("p10", 0.10),
            ("p25", 0.25),
            ("p50", 0.50),
            ("p75", 0.75),
            ("p90", 0.90),
            ("p95", 0.95),
            ("p99", 0.99),
        )
    }


def _summarize(
    records: Sequence[Mapping[str, object]], layout_count: int
) -> dict[str, object]:
    layouts: list[dict[str, object]] = []
    for layout_index in range(layout_count):
        subset = [
            record for record in records if int(record["layout_index"]) == layout_index
        ]
        if not subset:
            continue
        results = [record["validation"] for record in subset]
        costs = [float(result["expected_cost"]) for result in results]
        accuracies = [float(result["accuracy"]) for result in results]
        best_record = min(
            subset,
            key=lambda record: (
                not bool(record["validation"]["feasible"]),
                float(record["validation"]["expected_cost"]),
                -float(record["validation"]["accuracy"]),
            ),
        )
        layouts.append(
            {
                "layout_index": layout_index,
                "layout_id": subset[0]["layout_id"],
                "layout": subset[0]["layout"],
                "distinct_classifier_count": subset[0]["distinct_classifier_count"],
                "trials": len(subset),
                "feasible_trials": sum(bool(result["feasible"]) for result in results),
                "cost_ms": {
                    "minimum": min(costs),
                    "maximum": max(costs),
                    "mean": mean(costs),
                    "median": median(costs),
                    "standard_deviation": float(np.std(costs, ddof=1)),
                    "quantiles": _quantiles(costs),
                },
                "accuracy": {
                    "minimum": min(accuracies),
                    "maximum": max(accuracies),
                    "mean": mean(accuracies),
                    "median": median(accuracies),
                },
                "mean_elapsed_seconds": mean(
                    float(result["elapsed_seconds"]) for result in results
                ),
                "best_trial": {
                    "trial_index": best_record["trial_index"],
                    "trial_seed": best_record["trial_seed"],
                    "validation": best_record["validation"],
                },
            }
        )
    return {"completed_trials": len(records), "layouts": layouts}


def _cascade_from_payload(layout: Mapping[str, object]) -> Cascade:
    specialized_raw = layout.get("specialized", {})
    if not isinstance(specialized_raw, Mapping):
        raise ValueError("layout.specialized must be an object.")
    specialized: dict[tuple[str, str], list[str]] = {}
    for key, chain in specialized_raw.items():
        router_id, group = str(key).split(":", maxsplit=1)
        if not isinstance(chain, Sequence) or isinstance(chain, (str, bytes)):
            raise ValueError(f"Specialized chain {key!r} must be a list.")
        specialized[(router_id, group)] = [str(value) for value in chain]
    return Cascade(
        expected_cost=0.0,
        initial=[str(value) for value in layout.get("initial", [])],
        specialized=specialized,
        detector="detector",
    )


def _cost_distribution(values: Sequence[float]) -> dict[str, object]:
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean(values),
        "median": median(values),
        "standard_deviation": float(np.std(values, ddof=1)),
        "quantiles": _quantiles(values),
        "fraction_at_most_800_ms": sum(value <= 800.0 for value in values) / len(values),
        "fraction_over_1000_ms": sum(value > 1_000.0 for value in values) / len(values),
        "fraction_over_2000_ms": sum(value > 2_000.0 for value in values) / len(values),
    }


def replay_holdout(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    validation_records_path: Path = DEFAULT_OUTPUT_DIR / "trial_packets.jsonl",
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> dict[str, object]:
    """Replay every saved validation policy once on the untouched holdout."""
    validation_records = _read_records(validation_records_path)
    if not validation_records:
        raise ValueError(f"No validation packets found in {validation_records_path}.")
    payload = load_empirical_outcomes(outcomes)
    _, holdout_payload, split = split_empirical_outcomes(
        payload,
        holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
        split_strategy=DEFAULT_SPLIT_STRATEGY,
        random_seed=0,
    )
    target_accuracy = float(validation_records[0]["validation"]["target_accuracy"])
    optimizer = HierarchyOptimizer(
        holdout_payload,
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )
    evaluators: dict[int, FixedLayoutThresholdEvaluator] = {}
    for record in validation_records:
        layout_index = int(record["layout_index"])
        if layout_index not in evaluators:
            evaluators[layout_index] = FixedLayoutThresholdEvaluator(
                optimizer, _cascade_from_payload(record["layout"])
            )

    replayed: list[dict[str, object]] = []
    started = perf_counter()
    for position, source in enumerate(validation_records, 1):
        evaluator = evaluators[int(source["layout_index"])]
        evaluation_started = perf_counter()
        metrics = evaluator.evaluate(source["validation"]["thresholds"])
        elapsed = perf_counter() - evaluation_started
        metrics.update(
            {
                "feasible": bool(float(metrics["accuracy"]) >= target_accuracy),
                "target_accuracy": target_accuracy,
                "method": "fixed_policy_holdout_replay",
                "evaluations": 1,
                "elapsed_seconds": elapsed,
            }
        )
        record = dict(source)
        record["holdout"] = _compact(metrics)
        replayed.append(record)
        if position % 500 == 0:
            print(f"Replayed {position:,}/{len(validation_records):,} policies on holdout")

    output_dir.mkdir(parents=True, exist_ok=True)
    holdout_records_path = output_dir / "holdout_trial_packets.jsonl"
    temporary_path = holdout_records_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in replayed:
            handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
    temporary_path.replace(holdout_records_path)

    layout_summaries: list[dict[str, object]] = []
    for layout_index in sorted(evaluators):
        subset = [
            record for record in replayed if int(record["layout_index"]) == layout_index
        ]
        validation_costs = [
            float(record["validation"]["expected_cost"]) for record in subset
        ]
        holdout_costs = [float(record["holdout"]["expected_cost"]) for record in subset]
        cost_deltas = [
            holdout - validation
            for validation, holdout in zip(validation_costs, holdout_costs, strict=True)
        ]
        validation_accuracies = [float(record["validation"]["accuracy"]) for record in subset]
        holdout_accuracies = [float(record["holdout"]["accuracy"]) for record in subset]
        layout_summaries.append(
            {
                "layout_index": layout_index,
                "layout_id": subset[0]["layout_id"],
                "layout": subset[0]["layout"],
                "trials": len(subset),
                "validation_cost_ms": _cost_distribution(validation_costs),
                "holdout_cost_ms": _cost_distribution(holdout_costs),
                "paired_holdout_minus_validation_cost_ms": {
                    "mean": mean(cost_deltas),
                    "median": median(cost_deltas),
                    "minimum": min(cost_deltas),
                    "maximum": max(cost_deltas),
                },
                "cost_pearson_correlation": float(
                    np.corrcoef(validation_costs, holdout_costs)[0, 1]
                ),
                "validation_accuracy": {
                    "mean": mean(validation_accuracies),
                    "minimum": min(validation_accuracies),
                    "maximum": max(validation_accuracies),
                },
                "holdout_accuracy": {
                    "mean": mean(holdout_accuracies),
                    "minimum": min(holdout_accuracies),
                    "maximum": max(holdout_accuracies),
                },
                "holdout_feasible_trials": sum(
                    bool(record["holdout"]["feasible"]) for record in subset
                ),
            }
        )
    summary = {
        "dataset": "m3n_vc/h24",
        "source_validation_packets": str(validation_records_path.resolve()),
        "source_validation_packets_sha256": _file_sha256(validation_records_path),
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "partition": "holdout",
        "policy_selection_partition": "validation",
        "target_accuracy": target_accuracy,
        "split": split,
        "trials": len(replayed),
        "elapsed_seconds": perf_counter() - started,
        "layouts": layout_summaries,
    }
    _write_json_atomic(output_dir / "holdout_summary.json", summary)
    plot_histograms(
        holdout_records_path,
        figures_dir,
        partition="holdout",
        filename_prefix="paper_sa_holdout",
    )
    return summary


def _layout_caption(layout: Mapping[str, object]) -> str:
    initial = " -> ".join(str(value) for value in layout.get("initial", []))
    branch_parts = []
    specialized = layout.get("specialized", {})
    if isinstance(specialized, Mapping):
        for router_or_branch, groups_or_chain in specialized.items():
            if isinstance(groups_or_chain, Mapping):
                for group, chain in groups_or_chain.items():
                    if chain:
                        branch_parts.append(
                            f"{router_or_branch}:{group}"
                            f"[{' -> '.join(str(value) for value in chain)}]"
                        )
            elif groups_or_chain:
                branch_parts.append(
                    f"{router_or_branch}"
                    f"[{' -> '.join(str(value) for value in groups_or_chain)}]"
                )
    return " | ".join([initial or "detector only", *branch_parts])


def plot_histograms(
    records_path: Path,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    *,
    partition: str = "validation",
    filename_prefix: str = "paper_sa",
) -> list[Path]:
    """Plot cost distributions using only the persisted trial packets."""
    records = _read_records(records_path)
    if not records:
        raise ValueError(f"No trial packets found in {records_path}.")
    layout_indices = sorted({int(record["layout_index"]) for record in records})
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap("tab10").colors
    saved: list[Path] = []

    rows = 2 if len(layout_indices) > 3 else 1
    columns = min(3, len(layout_indices))
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.5 * columns, 4.3 * rows), squeeze=False,
    )
    figure.subplots_adjust(top=0.84, bottom=0.08, hspace=0.62, wspace=0.13)
    flat_axes = list(axes.flat)

    if partition not in {"validation", "holdout"}:
        raise ValueError("partition must be 'validation' or 'holdout'.")
    costs_by_layout = {
        index: np.asarray(
            [
                float(record[partition]["expected_cost"])
                for record in records
                if int(record["layout_index"]) == index
            ],
            dtype=float,
        )
        for index in layout_indices
    }
    all_costs = np.concatenate(tuple(costs_by_layout.values()))
    if float(np.min(all_costs)) == float(np.max(all_costs)):
        common_bins = np.linspace(float(all_costs[0]) - 0.5, float(all_costs[0]) + 0.5, 3)
    else:
        common_bins = np.linspace(float(np.min(all_costs)), float(np.max(all_costs)), 61)

    for position, layout_index in enumerate(layout_indices):
        subset = [
            record for record in records if int(record["layout_index"]) == layout_index
        ]
        costs = costs_by_layout[layout_index]
        layout = subset[0]["layout"]
        color = colors[position % len(colors)]

        def draw(axis, *, detailed: bool) -> None:
            axis.hist(costs, bins=common_bins, color=color, alpha=0.78, edgecolor="white")
            axis.axvline(float(np.min(costs)), color="#2b8cbe", linewidth=1.5, label="Best")
            axis.axvline(float(np.median(costs)), color="#238b45", linewidth=1.5, label="Median")
            axis.axvline(float(np.mean(costs)), color="#cb181d", linewidth=1.5, label="Mean")
            axis.set_xlabel(
                "Best feasible validation cost found (ms)"
                if partition == "validation"
                else "Holdout expected cost of validation-selected policy (ms)"
            )
            axis.set_ylabel("Trials")
            axis.grid(axis="y", alpha=0.22)
            axis.set_title(f"Layout {layout_index + 1} (n={len(costs):,})")
            if detailed:
                axis.text(
                    0.99,
                    0.97,
                    "\n".join(
                        (
                            f"best {np.min(costs):.1f} ms",
                            f"median {np.median(costs):.1f} ms",
                            f"mean {np.mean(costs):.1f} ms",
                            f"p95 {np.quantile(costs, 0.95):.1f} ms",
                        )
                    ),
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#cccccc"},
                )
            axis.legend(fontsize=8, loc="upper right" if not detailed else "upper left")

        draw(flat_axes[position], detailed=False)
        if position % columns:
            flat_axes[position].set_ylabel("")
        initial_caption = " -> ".join(
            str(value) for value in layout.get("initial", [])
        )
        flat_axes[position].set_title(
            f"Layout {layout_index + 1} (n={len(costs):,})\n"
            f"Initial: {initial_caption}\nID: {subset[0]['layout_id']}",
            fontsize=10,
        )

        individual, individual_axis = plt.subplots(
            figsize=(8.2, 5.1), layout="constrained"
        )
        draw(individual_axis, detailed=True)
        individual_axis.set_title(
            f"Chellapilla SA reliability — Layout {layout_index + 1}\n"
            f"{fill(_layout_caption(layout), 88)}"
        )
        partition_suffix = "" if partition == "validation" else f"_{partition}"
        individual_path = figures_dir / (
            f"layout_{layout_index + 1:02d}{partition_suffix}_cost_histogram.png"
        )
        individual.savefig(individual_path, dpi=200)
        plt.close(individual)
        saved.append(individual_path)

    for axis in flat_axes[len(layout_indices) :]:
        axis.remove()
    figure.suptitle(
        "Chellapilla continuous SA: 1,000 independent 1,000-iteration trials per layout"
        if partition == "validation"
        else "Holdout replay of 1,000 validation-optimized SA policies per layout",
        fontsize=14,
        y=0.975,
    )
    combined_png = figures_dir / f"{filename_prefix}_cost_histograms.png"
    combined_pdf = figures_dir / f"{filename_prefix}_cost_histograms.pdf"
    figure.savefig(combined_png, dpi=200)
    figure.savefig(combined_pdf)
    plt.close(figure)
    saved.extend((combined_png, combined_pdf))
    return saved


def run_benchmark(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    layout_count: int = DEFAULT_LAYOUT_COUNT,
    trial_count: int = DEFAULT_TRIAL_COUNT,
    iterations: int = DEFAULT_ITERATIONS,
    layout_seed: int = DEFAULT_LAYOUT_SEED,
    minimum_classifiers: int = DEFAULT_MINIMUM_CLASSIFIERS,
    workers: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if min(layout_count, trial_count, iterations, minimum_classifiers) < 1:
        raise ValueError("Layout, trial, iteration, and classifier counts must be positive.")
    if workers is None:
        workers = min(16, max(1, (os.cpu_count() or 2) - 1))
    if workers < 1:
        raise ValueError("workers must be positive.")

    payload, split, space, genomes, cascades, evaluators = _build_experiment(
        outcomes, layout_count, layout_seed, minimum_classifiers
    )
    settings = {
        "dataset": "m3n_vc/h24",
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "partition": "validation",
        "holdout_usage": "not_evaluated",
        "split_strategy": DEFAULT_SPLIT_STRATEGY,
        "split_seed": 0,
        "target_accuracy": float(target_accuracy),
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "optimizer": {
            "method": "chellapilla_continuous_gaussian_sa",
            "citation": "Chellapilla, Shilman, and Simard (DAS 2006)",
            "url": PAPER_URL,
            "iterations": int(iterations),
            "continuous_thresholds": True,
            "post_sa_polisher": False,
            "random_global_proposal": False,
        },
        "layout_sampling": "uniform_over_legal_layouts_conditioned_on_minimum",
        "legal_layout_space_size": legal_layout_count(space),
        "layout_count": int(layout_count),
        "minimum_distinct_non_detector_classifiers": int(minimum_classifiers),
        "layout_seed": int(layout_seed),
        "trial_count_per_layout": int(trial_count),
        "trial_seed_rule": f"layout_index * {TRIAL_SEED_STRIDE} + trial_index",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "settings.json"
    records_path = output_dir / "trial_packets.jsonl"
    summary_path = output_dir / "summary.json"
    if overwrite:
        settings_path.unlink(missing_ok=True)
        records_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if settings_path.exists():
        existing = json.loads(settings_path.read_text(encoding="utf-8"))
        if existing != settings:
            raise ValueError(f"{output_dir} has different settings; use --overwrite.")
    else:
        _write_json_atomic(settings_path, settings)

    records = _read_records(records_path)
    completed = {
        (int(record["layout_index"]), int(record["trial_index"]))
        for record in records
    }
    tasks = [
        (
            layout_index,
            trial_index,
            layout_index * TRIAL_SEED_STRIDE + trial_index,
            float(target_accuracy),
            int(iterations),
        )
        for layout_index in range(layout_count)
        for trial_index in range(trial_count)
        if (layout_index, trial_index) not in completed
    ]
    layout_metadata = {
        index: {
            "layout_id": layout_id(genome, space),
            "layout": _cascade_payload(cascade),
            "distinct_classifier_count": len(_classifier_ids(genome)),
        }
        for index, (genome, cascade) in enumerate(zip(genomes, cascades, strict=True))
    }
    started = perf_counter()

    def persist(worker_result: tuple[int, int, int, dict]) -> None:
        layout_index, trial_index, trial_seed, result = worker_result
        metadata = layout_metadata[layout_index]
        record = {
            "schema_version": "threshold-optimization-trial/v1",
            "dataset": "m3n_vc/h24",
            "partition": "validation",
            "method": "chellapilla_continuous_gaussian_sa",
            "iterations": int(iterations),
            "layout_index": layout_index,
            "layout_id": metadata["layout_id"],
            "layout": metadata["layout"],
            "distinct_classifier_count": metadata["distinct_classifier_count"],
            "trial_index": trial_index,
            "trial_seed": trial_seed,
            "validation": result,
        }
        records.append(record)
        with records_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")

    if tasks and workers == 1:
        global _WORKER_EVALUATORS
        _WORKER_EVALUATORS = evaluators
        for completed_now, task in enumerate(tasks, 1):
            persist(_run_trial(task))
            if completed_now % 50 == 0 or completed_now == len(tasks):
                elapsed = perf_counter() - started
                eta = elapsed / completed_now * (len(tasks) - completed_now)
                print(f"Completed {completed_now:,}/{len(tasks):,} pending trials; ETA {eta / 60:.1f} min")
    elif tasks:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(str(outcomes), layout_count, layout_seed, minimum_classifiers),
        ) as executor:
            futures = [executor.submit(_run_trial, task) for task in tasks]
            for completed_now, future in enumerate(as_completed(futures), 1):
                persist(future.result())
                if completed_now % 50 == 0 or completed_now == len(tasks):
                    elapsed = perf_counter() - started
                    eta = elapsed / completed_now * (len(tasks) - completed_now)
                    print(
                        f"Completed {completed_now:,}/{len(tasks):,} pending trials "
                        f"with {workers} workers; ETA {eta / 60:.1f} min"
                    )

    records.sort(key=lambda record: (int(record["layout_index"]), int(record["trial_index"])))
    summary = {
        "settings": settings,
        "split": split,
        "wall_elapsed_seconds_this_invocation": perf_counter() - started,
        **_summarize(records, layout_count),
    }
    _write_json_atomic(summary_path, summary)
    plot_histograms(records_path, figures_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY)
    parser.add_argument("--layout-count", type=int, default=DEFAULT_LAYOUT_COUNT)
    parser.add_argument("--trial-count", type=int, default=DEFAULT_TRIAL_COUNT)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--layout-seed", type=int, default=DEFAULT_LAYOUT_SEED)
    parser.add_argument("--minimum-classifiers", type=int, default=DEFAULT_MINIMUM_CLASSIFIERS)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--replay-holdout",
        action="store_true",
        help="Replay existing validation trial packets on the untouched holdout.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.replay_holdout:
        summary = replay_holdout(
            outcomes=args.outcomes,
            validation_records_path=args.output_dir / "trial_packets.jsonl",
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
        )
        print(json.dumps(summary, indent=2, sort_keys=True, default=float))
        return
    summary = run_benchmark(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        target_accuracy=args.target_accuracy,
        layout_count=args.layout_count,
        trial_count=args.trial_count,
        iterations=args.iterations,
        layout_seed=args.layout_seed,
        minimum_classifiers=args.minimum_classifiers,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
