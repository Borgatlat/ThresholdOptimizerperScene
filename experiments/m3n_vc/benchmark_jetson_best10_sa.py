"""Run the 10-layout x 100-trial best-of-ten SA benchmark on this PC.

The input empirical packet must have been collected with the Jetson-profiled
registry, so every optimizer cost uses deployment-device timings. Thresholds
are selected exclusively on validation and then replayed once on testing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

import numpy as np

from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.benchmark_chellapilla_single_vs_best10 import (
    DEFAULT_ITERATIONS,
    DEFAULT_LAYOUT_COUNT,
    DEFAULT_LAYOUT_SEED,
    DEFAULT_MINIMUM_CLASSIFIERS,
    DEFAULT_TRIALS_PER_LAYOUT,
    DEFAULT_WORKERS,
    RESTARTS,
    _build_experiment,
    run_benchmark as run_validation_benchmark,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    _file_sha256,
    _write_json_atomic,
)
from hierarchy_optimizer import HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import FixedLayoutThresholdEvaluator, split_empirical_outcomes


DEFAULT_OUTCOMES = Path("checkpoints/empirical_outcomes_h24_jetson_nano.pkl")
DEFAULT_OUTPUT_DIR = Path("checkpoints/jetson_best10_sa_h24_target_095")
DEFAULT_TARGET_ACCURACY = 0.95


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    maximum = float(array.max())
    return {
        "minimum": float(array.min()),
        "maximum": maximum,
        "highest": maximum,
        "mean": float(mean(array)),
        "median": float(median(array)),
        "standard_deviation": (
            float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
        ),
    }


def _metrics_summary(
    packets: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, object]:
    metrics = [packet[field] for packet in packets]
    return {
        "count": len(metrics),
        "feasible_count": sum(bool(value["feasible"]) for value in metrics),
        "feasible_rate": (
            sum(bool(value["feasible"]) for value in metrics) / len(metrics)
            if metrics
            else None
        ),
        "cost_ms": _distribution(
            [float(value["expected_cost"]) for value in metrics]
        ),
        "accuracy": _distribution([float(value["accuracy"]) for value in metrics]),
    }


def _summarize(
    packets: Sequence[Mapping[str, object]],
    layout_count: int,
) -> dict[str, object]:
    per_layout: list[dict[str, object]] = []
    for layout_index in range(layout_count):
        subset = [
            packet
            for packet in packets
            if int(packet["layout_index"]) == layout_index
        ]
        per_layout.append(
            {
                "layout_index": layout_index,
                "layout_id": subset[0]["layout_id"] if subset else None,
                "layout": subset[0]["layout"] if subset else None,
                "completed_trials": len(subset),
                "validation": _metrics_summary(subset, "validation"),
                "testing": _metrics_summary(subset, "testing"),
            }
        )

    route_counts: Counter[str] = Counter()
    for packet in packets:
        route_counts.update(packet["testing"].get("route_counts", {}))
    return {
        "per_layout": per_layout,
        "pooled": {
            "validation": _metrics_summary(packets, "validation"),
            "testing": _metrics_summary(packets, "testing"),
            "testing_route_counts": dict(sorted(route_counts.items())),
            "testing_minus_validation_cost_ms": _distribution(
                [
                    float(packet["testing"]["expected_cost"])
                    - float(packet["validation"]["expected_cost"])
                    for packet in packets
                ]
            ),
            "testing_minus_validation_accuracy": _distribution(
                [
                    float(packet["testing"]["accuracy"])
                    - float(packet["validation"]["accuracy"])
                    for packet in packets
                ]
            ),
        },
    }


def _candidate_costs(payload: Mapping[str, object]) -> dict[str, float]:
    candidates = payload["candidates"]
    costs = {
        str(row.id): float(row.cost)
        for row in candidates[["id", "cost"]].itertuples(index=False)
    }
    if not costs or not np.isfinite(list(costs.values())).all():
        raise ValueError("Empirical outcomes contain missing/non-finite classifier costs.")
    return costs


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
    validation_dir = output_dir / "validation_search"
    validation_summary = run_validation_benchmark(
        outcomes=outcomes,
        output_dir=validation_dir,
        target_accuracy=target_accuracy,
        layout_count=layout_count,
        trials_per_layout=trials_per_layout,
        iterations=iterations,
        layout_seed=layout_seed,
        minimum_classifiers=minimum_classifiers,
        workers=workers,
        overwrite=overwrite,
    )
    if validation_summary["status"] != "complete":
        raise RuntimeError("Validation benchmark did not complete.")

    payload = load_empirical_outcomes(outcomes)
    _, holdout_payload, split = split_empirical_outcomes(
        payload,
        holdout_fraction=0.20,
        split_strategy="blocked_per_run",
        random_seed=0,
    )
    holdout_optimizer = HierarchyOptimizer(
        holdout_payload,
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )
    _, _, _, cascades, _ = _build_experiment(
        outcomes, layout_count, layout_seed, minimum_classifiers
    )
    evaluators = tuple(
        FixedLayoutThresholdEvaluator(holdout_optimizer, cascade)
        for cascade in cascades
    )

    validation_records = _read_jsonl(validation_dir / "trial_packets.jsonl")
    expected = layout_count * trials_per_layout
    if len(validation_records) != expected:
        raise ValueError(
            f"Expected {expected} validation records, found {len(validation_records)}."
        )
    combined: list[dict[str, object]] = []
    for record in sorted(
        validation_records,
        key=lambda value: (int(value["layout_index"]), int(value["trial_index"])),
    ):
        layout_index = int(record["layout_index"])
        validation = dict(record["best_of_10"])
        testing = dict(
            evaluators[layout_index].evaluate(
                validation["thresholds"],
                strict_thresholds=True,
                active_slots=validation.get("active_slots"),
            )
        )
        testing["feasible"] = bool(float(testing["accuracy"]) >= target_accuracy)
        testing["target_accuracy"] = float(target_accuracy)
        testing["method"] = "validation_selected_best_of_10_holdout_replay"
        combined.append(
            {
                "layout_index": layout_index,
                "layout_id": record["layout_id"],
                "layout": record["layout"],
                "trial_index": int(record["trial_index"]),
                "base_seed": int(record["base_seed"]),
                "restart_seeds": record["restart_seeds"],
                "validation": validation,
                "testing": testing,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    packets_path = output_dir / "best_of_10_validation_testing_packets.jsonl"
    packets_path.write_text(
        "".join(json.dumps(packet, sort_keys=True, default=float) + "\n" for packet in combined),
        encoding="utf-8",
    )
    collection = dict(payload.get("collection", {}))
    summary: dict[str, object] = {
        "schema_version": "jetson-cost-best10-sa-summary/v1",
        "status": "complete",
        "settings": {
            "dataset": "m3n_vc/h24",
            "outcomes": str(outcomes.resolve()),
            "outcomes_sha256": _file_sha256(outcomes),
            "empirical_collection": collection,
            "candidate_costs_ms": _candidate_costs(payload),
            "detector_mode": "paper",
            "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
            "target_accuracy": float(target_accuracy),
            "split_strategy": "blocked_per_run",
            "split_seed": 0,
            "holdout_fraction": 0.20,
            "layout_count": int(layout_count),
            "layout_seed": int(layout_seed),
            "minimum_distinct_non_detector_classifiers": int(minimum_classifiers),
            "trials_per_layout": int(trials_per_layout),
            "restarts_per_trial": RESTARTS,
            "iterations_per_restart": int(iterations),
            "workers": int(workers),
            "optimization_device": "this_PC",
            "cost_device": "registry_recorded_in_empirical_collection",
        },
        "split": split,
        "completed_trials": len(combined),
        "expected_trials": expected,
        "validation_search_summary": str(
            (validation_dir / "summary.json").resolve()
        ),
        "packets": str(packets_path.resolve()),
        **_summarize(combined, layout_count),
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY)
    parser.add_argument("--layout-count", type=int, default=DEFAULT_LAYOUT_COUNT)
    parser.add_argument("--trials-per-layout", type=int, default=DEFAULT_TRIALS_PER_LAYOUT)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--layout-seed", type=int, default=DEFAULT_LAYOUT_SEED)
    parser.add_argument("--minimum-classifiers", type=int, default=DEFAULT_MINIMUM_CLASSIFIERS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
    print(json.dumps(summary["pooled"], indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
