"""Audit a pure K3 -> detector cascade against the K1-free benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.brute_force_k1_free_layouts import _without_candidates
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    _file_sha256,
    _write_json_atomic,
)
from hierarchy_optimizer import Cascade, HierarchyOptimizer
from threshold_optimizer import (
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


DEFAULT_BENCHMARK_SUMMARY = Path(
    "checkpoints/k1_free_full_benchmark_h24/summary.json"
)
DEFAULT_PREVIOUS_RESULT = Path(
    "checkpoints/result_packets/m3n_vc_h24/linear_k3.json"
)
DEFAULT_OUTPUT_DIR = Path("checkpoints/k3_linear_sa_audit_h24")


def _selection_key(metrics: Mapping[str, object], target: float) -> tuple[float, ...]:
    accuracy = float(metrics["accuracy"])
    cost = float(metrics["expected_cost"])
    feasible = accuracy >= target
    return (
        0.0 if feasible else 1.0,
        cost if feasible else -accuracy,
        -accuracy if feasible else cost,
    )


def _with_constraint(metrics: Mapping[str, object], target: float, method: str) -> dict:
    result = dict(metrics)
    result.update(
        {
            "feasible": bool(float(result["accuracy"]) >= target),
            "target_accuracy": float(target),
            "method": method,
        }
    )
    return result


def _exact_one_dimensional_search(
    evaluator: FixedLayoutThresholdEvaluator,
    target: float,
) -> dict:
    """Enumerate every empirically distinct strict K3 acceptance policy."""

    confidence = evaluator.confidence["K3"]
    thresholds = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, 1.0]),
                np.nextafter(np.unique(confidence), -np.inf),
            )
        )
    )
    thresholds = thresholds[(thresholds >= 0.0) & (thresholds <= 1.0)]
    best: dict | None = None
    started = perf_counter()
    for threshold in thresholds:
        metrics = evaluator.evaluate(
            {"K3": float(threshold)},
            include_route_counts=False,
            include_class_metrics=False,
            prune_reject_all_stages=True,
            strict_thresholds=True,
        )
        if best is None or _selection_key(metrics, target) < _selection_key(best, target):
            best = metrics
    if best is None:
        raise RuntimeError("The exact K3 threshold search evaluated no policies.")
    result = evaluator.evaluate(
        best["thresholds"],
        prune_reject_all_stages=True,
        strict_thresholds=True,
    )
    return _with_constraint(
        {
            **result,
            "evaluations": int(len(thresholds)),
            "elapsed_seconds": float(perf_counter() - started),
        },
        target,
        "exact_empirical_1d_strict_threshold_search",
    )


def _holdout_replay(
    evaluator: FixedLayoutThresholdEvaluator,
    validation: Mapping[str, object],
    target: float,
) -> dict:
    metrics = evaluator.evaluate(
        validation["thresholds"],
        strict_thresholds=True,
        active_slots=validation["active_slots"],
    )
    return _with_constraint(metrics, target, "validation_policy_holdout_replay")


def _cost_decomposition(metrics: Mapping[str, object], detector_cost: float) -> dict:
    detector_routes = int(metrics["route_counts"].get("detector", 0))
    total = int(metrics["total"])
    detector_contribution = detector_cost * detector_routes / total
    return {
        "detector_routes": detector_routes,
        "detector_contribution_ms": float(detector_contribution),
        "non_detector_contribution_ms": float(
            float(metrics["expected_cost"]) - detector_contribution
        ),
    }


def _load_previous_run_contract(previous: Mapping[str, object]) -> dict:
    source = Path(previous["provenance"]["source"])
    if not source.exists():
        return {"source": str(source), "available": False}
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["layout"]["initial"] == ["K3", "detector"]:
                validation = record["validation"]
                settings = record["settings"]
                return {
                    "source": str(source.resolve()),
                    "available": True,
                    "method": validation["method"],
                    "iterations": int(settings["iterations"]),
                    "quantile_points": int(settings["quantile_points"]),
                    "coordinate_descent_passes": int(
                        validation.get("coordinate_descent_passes", 0)
                    ),
                    "seed": int(settings["seed"]),
                    "target_accuracy": float(settings["target_accuracy"]),
                }
    raise RuntimeError(f"No K3 -> detector result was found in {source}.")


def run_audit(
    *,
    benchmark_summary: Path = DEFAULT_BENCHMARK_SUMMARY,
    previous_result: Path = DEFAULT_PREVIOUS_RESULT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict:
    benchmark = json.loads(benchmark_summary.read_text(encoding="utf-8"))
    previous = json.loads(previous_result.read_text(encoding="utf-8"))
    settings = benchmark["settings"]
    target = float(benchmark["target_accuracy"])
    outcomes_path = Path(settings["outcomes"])
    payload = _without_candidates(
        load_empirical_outcomes(outcomes_path), settings["removed_candidates"]
    )
    validation_payload, holdout_payload, split = split_empirical_outcomes(
        payload,
        holdout_fraction=float(settings["holdout_fraction"]),
        split_strategy=str(settings["split_strategy"]),
        random_seed=int(settings["split_seed"]),
    )
    validation_optimizer = HierarchyOptimizer(
        validation_payload,
        detector_mode=str(settings["detector_mode"]),
        detector_cost_ms=float(settings["detector_cost_ms"]),
    )
    holdout_optimizer = HierarchyOptimizer(
        holdout_payload,
        detector_mode=str(settings["detector_mode"]),
        detector_cost_ms=float(settings["detector_cost_ms"]),
    )
    cascade = Cascade(
        expected_cost=0.0,
        initial=["K3", validation_optimizer.detector_id],
        specialized={},
        detector=validation_optimizer.detector_id,
    )
    validation_evaluator = FixedLayoutThresholdEvaluator(
        validation_optimizer, cascade
    )
    holdout_evaluator = FixedLayoutThresholdEvaluator(holdout_optimizer, cascade)

    annealed_validation = optimize_fixed_layout_thresholds_simulated_annealing(
        validation_evaluator,
        target,
        n_iterations=1_000,
        restarts=10,
        random_seed=0,
        show_progress=False,
    )
    annealed_holdout = _holdout_replay(
        holdout_evaluator, annealed_validation, target
    )
    exact_validation = _exact_one_dimensional_search(validation_evaluator, target)
    exact_holdout = _holdout_replay(holdout_evaluator, exact_validation, target)
    previous_target = float(previous["method"]["target_accuracy"])
    previous_target_exact_validation = _exact_one_dimensional_search(
        validation_evaluator, previous_target
    )
    previous_target_exact_holdout = _holdout_replay(
        holdout_evaluator, previous_target_exact_validation, previous_target
    )

    dp = benchmark["methods"]["sa_on_dp_layout"]
    joint = benchmark["methods"]["exhaustive_joint"]
    detector_cost = float(settings["detector_cost_ms"])
    comparison = {
        "pure_minus_dp_validation_cost_ms": float(
            annealed_validation["expected_cost"] - dp["validation"]["expected_cost"]
        ),
        "pure_minus_dp_holdout_cost_ms": float(
            annealed_holdout["expected_cost"] - dp["holdout"]["expected_cost"]
        ),
        "pure_minus_joint_validation_cost_ms": float(
            annealed_validation["expected_cost"]
            - joint["validation"]["expected_cost"]
        ),
        "pure_minus_joint_holdout_cost_ms": float(
            annealed_holdout["expected_cost"] - joint["holdout"]["expected_cost"]
        ),
        "pure_validation_cost_decomposition": _cost_decomposition(
            annealed_validation, detector_cost
        ),
        "dp_validation_cost_decomposition": _cost_decomposition(
            dp["validation"], detector_cost
        ),
        "pure_holdout_cost_decomposition": _cost_decomposition(
            annealed_holdout, detector_cost
        ),
        "dp_holdout_cost_decomposition": _cost_decomposition(
            dp["holdout"], detector_cost
        ),
        "previous_minus_current_validation_cost_ms": float(
            previous["partitions"]["validation"]["expected_cost_ms"]
            - annealed_validation["expected_cost"]
        ),
        "previous_grid_minus_exact_previous_target_validation_cost_ms": float(
            previous["partitions"]["validation"]["expected_cost_ms"]
            - previous_target_exact_validation["expected_cost"]
        ),
        "previous_target_minus_current_target_exact_validation_cost_ms": float(
            previous_target_exact_validation["expected_cost"]
            - exact_validation["expected_cost"]
        ),
    }
    result = {
        "schema_version": "k3-linear-sa-audit/v1",
        "settings": {
            "dataset": settings["dataset"],
            "outcomes": str(outcomes_path.resolve()),
            "outcomes_sha256": _file_sha256(outcomes_path),
            "removed_candidates": list(settings["removed_candidates"]),
            "split_strategy": settings["split_strategy"],
            "split_seed": int(settings["split_seed"]),
            "holdout_fraction": float(settings["holdout_fraction"]),
            "target_accuracy": target,
            "restart_seeds": list(range(10)),
            "iterations_per_restart": 1_000,
            "acceptance": "confidence_strictly_greater_than_threshold",
            "reject_all_stage_pruning": True,
        },
        "split": split,
        "layout": {"initial": ["K3", "detector"], "specialized": {}},
        "pure_k3_paper_sa": {
            "validation": annealed_validation,
            "holdout": annealed_holdout,
        },
        "exact_empirical_1d_oracle": {
            "validation": exact_validation,
            "holdout": exact_holdout,
        },
        "exact_empirical_1d_oracle_at_previous_target": {
            "target_accuracy": previous_target,
            "validation": previous_target_exact_validation,
            "holdout": previous_target_exact_holdout,
        },
        "completed_benchmark_references": {
            "source": str(benchmark_summary.resolve()),
            "sa_on_dp_layout": dp,
            "exhaustive_joint": joint,
        },
        "previous_linear_k3": {
            "source": str(previous_result.resolve()),
            "optimizer_contract": _load_previous_run_contract(previous),
            "result_packet": previous,
        },
        "comparison": comparison,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-summary", type=Path, default=DEFAULT_BENCHMARK_SUMMARY)
    parser.add_argument("--previous-result", type=Path, default=DEFAULT_PREVIOUS_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run_audit(
        benchmark_summary=args.benchmark_summary,
        previous_result=args.previous_result,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
