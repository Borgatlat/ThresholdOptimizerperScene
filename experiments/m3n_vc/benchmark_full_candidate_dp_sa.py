"""Optimize thresholds on the full-candidate h24 DP layout with paper SA."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.brute_force_k1_free_layouts import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_ITERATIONS,
    DEFAULT_OUTCOMES,
    DEFAULT_SEED,
    DEFAULT_SPLIT_STRATEGY,
    _cascade_payload,
    _compact_optimization,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    _file_sha256,
    _write_json_atomic,
)
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_SA_RESTARTS,
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


DEFAULT_OUTPUT = Path(
    "checkpoints/joint_ga_with_k1_h24_dp_target_paper_sa/"
    "dp_layout_threshold_optimization.json"
)


def _constrained(metrics: dict[str, object], target_accuracy: float, method: str) -> dict[str, object]:
    result = dict(metrics)
    result.update(
        {
            "feasible": bool(float(result["accuracy"]) >= target_accuracy),
            "target_accuracy": float(target_accuracy),
            "method": method,
        }
    )
    return result


def run_benchmark(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output: Path = DEFAULT_OUTPUT,
    iterations: int = DEFAULT_ITERATIONS,
    restarts: int = DEFAULT_SA_RESTARTS,
    seed: int = DEFAULT_SEED,
    target_accuracy: float | None = None,
) -> dict[str, object]:
    if iterations < 1 or restarts < 1:
        raise ValueError("iterations and restarts must both be positive.")

    payload = load_empirical_outcomes(outcomes)
    validation_payload, holdout_payload, split = split_empirical_outcomes(
        payload,
        holdout_fraction=DEFAULT_HOLDOUT_FRACTION,
        split_strategy=DEFAULT_SPLIT_STRATEGY,
        random_seed=DEFAULT_SEED,
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
    cascade = validation_optimizer.synthesize()
    validation_evaluator = FixedLayoutThresholdEvaluator(validation_optimizer, cascade)
    holdout_evaluator = FixedLayoutThresholdEvaluator(holdout_optimizer, cascade)

    fixed_validation = validation_evaluator.evaluate(
        prune_reject_all_stages=True,
        strict_thresholds=True,
    )
    dp_fixed_validation_accuracy = float(fixed_validation["accuracy"])
    if target_accuracy is None:
        target_accuracy = dp_fixed_validation_accuracy
        target_accuracy_source = "full_candidate_dp_fixed_threshold_validation_accuracy"
    else:
        target_accuracy = float(target_accuracy)
        target_accuracy_source = "explicit_cli_or_api_override"
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1 inclusive.")
    fixed_validation = _constrained(
        fixed_validation,
        target_accuracy,
        "full_candidate_dp_fixed_thresholds",
    )
    fixed_holdout = _constrained(
        holdout_evaluator.evaluate(
            strict_thresholds=True,
            active_slots=fixed_validation["active_slots"],
        ),
        target_accuracy,
        "validation_pruned_policy_holdout_replay",
    )

    started = perf_counter()
    sa_validation = optimize_fixed_layout_thresholds_simulated_annealing(
        validation_evaluator,
        target_accuracy,
        n_iterations=iterations,
        random_seed=seed,
        show_progress=False,
        restarts=restarts,
    )
    completion_seconds = perf_counter() - started
    sa_holdout = _constrained(
        holdout_evaluator.evaluate(
            sa_validation["thresholds"],
            strict_thresholds=True,
            active_slots=sa_validation["active_slots"],
        ),
        target_accuracy,
        "validation_pruned_policy_holdout_replay",
    )

    linear_cascade = Cascade(
        expected_cost=0.0,
        initial=["K3", "K2", validation_optimizer.detector_id],
        specialized={},
        detector=validation_optimizer.detector_id,
    )
    linear_validation_evaluator = FixedLayoutThresholdEvaluator(
        validation_optimizer, linear_cascade
    )
    linear_holdout_evaluator = FixedLayoutThresholdEvaluator(
        holdout_optimizer, linear_cascade
    )
    linear_started = perf_counter()
    linear_validation = optimize_fixed_layout_thresholds_simulated_annealing(
        linear_validation_evaluator,
        target_accuracy,
        n_iterations=iterations,
        random_seed=seed,
        show_progress=False,
        restarts=restarts,
    )
    linear_completion_seconds = perf_counter() - linear_started
    linear_holdout = _constrained(
        linear_holdout_evaluator.evaluate(
            linear_validation["thresholds"],
            strict_thresholds=True,
            active_slots=linear_validation["active_slots"],
        ),
        target_accuracy,
        "validation_pruned_policy_holdout_replay",
    )

    summary: dict[str, object] = {
        "settings": {
            "dataset": "m3n_vc/h24",
            "outcomes": str(outcomes.resolve()),
            "outcomes_sha256": _file_sha256(outcomes),
            "removed_candidates": [],
            "detector_mode": "paper",
            "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
            "split_strategy": DEFAULT_SPLIT_STRATEGY,
            "split_seed": DEFAULT_SEED,
            "holdout_fraction": DEFAULT_HOLDOUT_FRACTION,
            "target_accuracy": target_accuracy,
            "target_accuracy_source": target_accuracy_source,
            "dp_fixed_validation_accuracy": dp_fixed_validation_accuracy,
            "threshold_optimizer": {
                "method": f"best_of_{restarts}_chellapilla_continuous_gaussian_sa",
                "iterations_per_restart": iterations,
                "restarts": restarts,
                "restart_seeds": [seed + index for index in range(restarts)],
                "continuous_thresholds": True,
                "prune_stages_accepting_zero_validation_samples": True,
                "freeze_validation_active_slots_on_holdout": True,
            },
        },
        "split": split,
        "layout": _cascade_payload(cascade),
        "target_accuracy": target_accuracy,
        "methods": {
            "dp_fixed_thresholds": {
                "validation": _compact_optimization(fixed_validation),
                "holdout": _compact_optimization(fixed_holdout),
            },
            "sa_on_dp_layout": {
                "completion_seconds": completion_seconds,
                "validation": _compact_optimization(sa_validation),
                "holdout": _compact_optimization(sa_holdout),
            },
            "sa_on_k3_k2_linear": {
                "completion_seconds": linear_completion_seconds,
                "layout": _cascade_payload(linear_cascade),
                "validation": _compact_optimization(linear_validation),
                "holdout": _compact_optimization(linear_holdout),
            },
        },
    }
    _write_json_atomic(output, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--restarts", type=int, default=DEFAULT_SA_RESTARTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-accuracy", type=float)
    args = parser.parse_args()
    summary = run_benchmark(
        outcomes=args.outcomes,
        output=args.output,
        iterations=args.iterations,
        restarts=args.restarts,
        seed=args.seed,
        target_accuracy=args.target_accuracy,
    )
    print(summary["methods"]["sa_on_dp_layout"])


if __name__ == "__main__":
    main()
