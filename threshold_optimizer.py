"""Threshold optimization for a fixed empirical hierarchy layout.

The hierarchy DP chooses a *topology* from the acceptance decisions recorded
at the registry thresholds.  This module keeps that topology fixed and
chooses the confidence threshold for every non-deterministic Ki that occurs
in it.  It never re-runs a neural network: each candidate policy is replayed
against the shared ``empirical_outcomes.pkl`` confidence/prediction table.

Two optimizers are provided:

* ``optimize_fixed_layout_thresholds_exhaustive`` enumerates a Cartesian
  product of threshold grids.  It is an exact baseline for that *discrete*
  grid, guarded against accidentally attempting millions/billions of policies.
* ``optimize_fixed_layout_thresholds_simulated_annealing`` explores the same
  grid stochastically and then runs coordinate descent as a local polish.

The two methods can therefore be compared fairly with
``benchmark_threshold_optimizers``.  Setting ``quantile_points=None`` uses
every distinct observed confidence, which is exact on the empirical data but
usually too large for an exhaustive Cartesian search.
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

from empirical_outcomes import DEFAULT_OUTPUT_PATH, load_empirical_outcomes
from hierarchy_optimizer import (
    PAPER_DETECTOR_COST_MS,
    Cascade,
    HierarchyOptimizer,
    optimize_empirical_hierarchy,
)
from utils.labels import GLOBAL_CLASS_NAMES


DEFAULT_TARGET_ACCURACY = 0.95
DEFAULT_QUANTILE_POINTS = 2
DEFAULT_MAX_EXHAUSTIVE_COMBINATIONS = 500_000
# A deployment policy must be selected against the fallback that will really
# run. The paper's perfect, 10-second surrogate remains available explicitly.
DEFAULT_DETECTOR_MODE = "trained"


class FixedLayoutThresholdEvaluator:
    """Vectorized replay of one fixed cascade layout.

    ``HierarchyOptimizer.evaluate_cascade`` evaluates a cascade using the
    ``accepted`` column saved at collection time.  Threshold tuning instead
    derives ``accepted = confidence >= threshold`` afresh for each candidate
    policy, while using the raw predictions recorded in that same payload.
    """

    def __init__(self, optimizer: HierarchyOptimizer, cascade: Cascade):
        self.optimizer = optimizer
        self.cascade = cascade
        self.candidates = optimizer.candidates
        self.detector_id = optimizer.detector_id
        self.sample_ids, self.true_global = self._load_true_labels()
        self.sample_count = len(self.sample_ids)

        self.tunable_ids = self._collect_tunable_ids()
        required_outcomes = set(self.tunable_ids)
        if optimizer.detector_mode == "trained":
            required_outcomes.add(optimizer.detector_outcome_id)

        self.prediction: dict[str, np.ndarray] = {}
        self.confidence: dict[str, np.ndarray] = {}
        for candidate_id in required_outcomes:
            prediction, confidence = self._load_candidate_arrays(candidate_id)
            self.prediction[candidate_id] = prediction
            self.confidence[candidate_id] = confidence

        self.default_thresholds = {
            candidate_id: self._default_threshold(candidate_id)
            for candidate_id in self.tunable_ids
        }
        self._intermediate_idx_to_group = dict(optimizer._intermediate_idx_to_group)
        self._specialized_groups = set(optimizer.groups)
        self._global_name_to_idx = {
            name: idx for idx, name in enumerate(GLOBAL_CLASS_NAMES)
        }
        ending_ids = [*self.tunable_ids, self.detector_id]
        if optimizer.detector_mode == "trained":
            ending_ids[-1] = optimizer.detector_outcome_id
        self._ending_ids = tuple(dict.fromkeys(ending_ids))
        self._ending_codes = {
            candidate_id: index for index, candidate_id in enumerate(self._ending_ids)
        }

    def _load_true_labels(self) -> tuple[np.ndarray, np.ndarray]:
        labels = self.optimizer.labels.sort_values("sample_id")
        sample_ids = labels["sample_id"].to_numpy(dtype=int)
        expected = np.arange(len(labels), dtype=int)
        if not np.array_equal(sample_ids, expected):
            raise ValueError(
                "The outcome labels must use contiguous sample_id values 0..N-1. "
                "Regenerate empirical outcomes before threshold optimization."
            )

        label_to_idx = {name: idx for idx, name in enumerate(GLOBAL_CLASS_NAMES)}
        true_global = labels["true_global_label"].map(label_to_idx)
        if true_global.isna().any():
            unknown = sorted(labels.loc[true_global.isna(), "true_global_label"].unique())
            raise ValueError(f"Unknown global label(s) in empirical outcomes: {unknown}")
        return sample_ids, true_global.to_numpy(dtype=int)

    def _collect_tunable_ids(self) -> tuple[str, ...]:
        ordered: list[str] = []
        seen: set[str] = set()

        for chain in [self.cascade.initial, *self.cascade.specialized.values()]:
            for candidate_id in chain:
                if candidate_id == self.detector_id or candidate_id in seen:
                    continue
                if candidate_id not in self.candidates.index:
                    raise ValueError(f"Cascade refers to unknown candidate {candidate_id!r}")
                if self.candidates.loc[candidate_id, "kind"] == "detector":
                    continue
                seen.add(candidate_id)
                ordered.append(candidate_id)

        if not ordered:
            raise ValueError("The fixed cascade contains no tunable classifier.")
        return tuple(ordered)

    def _load_candidate_arrays(self, candidate_id: str) -> tuple[np.ndarray, np.ndarray]:
        rows = self.optimizer.outcomes[
            self.optimizer.outcomes["candidate_id"] == candidate_id
        ].sort_values("sample_id")
        sample_ids = rows["sample_id"].to_numpy(dtype=int)
        if not np.array_equal(sample_ids, self.sample_ids):
            raise ValueError(
                f"Outcomes for {candidate_id} do not contain one row for every "
                "shared sample. Regenerate empirical outcomes before optimizing."
            )
        if "confidence" not in rows or "prediction" not in rows:
            raise ValueError(
                "Empirical outcomes are missing confidence or prediction columns. "
                "Regenerate them with the current empirical_outcomes.py collector."
            )
        confidence = rows["confidence"].to_numpy(dtype=float)
        if not np.isfinite(confidence).all():
            raise ValueError(f"Outcomes for {candidate_id} contain non-finite confidences.")
        return rows["prediction"].to_numpy(dtype=int), confidence

    def _default_threshold(self, candidate_id: str) -> float:
        threshold = self.candidates.loc[candidate_id, "threshold"]
        if threshold is None or not np.isfinite(float(threshold)):
            raise ValueError(
                f"Candidate {candidate_id} has no finite collection threshold."
            )
        return float(threshold)

    def _normalise_thresholds(self, thresholds: Mapping[str, float] | None) -> dict[str, float]:
        supplied = self.default_thresholds if thresholds is None else thresholds
        missing = set(self.tunable_ids) - set(supplied)
        extra = set(supplied) - set(self.tunable_ids)
        if missing or extra:
            raise ValueError(
                "Thresholds must match the fixed-layout models exactly; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        result = {candidate_id: float(supplied[candidate_id]) for candidate_id in self.tunable_ids}
        if not np.isfinite(list(result.values())).all():
            raise ValueError("Thresholds must be finite numbers.")
        return result

    def evaluate(
        self,
        thresholds: Mapping[str, float] | None = None,
        *,
        include_route_counts: bool = True,
        include_class_metrics: bool | None = None,
    ) -> dict:
        """Replay the fixed hierarchy for every logged sample.

        This is vectorized across samples.  Its runtime depends on cascade
        depth, not on the number of rows in the DataFrame, so evaluating a
        large threshold grid stays practical.
        """
        threshold_map = self._normalise_thresholds(thresholds)
        accepts = {
            candidate_id: self.confidence[candidate_id] >= threshold
            for candidate_id, threshold in threshold_map.items()
        }
        final_prediction = np.full(self.sample_count, -1, dtype=int)
        final_cost = np.zeros(self.sample_count, dtype=float)
        ending = np.full(self.sample_count, -1, dtype=np.int8)

        pending = np.ones(self.sample_count, dtype=bool)
        for candidate_id in self.cascade.initial:
            if not pending.any():
                break
            if candidate_id == self.detector_id:
                self._finish_detector(pending, final_prediction, final_cost, ending)
                pending[:] = False
                break

            final_cost[pending] += self._cost(candidate_id)
            accepted = pending & accepts[candidate_id]
            if self._is_identifier(candidate_id):
                self._route_identifier(
                    candidate_id,
                    accepted,
                    accepts,
                    final_prediction,
                    final_cost,
                    ending,
                )
            else:
                self._finish_candidate(
                    accepted,
                    candidate_id,
                    final_prediction,
                    ending,
                )
            pending &= ~accepted

        if pending.any():
            self._finish_detector(pending, final_prediction, final_cost, ending)

        if (final_prediction < 0).any() or (ending < 0).any():
            raise RuntimeError("Fixed-layout replay left one or more samples unresolved.")

        if include_class_metrics is None:
            include_class_metrics = include_route_counts

        correct_mask = final_prediction == self.true_global
        metrics = {
            "accuracy": float(np.mean(correct_mask)),
            "expected_cost": float(np.mean(final_cost)),
            "correct": int(np.sum(correct_mask)),
            "total": int(self.sample_count),
            "thresholds": threshold_map,
        }
        if include_route_counts:
            route_codes, route_counts = np.unique(ending, return_counts=True)
            metrics["route_counts"] = {
                self._ending_ids[int(code)]: int(count)
                for code, count in zip(route_codes, route_counts, strict=True)
            }
        if include_class_metrics:
            per_class: dict[str, dict[str, float | int | None]] = {}
            represented_accuracies: list[float] = []
            for class_index, class_name in enumerate(GLOBAL_CLASS_NAMES):
                class_mask = self.true_global == class_index
                support = int(np.sum(class_mask))
                correct = int(np.sum(correct_mask & class_mask))
                accuracy = correct / support if support else None
                per_class[class_name] = {
                    "accuracy": accuracy,
                    "correct": correct,
                    "total": support,
                }
                if accuracy is not None:
                    represented_accuracies.append(accuracy)

            metrics["per_class_accuracy"] = per_class
            metrics["macro_accuracy"] = (
                float(np.mean(represented_accuracies))
                if represented_accuracies
                else 0.0
            )
            metrics["worst_class_accuracy"] = (
                float(np.min(represented_accuracies))
                if represented_accuracies
                else 0.0
            )
        return metrics

    def _route_identifier(
        self,
        router_id: str,
        accepted: np.ndarray,
        accepts: Mapping[str, np.ndarray],
        final_prediction: np.ndarray,
        final_cost: np.ndarray,
        ending: np.ndarray,
    ) -> None:
        predictions = self.prediction[router_id]
        handled = np.zeros(self.sample_count, dtype=bool)

        for intermediate_idx, group in self._intermediate_idx_to_group.items():
            branch_mask = accepted & (predictions == intermediate_idx)
            if not branch_mask.any():
                continue
            handled |= branch_mask

            if group in self._specialized_groups:
                chain = self.cascade.specialized.get((router_id, group), [self.detector_id])
                self._run_specialized_chain(
                    chain,
                    branch_mask,
                    accepts,
                    final_prediction,
                    final_cost,
                    ending,
                )
            elif group in self._global_name_to_idx:
                final_prediction[branch_mask] = self._global_name_to_idx[group]
                ending[branch_mask] = self._ending_codes[router_id]
            else:
                self._finish_detector(branch_mask, final_prediction, final_cost, ending)

        unknown = accepted & ~handled
        if unknown.any():
            self._finish_detector(unknown, final_prediction, final_cost, ending)

    def _run_specialized_chain(
        self,
        chain: Sequence[str],
        initial_mask: np.ndarray,
        accepts: Mapping[str, np.ndarray],
        final_prediction: np.ndarray,
        final_cost: np.ndarray,
        ending: np.ndarray,
    ) -> None:
        pending = initial_mask.copy()
        for candidate_id in chain:
            if not pending.any():
                return
            if candidate_id == self.detector_id:
                self._finish_detector(pending, final_prediction, final_cost, ending)
                return

            final_cost[pending] += self._cost(candidate_id)
            accepted = pending & accepts[candidate_id]
            self._finish_candidate(accepted, candidate_id, final_prediction, ending)
            pending &= ~accepted

        if pending.any():
            self._finish_detector(pending, final_prediction, final_cost, ending)

    def _finish_candidate(
        self,
        mask: np.ndarray,
        candidate_id: str,
        final_prediction: np.ndarray,
        ending: np.ndarray,
    ) -> None:
        final_prediction[mask] = self.prediction[candidate_id][mask]
        ending[mask] = self._ending_codes[candidate_id]

    def _finish_detector(
        self,
        mask: np.ndarray,
        final_prediction: np.ndarray,
        final_cost: np.ndarray,
        ending: np.ndarray,
    ) -> None:
        final_cost[mask] += self.optimizer.detector_cost
        if self.optimizer.detector_mode == "paper":
            final_prediction[mask] = self.true_global[mask]
            ending[mask] = self._ending_codes[self.detector_id]
        else:
            candidate_id = self.optimizer.detector_outcome_id
            final_prediction[mask] = self.prediction[candidate_id][mask]
            ending[mask] = self._ending_codes[candidate_id]

    def _cost(self, candidate_id: str) -> float:
        return float(self.candidates.loc[candidate_id, "cost"])

    def _is_identifier(self, candidate_id: str) -> bool:
        return candidate_id in self.optimizer.identifier_ids

    @property
    def maximum_path_cost(self) -> float:
        """A conservative cost scale for the annealing accuracy penalty."""
        candidate_cost = sum(self._cost(candidate_id) for candidate_id in self.tunable_ids)
        return candidate_cost + self.optimizer.detector_cost


def build_fixed_layout_evaluator(
    path: str | Path = DEFAULT_OUTPUT_PATH,
    detector_mode: str = DEFAULT_DETECTOR_MODE,
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
) -> FixedLayoutThresholdEvaluator:
    """Synthesize the current layout, then prepare it for threshold replay."""
    optimizer, cascade = optimize_empirical_hierarchy(
        path,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    return FixedLayoutThresholdEvaluator(optimizer, cascade)


def _subset_empirical_payload(payload: dict, sample_ids: np.ndarray) -> dict:
    """Copy one subset of a shared outcome table and renumber sample ids.

    HierarchyOptimizer indexes its NumPy outcome arrays by ``sample_id``.  A
    subset must therefore be remapped from its original ids into 0..N-1
    before it can synthesize or replay a cascade independently.
    """
    original_ids = np.sort(np.asarray(sample_ids, dtype=int))
    labels = payload["labels"].set_index("sample_id", drop=False)
    subset_labels = labels.loc[original_ids].copy().reset_index(drop=True)
    subset_labels["sample_id"] = np.arange(len(subset_labels), dtype=int)

    id_map = {int(old_id): int(new_id) for new_id, old_id in enumerate(original_ids)}
    subset_outcomes = payload["outcomes"].loc[
        payload["outcomes"]["sample_id"].isin(original_ids)
    ].copy()
    subset_outcomes["sample_id"] = subset_outcomes["sample_id"].map(id_map).astype(int)

    return {
        "labels": subset_labels,
        "candidates": payload["candidates"].copy(),
        "detector": dict(payload["detector"]),
        "outcomes": subset_outcomes,
    }


def split_empirical_outcomes(
    payload: dict,
    holdout_fraction: float = 0.20,
    split_strategy: str = "blocked_per_run",
    random_seed: int = 0,
) -> tuple[dict, dict, dict]:
    """Split outcomes into optimization and holdout partitions.

    ``blocked_per_run`` is the default because consecutive M3N-VC segments
    are strongly correlated.  It assigns the final fraction of each run to
    holdout, preserving every class in both partitions for the current h24
    table, where each run has exactly one class.  ``random_per_run`` is
    provided only as a less conservative comparison point.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1.")
    if split_strategy not in {"blocked_per_run", "random_per_run"}:
        raise ValueError(
            "split_strategy must be 'blocked_per_run' or 'random_per_run'."
        )

    labels = payload["labels"].sort_values("sample_id")
    sample_ids = labels["sample_id"].to_numpy(dtype=int)
    if not np.array_equal(sample_ids, np.arange(len(labels), dtype=int)):
        raise ValueError(
            "Empirical labels must use contiguous sample_id values 0..N-1 "
            "before they can be split. Regenerate empirical outcomes."
        )

    rng = np.random.default_rng(random_seed)
    holdout_mask = np.zeros(len(labels), dtype=bool)
    per_run: dict[str, dict[str, int]] = {}
    for run_id, run_labels in labels.groupby("run_id", sort=False):
        run_sample_ids = run_labels["sample_id"].to_numpy(dtype=int)
        holdout_count = int(round(len(run_sample_ids) * holdout_fraction))
        holdout_count = min(max(holdout_count, 1), len(run_sample_ids) - 1)
        if holdout_count <= 0:
            raise ValueError(
                f"Run {run_id!r} has too few samples for a train/holdout split."
            )

        if split_strategy == "blocked_per_run":
            selected = run_sample_ids[-holdout_count:]
        else:
            selected = rng.choice(run_sample_ids, size=holdout_count, replace=False)
        holdout_mask[selected] = True
        per_run[str(run_id)] = {
            "optimization": int(len(run_sample_ids) - holdout_count),
            "holdout": int(holdout_count),
        }

    optimization_ids = sample_ids[~holdout_mask]
    holdout_ids = sample_ids[holdout_mask]
    return (
        _subset_empirical_payload(payload, optimization_ids),
        _subset_empirical_payload(payload, holdout_ids),
        {
            "strategy": split_strategy,
            "random_seed": int(random_seed),
            "holdout_fraction": float(holdout_fraction),
            "optimization_samples": int(len(optimization_ids)),
            "holdout_samples": int(len(holdout_ids)),
            "per_run": per_run,
        },
    )


def build_holdout_evaluators(
    path: str | Path = DEFAULT_OUTPUT_PATH,
    detector_mode: str = DEFAULT_DETECTOR_MODE,
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
    holdout_fraction: float = 0.20,
    split_strategy: str = "blocked_per_run",
    random_seed: int = 0,
) -> tuple[FixedLayoutThresholdEvaluator, FixedLayoutThresholdEvaluator, dict]:
    """Build train/test evaluators without letting holdout outcomes pick layout.

    The cascade topology is synthesized from the optimization split only, then
    replayed unchanged on the holdout split.  This is stricter than merely
    holding out threshold tuning, because it also prevents topology selection
    from using held-out accept/reject outcomes.
    """
    payload = load_empirical_outcomes(path)
    optimization_payload, holdout_payload, split = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=random_seed,
    )

    optimization_optimizer = HierarchyOptimizer(
        optimization_payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    optimization_cascade = optimization_optimizer.synthesize()
    holdout_optimizer = HierarchyOptimizer(
        holdout_payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )

    optimization_evaluator = FixedLayoutThresholdEvaluator(
        optimization_optimizer,
        optimization_cascade,
    )
    holdout_evaluator = FixedLayoutThresholdEvaluator(
        holdout_optimizer,
        optimization_cascade,
    )
    if optimization_evaluator.tunable_ids != holdout_evaluator.tunable_ids:
        raise RuntimeError("Optimization and holdout evaluators have incompatible layouts.")

    split["initial_layout"] = list(optimization_cascade.initial)
    split["specialized_layout"] = {
        f"{router_id}:{group}": list(chain)
        for (router_id, group), chain in optimization_cascade.specialized.items()
    }
    return optimization_evaluator, holdout_evaluator, split


def _holdout_summary(
    optimization_metrics: Mapping[str, object],
    holdout_evaluator: FixedLayoutThresholdEvaluator,
    target_accuracy: float,
) -> dict:
    holdout_metrics = holdout_evaluator.evaluate(optimization_metrics["thresholds"])
    return {
        "optimization": dict(optimization_metrics),
        "holdout": holdout_metrics,
        "accuracy_gap": float(optimization_metrics["accuracy"]) - float(holdout_metrics["accuracy"]),
        "cost_gap_ms": float(holdout_metrics["expected_cost"])
        - float(optimization_metrics["expected_cost"]),
        "holdout_feasible": bool(holdout_metrics["accuracy"] >= target_accuracy),
    }


def optimize_and_evaluate_holdout(
    path: str | Path = DEFAULT_OUTPUT_PATH,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    *,
    method: str = "anneal",
    detector_mode: str = DEFAULT_DETECTOR_MODE,
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
    holdout_fraction: float = 0.20,
    split_strategy: str = "blocked_per_run",
    quantile_points: int | None = DEFAULT_QUANTILE_POINTS,
    max_combinations: int = DEFAULT_MAX_EXHAUSTIVE_COMBINATIONS,
    annealing_iterations: int = 2_000,
    random_seed: int = 0,
) -> dict:
    """Optimize on one partition and report the frozen policy on another."""
    if method not in {"evaluate", "exhaustive", "anneal", "benchmark"}:
        raise ValueError("method must be evaluate, exhaustive, anneal, or benchmark.")

    optimization_evaluator, holdout_evaluator, split = build_holdout_evaluators(
        path,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=random_seed,
    )
    baseline = _holdout_summary(
        optimization_evaluator.evaluate(),
        holdout_evaluator,
        target_accuracy,
    )
    result: dict = {
        "target_accuracy": float(target_accuracy),
        "detector_mode": detector_mode,
        "detector": {
            "id": optimization_evaluator.optimizer.detector_outcome_id,
            "cost_ms": float(optimization_evaluator.optimizer.detector_cost),
        },
        "split": split,
        "baseline": baseline,
    }
    if method == "evaluate":
        return result

    if method == "exhaustive":
        optimized = optimize_fixed_layout_thresholds_exhaustive(
            optimization_evaluator,
            target_accuracy,
            quantile_points=quantile_points,
            max_combinations=max_combinations,
        )
        result["exhaustive"] = _holdout_summary(
            optimized, holdout_evaluator, target_accuracy
        )
        return result

    if method == "anneal":
        optimized = optimize_fixed_layout_thresholds_simulated_annealing(
            optimization_evaluator,
            target_accuracy,
            quantile_points=quantile_points,
            n_iterations=annealing_iterations,
            random_seed=random_seed,
        )
        result["annealing"] = _holdout_summary(
            optimized, holdout_evaluator, target_accuracy
        )
        return result

    comparison = benchmark_threshold_optimizers(
        optimization_evaluator,
        target_accuracy,
        quantile_points=quantile_points,
        max_combinations=max_combinations,
        annealing_iterations=annealing_iterations,
        random_seed=random_seed,
    )
    result["grid_sizes"] = comparison["grid_sizes"]
    result["comparison"] = {
        "annealing_cost_gap_ms": comparison["annealing_cost_gap_ms"],
        "annealing_runtime_speedup": comparison["annealing_runtime_speedup"],
    }
    result["exhaustive"] = _holdout_summary(
        comparison["exhaustive"], holdout_evaluator, target_accuracy
    )
    result["annealing"] = _holdout_summary(
        comparison["annealing"], holdout_evaluator, target_accuracy
    )
    return result


def build_threshold_grids(
    evaluator: FixedLayoutThresholdEvaluator,
    quantile_points: int | None = DEFAULT_QUANTILE_POINTS,
) -> dict[str, np.ndarray]:
    """Return one empirically meaningful threshold grid per active Ki.

    An empirical policy changes only when a threshold crosses an observed
    confidence.  ``quantile_points=None`` exposes all such breakpoints.  A
    finite value samples that space by quantile, retains the current threshold,
    and always includes both "accept everything" and "reject everything".
    """
    if quantile_points is not None and quantile_points < 2:
        raise ValueError("quantile_points must be at least 2, or None for exact breakpoints.")

    grids: dict[str, np.ndarray] = {}
    for candidate_id in evaluator.tunable_ids:
        confidence = evaluator.confidence[candidate_id]
        if quantile_points is None:
            values = np.unique(confidence)
        else:
            quantiles = np.linspace(0.0, 1.0, quantile_points)
            values = np.quantile(confidence, quantiles)

        values = np.concatenate(
            (
                np.array([0.0, evaluator.default_thresholds[candidate_id]]),
                np.asarray(values, dtype=float),
                np.array([np.nextafter(float(np.max(confidence)), np.inf)]),
            )
        )
        grids[candidate_id] = np.unique(values)
    return grids


def _validate_grids(
    evaluator: FixedLayoutThresholdEvaluator,
    grids: Mapping[str, Sequence[float]] | None,
    quantile_points: int | None,
) -> dict[str, np.ndarray]:
    resolved = build_threshold_grids(evaluator, quantile_points) if grids is None else grids
    if set(resolved) != set(evaluator.tunable_ids):
        raise ValueError(
            "Threshold grids must contain exactly the fixed-layout classifier ids; "
            f"expected={list(evaluator.tunable_ids)}, got={sorted(resolved)}"
        )

    validated: dict[str, np.ndarray] = {}
    for candidate_id in evaluator.tunable_ids:
        values = np.unique(np.asarray(resolved[candidate_id], dtype=float))
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"Threshold grid for {candidate_id} must contain finite values.")
        validated[candidate_id] = values
    return validated


def _policy_key(metrics: Mapping[str, float], target_accuracy: float) -> tuple[float, ...]:
    """Constrained ordering: feasible policies first, then lower runtime."""
    accuracy = float(metrics["accuracy"])
    cost = float(metrics["expected_cost"])
    if accuracy >= target_accuracy:
        return (0.0, cost, -accuracy)
    return (1.0, target_accuracy - accuracy, cost)


def _result(
    metrics: dict,
    target_accuracy: float,
    evaluations: int,
    elapsed_seconds: float,
    **extra: object,
) -> dict:
    result = dict(metrics)
    result.update(
        {
            "feasible": bool(metrics["accuracy"] >= target_accuracy),
            "target_accuracy": float(target_accuracy),
            "evaluations": int(evaluations),
            "elapsed_seconds": float(elapsed_seconds),
        }
    )
    result.update(extra)
    return result


def optimize_fixed_layout_thresholds_exhaustive(
    evaluator: FixedLayoutThresholdEvaluator,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    *,
    grids: Mapping[str, Sequence[float]] | None = None,
    quantile_points: int | None = DEFAULT_QUANTILE_POINTS,
    max_combinations: int = DEFAULT_MAX_EXHAUSTIVE_COMBINATIONS,
) -> dict:
    """Find the exact best policy in a discrete Cartesian threshold grid."""
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1.")
    threshold_grids = _validate_grids(evaluator, grids, quantile_points)
    grid_sizes = {candidate_id: len(values) for candidate_id, values in threshold_grids.items()}
    combinations = math.prod(grid_sizes.values())
    if combinations > max_combinations:
        raise ValueError(
            f"Exhaustive search would evaluate {combinations:,} policies over "
            f"{grid_sizes}. The current limit is {max_combinations:,}. "
            "Use fewer quantile points, pass a larger max_combinations value, "
            "or use simulated annealing."
        )

    candidate_ids = evaluator.tunable_ids
    best_metrics: dict | None = None
    evaluations = 0
    started = perf_counter()
    for values in product(*(threshold_grids[candidate_id] for candidate_id in candidate_ids)):
        policy = dict(zip(candidate_ids, values, strict=True))
        metrics = evaluator.evaluate(policy, include_route_counts=False)
        evaluations += 1
        if best_metrics is None or _policy_key(metrics, target_accuracy) < _policy_key(best_metrics, target_accuracy):
            best_metrics = metrics

    assert best_metrics is not None
    best_metrics = evaluator.evaluate(best_metrics["thresholds"])
    return _result(
        best_metrics,
        target_accuracy,
        evaluations,
        perf_counter() - started,
        method="exhaustive",
        combinations=int(combinations),
        grid_sizes=grid_sizes,
    )


def coordinate_descent_thresholds(
    evaluator: FixedLayoutThresholdEvaluator,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    *,
    grids: Mapping[str, Sequence[float]] | None = None,
    quantile_points: int | None = DEFAULT_QUANTILE_POINTS,
    initial_thresholds: Mapping[str, float] | None = None,
    max_passes: int = 25,
) -> dict:
    """Optimize one threshold at a time until no grid coordinate improves.

    It is intentionally a local optimizer.  Simulated annealing uses it only
    after its global exploration phase; exposing it separately makes the
    distinction and the local-minimum trade-off explicit.
    """
    if max_passes < 1:
        raise ValueError("max_passes must be at least 1.")
    threshold_grids = _validate_grids(evaluator, grids, quantile_points)
    current = evaluator._normalise_thresholds(initial_thresholds)
    # Snap a caller-provided continuous initial policy to the discrete grid.
    current = {
        candidate_id: float(
            threshold_grids[candidate_id][
                np.argmin(np.abs(threshold_grids[candidate_id] - value))
            ]
        )
        for candidate_id, value in current.items()
    }
    started = perf_counter()
    current_metrics = evaluator.evaluate(current, include_route_counts=False)
    evaluations = 1
    passes = 0

    for _ in range(max_passes):
        passes += 1
        changed = False
        for candidate_id in evaluator.tunable_ids:
            coordinate_best = current_metrics
            coordinate_value = current[candidate_id]
            for value in threshold_grids[candidate_id]:
                if value == current[candidate_id]:
                    continue
                proposal = dict(current)
                proposal[candidate_id] = float(value)
                metrics = evaluator.evaluate(proposal, include_route_counts=False)
                evaluations += 1
                if _policy_key(metrics, target_accuracy) < _policy_key(coordinate_best, target_accuracy):
                    coordinate_best = metrics
                    coordinate_value = float(value)
            if coordinate_value != current[candidate_id]:
                current[candidate_id] = coordinate_value
                current_metrics = coordinate_best
                changed = True
        if not changed:
            break

    current_metrics = evaluator.evaluate(current_metrics["thresholds"])
    return _result(
        current_metrics,
        target_accuracy,
        evaluations,
        perf_counter() - started,
        method="coordinate_descent",
        passes=passes,
    )


def optimize_fixed_layout_thresholds_simulated_annealing(
    evaluator: FixedLayoutThresholdEvaluator,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    *,
    grids: Mapping[str, Sequence[float]] | None = None,
    quantile_points: int | None = DEFAULT_QUANTILE_POINTS,
    n_iterations: int = 2_000,
    random_seed: int = 0,
    coordinate_descent_passes: int = 25,
    accuracy_penalty: float | None = None,
) -> dict:
    """Anneal on a discrete threshold grid, then polish with coordinate descent.

    The energy is a Lagrangian-style runtime plus an accuracy-shortfall
    penalty.  The returned winner is still selected by the hard constraint:
    a feasible policy always beats an infeasible one regardless of energy.
    """
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1.")
    threshold_grids = _validate_grids(evaluator, grids, quantile_points)
    candidate_ids = evaluator.tunable_ids
    grid_indices = {
        candidate_id: int(
            np.argmin(
                np.abs(
                    threshold_grids[candidate_id]
                    - evaluator.default_thresholds[candidate_id]
                )
            )
        )
        for candidate_id in candidate_ids
    }
    rng = np.random.default_rng(random_seed)
    if accuracy_penalty is None:
        # A 0.1% shortfall costs at least ten complete worst-case paths.
        accuracy_penalty = max(1.0, evaluator.maximum_path_cost * 10_000.0)

    def policy_from_indices(indices: Mapping[str, int]) -> dict[str, float]:
        return {
            candidate_id: float(threshold_grids[candidate_id][indices[candidate_id]])
            for candidate_id in candidate_ids
        }

    def energy(metrics: Mapping[str, float]) -> float:
        shortfall = max(0.0, target_accuracy - float(metrics["accuracy"]))
        return float(metrics["expected_cost"]) + accuracy_penalty * shortfall

    started = perf_counter()
    current_indices = dict(grid_indices)
    current_metrics = evaluator.evaluate(
        policy_from_indices(current_indices), include_route_counts=False
    )
    current_energy = energy(current_metrics)
    best_metrics = current_metrics
    best_indices = dict(current_indices)
    evaluations = 1
    accepted_moves = 0
    initial_temperature = max(1.0, evaluator.maximum_path_cost * 0.25)
    final_temperature = max(1e-6, initial_temperature * 1e-3)

    for iteration in range(n_iterations):
        progress = iteration / max(n_iterations - 1, 1)
        temperature = initial_temperature * (final_temperature / initial_temperature) ** progress
        candidate_id = str(rng.choice(candidate_ids))
        grid = threshold_grids[candidate_id]
        current_index = current_indices[candidate_id]
        proposal_indices = dict(current_indices)

        if len(grid) > 1 and rng.random() < 0.8:
            max_step = max(1, int(round((1.0 - progress) * (len(grid) - 1))))
            step = int(rng.integers(-max_step, max_step + 1))
            if step == 0:
                step = 1 if current_index < len(grid) - 1 else -1
            proposal_indices[candidate_id] = int(np.clip(current_index + step, 0, len(grid) - 1))
        else:
            proposal_indices[candidate_id] = int(rng.integers(0, len(grid)))

        if proposal_indices[candidate_id] == current_index:
            continue

        proposal_metrics = evaluator.evaluate(
            policy_from_indices(proposal_indices), include_route_counts=False
        )
        evaluations += 1
        proposal_energy = energy(proposal_metrics)
        energy_delta = proposal_energy - current_energy
        if energy_delta <= 0.0 or rng.random() < math.exp(-energy_delta / temperature):
            current_indices = proposal_indices
            current_metrics = proposal_metrics
            current_energy = proposal_energy
            accepted_moves += 1

        if _policy_key(proposal_metrics, target_accuracy) < _policy_key(best_metrics, target_accuracy):
            best_metrics = proposal_metrics
            best_indices = proposal_indices

    annealing_elapsed = perf_counter() - started
    polished = coordinate_descent_thresholds(
        evaluator,
        target_accuracy,
        grids=threshold_grids,
        initial_thresholds=policy_from_indices(best_indices),
        max_passes=coordinate_descent_passes,
    )
    best_before_polish = _result(
        evaluator.evaluate(best_metrics["thresholds"]),
        target_accuracy,
        evaluations,
        annealing_elapsed,
        method="simulated_annealing",
    )
    if _policy_key(polished, target_accuracy) < _policy_key(best_before_polish, target_accuracy):
        winner = dict(polished)
    else:
        winner = best_before_polish

    winner.update(
        {
            "method": "simulated_annealing_plus_coordinate_descent",
            "annealing_iterations": int(n_iterations),
            "annealing_evaluations": int(evaluations),
            "annealing_elapsed_seconds": float(annealing_elapsed),
            "annealing_accepted_moves": int(accepted_moves),
            "coordinate_descent_evaluations": int(polished["evaluations"]),
            "coordinate_descent_elapsed_seconds": float(polished["elapsed_seconds"]),
            "coordinate_descent_passes": int(polished["passes"]),
            "evaluations": int(evaluations + polished["evaluations"]),
            "elapsed_seconds": float(annealing_elapsed + polished["elapsed_seconds"]),
        }
    )
    return winner


def benchmark_threshold_optimizers(
    evaluator: FixedLayoutThresholdEvaluator,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    *,
    quantile_points: int | None = DEFAULT_QUANTILE_POINTS,
    max_combinations: int = DEFAULT_MAX_EXHAUSTIVE_COMBINATIONS,
    annealing_iterations: int = 2_000,
    random_seed: int = 0,
) -> dict:
    """Run exact grid search and annealing/polish on exactly the same grid."""
    grids = build_threshold_grids(evaluator, quantile_points)
    exhaustive = optimize_fixed_layout_thresholds_exhaustive(
        evaluator,
        target_accuracy,
        grids=grids,
        max_combinations=max_combinations,
    )
    annealing = optimize_fixed_layout_thresholds_simulated_annealing(
        evaluator,
        target_accuracy,
        grids=grids,
        n_iterations=annealing_iterations,
        random_seed=random_seed,
    )
    exhaustive_cost = float(exhaustive["expected_cost"])
    annealing_cost = float(annealing["expected_cost"])
    return {
        "target_accuracy": float(target_accuracy),
        "tunable_ids": list(evaluator.tunable_ids),
        "grid_sizes": {candidate_id: len(values) for candidate_id, values in grids.items()},
        "exhaustive": exhaustive,
        "annealing": annealing,
        "annealing_cost_gap_ms": annealing_cost - exhaustive_cost,
        "annealing_runtime_speedup": (
            float(exhaustive["elapsed_seconds"]) / float(annealing["elapsed_seconds"])
            if float(annealing["elapsed_seconds"]) > 0.0
            else float("inf")
        ),
    }


def _print_result(result: Mapping[str, object], output_path: Path | None = None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True, default=float)
    print(text)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
        print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize thresholds for the current empirical hierarchy layout."
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to an empirical_outcomes*.pkl payload.",
    )
    parser.add_argument(
        "--method",
        choices=("benchmark", "exhaustive", "anneal", "evaluate"),
        default="benchmark",
    )
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY)
    parser.add_argument(
        "--detector-mode",
        choices=("paper", "trained"),
        default=DEFAULT_DETECTOR_MODE,
        help=(
            "Use the logged Kdet predictions/cost (default) or the paper's "
            "always-correct fallback surrogate."
        ),
    )
    parser.add_argument(
        "--detector-cost-ms",
        type=float,
        default=PAPER_DETECTOR_COST_MS,
        help="Synthetic fallback cost; used only with --detector-mode paper.",
    )
    parser.add_argument(
        "--quantile-points",
        type=int,
        default=DEFAULT_QUANTILE_POINTS,
        help="Observed-confidence quantiles per Ki; add 0, current threshold, and reject-all.",
    )
    parser.add_argument(
        "--all-observed-thresholds",
        action="store_true",
        help="Use every distinct observed confidence; generally only practical with --method anneal.",
    )
    parser.add_argument("--max-combinations", type=int, default=DEFAULT_MAX_EXHAUSTIVE_COMBINATIONS)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON optimization report.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=None,
        help=(
            "Optimize on the earlier portion of every run and report the frozen "
            "policy on this held-out fraction."
        ),
    )
    parser.add_argument(
        "--split-strategy",
        choices=("blocked_per_run", "random_per_run"),
        default="blocked_per_run",
        help=(
            "Use contiguous held-out blocks by default; random_per_run is a less "
            "conservative comparison split."
        ),
    )
    args = parser.parse_args()

    quantile_points = None if args.all_observed_thresholds else args.quantile_points
    if args.holdout_fraction is not None:
        _print_result(
            optimize_and_evaluate_holdout(
                args.outcomes,
                args.target_accuracy,
                method=args.method,
                detector_mode=args.detector_mode,
                detector_cost_ms=args.detector_cost_ms,
                holdout_fraction=args.holdout_fraction,
                split_strategy=args.split_strategy,
                quantile_points=quantile_points,
                max_combinations=args.max_combinations,
                annealing_iterations=args.iterations,
                random_seed=args.seed,
            ),
            args.output,
        )
        return

    evaluator = build_fixed_layout_evaluator(
        args.outcomes,
        detector_mode=args.detector_mode,
        detector_cost_ms=args.detector_cost_ms,
    )
    print(f"fixed initial layout: {evaluator.cascade.initial}")
    print(f"fixed specialized layout: {evaluator.cascade.specialized}")
    print(f"tunable models: {list(evaluator.tunable_ids)}")

    if args.method == "evaluate":
        _print_result(evaluator.evaluate(), args.output)
        return

    if args.method == "exhaustive":
        _print_result(
            optimize_fixed_layout_thresholds_exhaustive(
                evaluator,
                args.target_accuracy,
                quantile_points=quantile_points,
                max_combinations=args.max_combinations,
            ),
            args.output,
        )
        return

    if args.method == "anneal":
        _print_result(
            optimize_fixed_layout_thresholds_simulated_annealing(
                evaluator,
                args.target_accuracy,
                quantile_points=quantile_points,
                n_iterations=args.iterations,
                random_seed=args.seed,
            ),
            args.output,
        )
        return

    _print_result(
        benchmark_threshold_optimizers(
            evaluator,
            args.target_accuracy,
            quantile_points=quantile_points,
            max_combinations=args.max_combinations,
            annealing_iterations=args.iterations,
            random_seed=args.seed,
        ),
        args.output,
    )


if __name__ == "__main__":
    main()
