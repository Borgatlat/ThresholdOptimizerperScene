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

from empirical_outcomes import DEFAULT_OUTPUT_PATH
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

        metrics = {
            "accuracy": float(np.mean(final_prediction == self.true_global)),
            "expected_cost": float(np.mean(final_cost)),
            "correct": int(np.sum(final_prediction == self.true_global)),
            "total": int(self.sample_count),
            "thresholds": threshold_map,
        }
        if include_route_counts:
            route_codes, route_counts = np.unique(ending, return_counts=True)
            metrics["route_counts"] = {
                self._ending_ids[int(code)]: int(count)
                for code, count in zip(route_codes, route_counts, strict=True)
            }
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
    detector_mode: str = "paper",
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
) -> FixedLayoutThresholdEvaluator:
    """Synthesize the current layout, then prepare it for threshold replay."""
    optimizer, cascade = optimize_empirical_hierarchy(
        path,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    return FixedLayoutThresholdEvaluator(optimizer, cascade)


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


def _print_result(result: Mapping[str, object]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True, default=float))


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
        default="paper",
        help="Use paper's always-correct fallback or the logged Kdet predictions.",
    )
    parser.add_argument("--detector-cost-ms", type=float, default=PAPER_DETECTOR_COST_MS)
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
    args = parser.parse_args()

    quantile_points = None if args.all_observed_thresholds else args.quantile_points
    evaluator = build_fixed_layout_evaluator(
        args.outcomes,
        detector_mode=args.detector_mode,
        detector_cost_ms=args.detector_cost_ms,
    )
    print(f"fixed initial layout: {evaluator.cascade.initial}")
    print(f"fixed specialized layout: {evaluator.cascade.specialized}")
    print(f"tunable models: {list(evaluator.tunable_ids)}")

    if args.method == "evaluate":
        _print_result(evaluator.evaluate())
        return

    if args.method == "exhaustive":
        _print_result(
            optimize_fixed_layout_thresholds_exhaustive(
                evaluator,
                args.target_accuracy,
                quantile_points=quantile_points,
                max_combinations=args.max_combinations,
            )
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
            )
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
        )
    )


if __name__ == "__main__":
    main()
