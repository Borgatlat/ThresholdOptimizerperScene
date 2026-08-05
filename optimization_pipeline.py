"""End-to-end dataset-neutral empirical and joint-optimization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cascade_profile import HierarchyProfile, profile_from_payload
from empirical_outcomes import (
    EvaluationSplit,
    PretrainedClassifier,
    ensure_empirical_outcomes,
    load_empirical_outcomes,
)
from hierarchy_optimizer import HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from joint_optimize_hierarchy_ga import (
    MemeticSearchConfig,
    MemeticSearchResult,
    run_memetic_search,
)
from layout_search import (
    LayoutSpace,
    TopologyGenome,
    cascade_from_genome,
)
from result_packets import create_result_packet, write_result_packet
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


@dataclass(frozen=True)
class PreparedOptimization:
    profile: HierarchyProfile
    validation_optimizer: HierarchyOptimizer
    test_optimizer: HierarchyOptimizer
    layout_space: LayoutSpace
    split: dict[str, object]


def prepare_optimization(
    outcomes_path: str | Path,
    *,
    detector_mode: str = "paper",
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
    holdout_fraction: float = 0.20,
    split_strategy: str = "blocked_per_run",
    split_seed: int = 0,
) -> PreparedOptimization:
    payload = load_empirical_outcomes(outcomes_path)
    profile = profile_from_payload(payload)
    validation, test, split = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=split_seed,
    )
    validation_optimizer = HierarchyOptimizer(
        validation,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    test_optimizer = HierarchyOptimizer(
        test,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    space = LayoutSpace.from_candidates(
        profile,
        validation["candidates"],
        validation_optimizer.detector_id,
    )
    return PreparedOptimization(
        profile,
        validation_optimizer,
        test_optimizer,
        space,
        split,
    )


class AnnealedLayoutFitness:
    """Optimize thresholds independently for every proposed topology."""

    def __init__(
        self,
        prepared: PreparedOptimization,
        *,
        target_accuracy: float,
        iterations: int = 8_000,
        quantile_points: int = DEFAULT_QUANTILE_POINTS,
        inner_seed: int = 0,
    ) -> None:
        self.prepared = prepared
        self.target_accuracy = float(target_accuracy)
        self.iterations = int(iterations)
        self.quantile_points = int(quantile_points)
        self.inner_seed = int(inner_seed)

    def __call__(self, genome: TopologyGenome) -> dict[str, object]:
        cascade = cascade_from_genome(genome, self.prepared.layout_space)
        if not genome.initial:
            validation = self.prepared.validation_optimizer.evaluate_cascade(cascade)
            validation["thresholds"] = {}
            test = self.prepared.test_optimizer.evaluate_cascade(cascade)
            test["thresholds"] = {}
        else:
            validation_evaluator = FixedLayoutThresholdEvaluator(
                self.prepared.validation_optimizer, cascade
            )
            validation = optimize_fixed_layout_thresholds_simulated_annealing(
                validation_evaluator,
                self.target_accuracy,
                quantile_points=self.quantile_points,
                n_iterations=self.iterations,
                random_seed=self.inner_seed,
                show_progress=False,
            )
            test_evaluator = FixedLayoutThresholdEvaluator(
                self.prepared.test_optimizer, cascade
            )
            test = test_evaluator.evaluate(validation["thresholds"])
        validation = dict(validation)
        test = dict(test)
        validation["feasible"] = bool(
            float(validation["accuracy"]) >= self.target_accuracy
        )
        test["feasible"] = bool(float(test["accuracy"]) >= self.target_accuracy)
        return {
            "layout": {
                "initial": list(genome.initial),
                "branches": {
                    f"{router_id}:{group}": list(chain)
                    for router_id, group, chain in genome.branches
                },
                "detector": self.prepared.layout_space.detector_id,
            },
            "validation": validation,
            "test": test,
        }


def optimize_joint_from_outcomes(
    outcomes_path: str | Path,
    *,
    target_accuracy: float,
    result_path: str | Path,
    method_id: str = "memetic_ga",
    method_label: str = "Memetic GA",
    iterations: int = 8_000,
    quantile_points: int = DEFAULT_QUANTILE_POINTS,
    inner_seed: int = 0,
    search_config: MemeticSearchConfig = MemeticSearchConfig(),
    seeds: Sequence[TopologyGenome] = (),
    detector_mode: str = "paper",
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
) -> tuple[MemeticSearchResult, dict[str, object]]:
    prepared = prepare_optimization(
        outcomes_path,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    fitness = AnnealedLayoutFitness(
        prepared,
        target_accuracy=target_accuracy,
        iterations=iterations,
        quantile_points=quantile_points,
        inner_seed=inner_seed,
    )
    search = run_memetic_search(
        prepared.layout_space,
        fitness,
        target_accuracy,
        config=search_config,
        seeds=seeds,
    )
    best = search.best_record
    packet = create_result_packet(
        profile=prepared.profile,
        method_id=method_id,
        method_label=method_label,
        target_accuracy=target_accuracy,
        layout=best["layout"],
        validation=best["validation"],
        test=best["test"],
        provenance={
            "outcomes": str(Path(outcomes_path)),
            "iterations": iterations,
            "quantile_points": quantile_points,
            "inner_seed": inner_seed,
            "outer_seed": search_config.random_seed,
            "unique_layouts_evaluated": len(search.records),
            "split": prepared.split,
        },
    )
    write_result_packet(packet, result_path)
    return search, packet


def ensure_and_optimize_joint(
    *,
    outcomes_path: str | Path,
    profile: HierarchyProfile,
    splits: Sequence[EvaluationSplit],
    classifiers: Sequence[PretrainedClassifier],
    target_accuracy: float,
    result_path: str | Path,
    force_outcomes: bool = False,
    **optimizer_options,
) -> tuple[MemeticSearchResult, dict[str, object]]:
    ensure_empirical_outcomes(
        outcomes_path,
        profile,
        splits,
        classifiers,
        force=force_outcomes,
    )
    return optimize_joint_from_outcomes(
        outcomes_path,
        target_accuracy=target_accuracy,
        result_path=result_path,
        **optimizer_options,
    )
