"""Run the h24 memetic layout search with both K0 and K1 available.

Unlike the historical K1-free experiment, the complete legal layout space
contains millions of topologies and is therefore represented dynamically
rather than materialized as a catalogue. Every visited topology receives the
canonical best-of-ten, 1,000-iteration continuous Chellapilla SA. The target
accuracy defaults to the fixed-threshold DP baseline with all candidates,
including K1, available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from cascade_profile import HierarchyProfile
from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.brute_force_k1_free_layouts import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_ITERATIONS,
    DEFAULT_OUTCOMES,
    DEFAULT_SEED,
    DEFAULT_SPLIT_STRATEGY,
    IndexedLayout,
    _cascade_payload,
    _compact_optimization,
    _direct_detector_metrics,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    InnerAnnealingFitness,
    _file_sha256,
    _load_jsonl,
    _settings_match,
    _write_json_atomic,
)
from hierarchy_optimizer import HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from joint_optimize_hierarchy_ga import (
    MemeticSearchConfig,
    run_memetic_search,
)
from layout_search import (
    LayoutSpace,
    TopologyGenome,
    cascade_from_genome,
    layout_id,
)
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    DEFAULT_SA_RESTARTS,
    FixedLayoutThresholdEvaluator,
    split_empirical_outcomes,
)


DEFAULT_TARGET_ACCURACY: float | None = None
DEFAULT_OUTPUT_DIR = Path("checkpoints/joint_ga_with_k1_h24_dp_target_paper_sa")
DEFAULT_POPULATION_SIZE = 32
DEFAULT_GENERATIONS = 24
DEFAULT_EVALUATION_BUDGET = 512
DEFAULT_ELITE_COUNT = 4
DEFAULT_TOURNAMENT_SIZE = 2
DEFAULT_CROSSOVER_RATE = 0.80
DEFAULT_MUTATION_RATE = 0.80
DEFAULT_RANDOM_IMMIGRANT_RATE = 0.20
DEFAULT_COMPONENT_RESAMPLE_RATE = 0.30

M3N_PROFILE = HierarchyProfile(
    dataset_id="m3n_vc/h24",
    global_classes=("gle350", "cx30", "mustang", "miata", "background"),
    groups={
        "coupe": ("mustang", "miata"),
        "suv": ("gle350", "cx30"),
    },
    router_outputs=("suv", "coupe", "background"),
    split_group_column="run_id",
)


def _ordered_subset_count(candidate_count: int) -> int:
    result = 1
    product = 1
    for selected in range(1, candidate_count + 1):
        product *= candidate_count - selected + 1
        result += product
    return result


def legal_layout_count(space: LayoutSpace) -> int:
    """Count the dynamic grammar without constructing its layouts."""

    from itertools import permutations

    total = 0
    initial_ids = space.initial_ids
    for length in range(len(initial_ids) + 1):
        for initial in permutations(initial_ids, length):
            branch_product = 1
            for router_id in initial:
                if router_id not in space.router_ids:
                    continue
                for group in space.profile.group_ids:
                    branch_product *= _ordered_subset_count(
                        len(space.allowed_branch_ids(initial, router_id, group))
                    )
            total += branch_product
    return total


def build_k1_layout_space(payload: Mapping[str, object]) -> LayoutSpace:
    candidates = payload.get("candidates")
    if candidates is None or not hasattr(candidates, "columns"):
        raise ValueError("Empirical outcomes have no candidate table.")
    space = LayoutSpace.from_candidates(M3N_PROFILE, candidates, "detector")
    if space.router_ids != ("K0", "K1"):
        raise ValueError(
            "The K1 experiment requires identifier candidates K0 and K1 in "
            f"that stable order; found {space.router_ids}."
        )
    return space


def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in (
        "experiments/m3n_vc/joint_optimize_hierarchy_ga_with_k1.py",
        "experiments/m3n_vc/joint_optimize_hierarchy_ga.py",
        "joint_optimize_hierarchy_ga.py",
        "layout_search.py",
        "hierarchy_optimizer.py",
        "threshold_optimizer.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


class CachedGenomeFitness:
    """Append-only, replayable fitness cache for the deterministic GA."""

    def __init__(
        self,
        optimizer: HierarchyOptimizer,
        space: LayoutSpace,
        *,
        target_accuracy: float,
        quantile_points: int,
        iterations: int,
        restarts: int,
        inner_seed: int,
        settings: Mapping[str, object],
        results_path: Path,
    ) -> None:
        self.space = space
        self.results_path = results_path
        self.settings = dict(settings)
        self.records = _load_jsonl(results_path)
        for record in self.records.values():
            if not _settings_match(record.get("settings"), self.settings):
                raise ValueError(
                    f"{results_path} belongs to another experiment; use "
                    "--overwrite or another output directory."
                )
        self.inner = InnerAnnealingFitness(
            optimizer,
            target_accuracy=target_accuracy,
            quantile_points=quantile_points,
            iterations=iterations,
            inner_seed=inner_seed,
            settings=self.settings,
            restarts=restarts,
        )
        self.cache_hits = 0
        self.new_evaluations = 0

    def __call__(self, genome: TopologyGenome) -> dict[str, object]:
        candidate_id = layout_id(genome, self.space)
        cached = self.records.get(candidate_id)
        if cached is not None:
            self.cache_hits += 1
            return dict(cached)

        indexed = IndexedLayout(
            len(self.records),
            candidate_id,
            cascade_from_genome(genome, self.space),
        )
        record = self.inner(indexed)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
        self.records[candidate_id] = record
        self.new_evaluations += 1
        if self.new_evaluations % 16 == 0:
            validation = record["validation"]
            assert isinstance(validation, Mapping)
            print(
                f"K1 GA fitness: {self.new_evaluations} new layouts; "
                f"latest={float(validation['expected_cost']):.3f} ms, "
                f"accuracy={float(validation['accuracy']):.6f}"
            )
        return dict(record)


def _holdout_metrics(
    genome: TopologyGenome,
    validation: Mapping[str, object],
    optimizer: HierarchyOptimizer,
    space: LayoutSpace,
    target_accuracy: float,
) -> dict[str, object]:
    cascade = cascade_from_genome(genome, space)
    if cascade.initial == [cascade.detector]:
        metrics = _direct_detector_metrics(optimizer, cascade, target_accuracy)
    else:
        thresholds = validation.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("The winning validation policy has no thresholds.")
        replay_options: dict[str, object] = {"strict_thresholds": True}
        if "active_slots" in validation:
            replay_options["active_slots"] = validation["active_slots"]
        metrics = FixedLayoutThresholdEvaluator(optimizer, cascade).evaluate(
            thresholds,
            **replay_options,
        )
    metrics = dict(metrics)
    metrics["feasible"] = bool(float(metrics["accuracy"]) >= target_accuracy)
    return _compact_optimization(metrics)


def run_k1_search(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_accuracy: float | None = DEFAULT_TARGET_ACCURACY,
    iterations: int = DEFAULT_ITERATIONS,
    restarts: int = DEFAULT_SA_RESTARTS,
    quantile_points: int = DEFAULT_QUANTILE_POINTS,
    inner_seed: int = DEFAULT_SEED,
    split_seed: int = DEFAULT_SEED,
    outer_seed: int = DEFAULT_SEED,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    evaluation_budget: int = DEFAULT_EVALUATION_BUDGET,
    overwrite: bool = False,
) -> dict[str, object]:
    payload = load_empirical_outcomes(outcomes)
    space = build_k1_layout_space(payload)
    layout_count = legal_layout_count(space)
    config = MemeticSearchConfig(
        population_size=population_size,
        generations=generations,
        evaluation_budget=evaluation_budget,
        elite_count=DEFAULT_ELITE_COUNT,
        tournament_size=DEFAULT_TOURNAMENT_SIZE,
        crossover_rate=DEFAULT_CROSSOVER_RATE,
        mutation_rate=DEFAULT_MUTATION_RATE,
        random_immigrant_rate=DEFAULT_RANDOM_IMMIGRANT_RATE,
        component_resample_rate=DEFAULT_COMPONENT_RESAMPLE_RATE,
        random_seed=outer_seed,
        allow_cached_reentry=False,
    )
    if iterations < 1 or restarts < 1:
        raise ValueError("iterations and restarts must both be positive.")
    if not population_size <= evaluation_budget <= layout_count:
        raise ValueError("evaluation_budget must be between population and space size.")

    validation_payload, holdout_payload, split = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=split_seed,
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
    dp_cascade = validation_optimizer.synthesize()
    dp_validation = dict(
        FixedLayoutThresholdEvaluator(validation_optimizer, dp_cascade).evaluate(
            prune_reject_all_stages=True,
            strict_thresholds=True,
        )
    )
    dp_fixed_validation_accuracy = float(dp_validation["accuracy"])
    if target_accuracy is None:
        target_accuracy = dp_fixed_validation_accuracy
        target_accuracy_source = "full_candidate_dp_fixed_threshold_validation_accuracy"
    else:
        target_accuracy = float(target_accuracy)
        target_accuracy_source = "explicit_cli_or_api_override"
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1 inclusive.")
    dp_validation.update(
        {
            "feasible": bool(dp_fixed_validation_accuracy >= target_accuracy),
            "target_accuracy": target_accuracy,
            "method": "full_candidate_dp_fixed_thresholds",
        }
    )
    dp_holdout = dict(
        FixedLayoutThresholdEvaluator(holdout_optimizer, dp_cascade).evaluate(
            strict_thresholds=True,
            active_slots=dp_validation["active_slots"],
        )
    )
    dp_holdout.update(
        {
            "feasible": bool(float(dp_holdout["accuracy"]) >= target_accuracy),
            "target_accuracy": target_accuracy,
            "method": "validation_pruned_policy_holdout_replay",
        }
    )

    settings: dict[str, object] = {
        "algorithm": "dynamic_constrained_memetic_genetic_algorithm",
        "layout_grammar": "depth_one_K0_K1",
        "layout_space_size": layout_count,
        "fitness_implementation_sha256": _implementation_sha256(),
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "removed_candidates": [],
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "target_accuracy": float(target_accuracy),
        "target_accuracy_source": target_accuracy_source,
        "dp_fixed_validation_accuracy": dp_fixed_validation_accuracy,
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
        "quantile_points_compatibility_argument": int(quantile_points),
        "inner_seed": int(inner_seed),
        "split_seed": int(split_seed),
        "outer_seed": int(outer_seed),
        "holdout_fraction": float(holdout_fraction),
        "split_strategy": split_strategy,
        **asdict(config),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "evaluations.jsonl"
    summary_path = output_dir / "summary.json"
    if overwrite:
        results_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if _settings_match(summary.get("settings"), settings):
            return summary
        raise ValueError("Existing summary belongs to another experiment.")

    fitness = CachedGenomeFitness(
        validation_optimizer,
        space,
        target_accuracy=target_accuracy,
        quantile_points=quantile_points,
        iterations=iterations,
        restarts=restarts,
        inner_seed=inner_seed,
        settings=settings,
        results_path=results_path,
    )
    print(
        f"K1-enabled memetic GA: {layout_count:,} legal layouts, "
        f"budget={evaluation_budget}, population={population_size}"
    )
    started = perf_counter()
    result = run_memetic_search(
        space,
        fitness,
        target_accuracy,
        config=config,
        seeds=(TopologyGenome(("K3",)),),
    )
    elapsed = perf_counter() - started
    winner = dict(result.best_record)
    winner_validation = winner.get("validation")
    if not isinstance(winner_validation, Mapping):
        raise ValueError("Winning record has no validation metrics.")
    winner["layout"] = _cascade_payload(
        cascade_from_genome(result.best_genome, space)
    )
    winner["holdout"] = _holdout_metrics(
        result.best_genome,
        winner_validation,
        holdout_optimizer,
        space,
        target_accuracy,
    )
    evaluated_ids = set(result.records)
    k1_layouts = sum(
        "K1" in result.genomes[candidate_id].initial
        for candidate_id in evaluated_ids
    )
    summary: dict[str, object] = {
        "settings": settings,
        "layout_space_size": layout_count,
        "fraction_of_layout_space": len(evaluated_ids) / layout_count,
        "unique_layouts_evaluated": len(evaluated_ids),
        "k1_layouts_evaluated": k1_layouts,
        "k1_fraction_evaluated": k1_layouts / len(evaluated_ids),
        "generations_completed": result.generations_completed,
        "cache_hits_this_invocation": fitness.cache_hits,
        "new_evaluations_this_invocation": fitness.new_evaluations,
        "elapsed_seconds_this_invocation": elapsed,
        "split": split,
        "dp_fixed_threshold_baseline": {
            "layout": _cascade_payload(dp_cascade),
            "validation": _compact_optimization(dp_validation),
            "holdout": _compact_optimization(dp_holdout),
        },
        "holdout_usage": "winner_only_after_validation_search",
        "winner": winner,
    }
    _write_json_atomic(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--restarts", type=int, default=DEFAULT_SA_RESTARTS)
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--inner-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--outer-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--evaluation-budget", type=int, default=DEFAULT_EVALUATION_BUDGET)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        payload = load_empirical_outcomes(args.outcomes)
        space = build_k1_layout_space(payload)
        print(
            json.dumps(
                {
                    "legal_layouts": legal_layout_count(space),
                    "evaluation_budget": args.evaluation_budget,
                    "fraction_evaluated": args.evaluation_budget
                    / legal_layout_count(space),
                    "iterations_per_layout": args.iterations,
                    "restarts_per_layout": args.restarts,
                    "total_iterations_per_layout": args.iterations * args.restarts,
                    "threshold_optimizer": "chellapilla_continuous_gaussian_sa",
                },
                indent=2,
            )
        )
        return
    summary = run_k1_search(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        target_accuracy=args.target_accuracy,
        iterations=args.iterations,
        restarts=args.restarts,
        quantile_points=args.quantile_points,
        inner_seed=args.inner_seed,
        split_seed=args.split_seed,
        outer_seed=args.outer_seed,
        population_size=args.population_size,
        generations=args.generations,
        evaluation_budget=args.evaluation_budget,
        overwrite=args.overwrite,
    )
    winner = summary["winner"]
    assert isinstance(winner, Mapping)
    print(json.dumps(winner, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
