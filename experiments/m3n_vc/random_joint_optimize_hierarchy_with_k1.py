"""Uniform random search over the implicit h24 K0/K1 layout grammar.

The complete grammar contains 11,589,085 layouts, so this module samples
canonical integer indices without materializing that catalogue.  Every index
maps bijectively to one legal topology.  Repeated random indices are rejected,
which makes each accepted prefix an exact uniform sample without replacement.
Each accepted layout receives the same canonical continuous threshold SA used
by the K1-enabled memetic search.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import islice, permutations
from pathlib import Path
from time import perf_counter
from typing import Iterator, Mapping, Sequence

import numpy as np

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
    _layout_selection_key,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga import (
    InnerAnnealingFitness,
    _file_sha256,
    _load_jsonl,
    _settings_match,
    _write_json_atomic,
)
from experiments.m3n_vc.joint_optimize_hierarchy_ga_with_k1 import (
    build_k1_layout_space,
    legal_layout_count,
)
from hierarchy_optimizer import HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from layout_search import (
    LayoutSpace,
    TopologyGenome,
    cascade_from_genome,
    layout_id,
    repair_genome,
)
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    DEFAULT_SA_RESTARTS,
    FixedLayoutThresholdEvaluator,
    split_empirical_outcomes,
)


DEFAULT_OUTPUT_DIR = Path("checkpoints/random_joint_with_k1_h24_target_090")
DEFAULT_TARGET_ACCURACY = 0.90
DEFAULT_TIME_BUDGET_SECONDS = 1_500.0

SETTINGS_SCHEMA_VERSION = "random-joint-layout-search-with-k1/v1"
SAMPLER_SCHEMA_VERSION = "implicit-depth-one-layout-sampler/v1"
ALGORITHM = "uniform_random_implicit_layout_sampling_without_replacement"


@dataclass(frozen=True)
class _BranchSpec:
    router_id: str
    group: str
    allowed: tuple[str, ...]
    option_count: int


@dataclass(frozen=True)
class _InitialBlock:
    start: int
    stop: int
    initial: tuple[str, ...]
    branches: tuple[_BranchSpec, ...]


@dataclass(frozen=True)
class SampledLayout:
    """One deterministic position in a random layout prefix."""

    sample_rank: int
    space_index: int
    layout_id: str
    genome: TopologyGenome


def _ordered_subset_count(candidate_count: int) -> int:
    return sum(math.perm(candidate_count, length) for length in range(candidate_count + 1))


def _ordered_subset_at(values: Sequence[str], index: int) -> tuple[str, ...]:
    """Unrank the ordered subsets emitted by length then permutations."""

    size = len(values)
    if index < 0 or index >= _ordered_subset_count(size):
        raise IndexError("Ordered-subset index is outside the legal range.")

    remaining_index = int(index)
    selected_length = 0
    for length in range(size + 1):
        block_size = math.perm(size, length)
        if remaining_index < block_size:
            selected_length = length
            break
        remaining_index -= block_size

    available = list(values)
    result: list[str] = []
    for position in range(selected_length):
        suffix_length = selected_length - position - 1
        suffix_count = math.perm(len(available) - 1, suffix_length)
        choice, remaining_index = divmod(remaining_index, suffix_count)
        result.append(str(available.pop(choice)))
    return tuple(result)


class ImplicitLayoutCatalogue:
    """Stable integer indexing for a dynamic depth-one layout grammar."""

    def __init__(self, space: LayoutSpace) -> None:
        self.space = space
        blocks: list[_InitialBlock] = []
        offset = 0
        for length in range(len(space.initial_ids) + 1):
            for initial in permutations(space.initial_ids, length):
                branches = tuple(
                    _BranchSpec(
                        router_id=router_id,
                        group=group,
                        allowed=space.allowed_branch_ids(initial, router_id, group),
                        option_count=_ordered_subset_count(
                            len(space.allowed_branch_ids(initial, router_id, group))
                        ),
                    )
                    for router_id in initial
                    if router_id in space.router_ids
                    for group in space.profile.group_ids
                )
                block_size = math.prod(
                    (branch.option_count for branch in branches), start=1
                )
                blocks.append(
                    _InitialBlock(
                        start=offset,
                        stop=offset + block_size,
                        initial=tuple(initial),
                        branches=branches,
                    )
                )
                offset += block_size

        expected = legal_layout_count(space)
        if offset != expected:
            raise RuntimeError(
                f"Implicit catalogue contains {offset:,} layouts; expected {expected:,}."
            )
        self._blocks = tuple(blocks)
        self._stops = tuple(block.stop for block in blocks)
        self._size = offset

    def __len__(self) -> int:
        return self._size

    @property
    def block_count(self) -> int:
        """Number of materialized initial-sequence blocks, not layouts."""

        return len(self._blocks)

    def genome_at(self, index: int) -> TopologyGenome:
        if index < 0 or index >= self._size:
            raise IndexError("Layout index is outside the legal space.")
        block = self._blocks[bisect.bisect_right(self._stops, int(index))]
        local_index = int(index) - block.start

        option_indices = [0] * len(block.branches)
        for position in range(len(block.branches) - 1, -1, -1):
            count = block.branches[position].option_count
            local_index, option_indices[position] = divmod(local_index, count)
        if local_index:
            raise RuntimeError("Mixed-radix layout decoding left a nonzero index.")

        genome = TopologyGenome(
            initial=block.initial,
            branches=tuple(
                (
                    branch.router_id,
                    branch.group,
                    _ordered_subset_at(branch.allowed, option_index),
                )
                for branch, option_index in zip(block.branches, option_indices)
            ),
        )
        canonical = repair_genome(genome, self.space)
        if canonical != genome:
            raise RuntimeError("Implicit layout decoding produced a noncanonical genome.")
        return canonical

    def sampled_layout(self, space_index: int, sample_rank: int) -> SampledLayout:
        genome = self.genome_at(space_index)
        return SampledLayout(
            sample_rank=int(sample_rank),
            space_index=int(space_index),
            layout_id=layout_id(genome, self.space),
            genome=genome,
        )


def uniform_layout_stream(
    catalogue: ImplicitLayoutCatalogue, random_seed: int
) -> Iterator[SampledLayout]:
    """Yield an exact uniform permutation prefix using constant extra space.

    Independent uniform integer draws with duplicates discarded are equivalent
    to sampling ordered indices without replacement.  The set grows only with
    the evaluated prefix, not with the complete layout space.
    """

    rng = np.random.default_rng(random_seed)
    selected_indices: set[int] = set()
    while len(selected_indices) < len(catalogue):
        space_index = int(rng.integers(len(catalogue)))
        if space_index in selected_indices:
            continue
        selected_indices.add(space_index)
        yield catalogue.sampled_layout(space_index, len(selected_indices) - 1)


def uniform_layout_prefix(
    catalogue: ImplicitLayoutCatalogue, count: int, random_seed: int
) -> tuple[SampledLayout, ...]:
    if count < 0 or count > len(catalogue):
        raise ValueError("Prefix count is outside the legal layout space.")
    return tuple(islice(uniform_layout_stream(catalogue, random_seed), count))


def _implementation_sha256() -> str:
    """Fingerprint the source defining sampling and cached layout fitness."""

    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in (
        "experiments/m3n_vc/random_joint_optimize_hierarchy_with_k1.py",
        "experiments/m3n_vc/joint_optimize_hierarchy_ga_with_k1.py",
        "experiments/m3n_vc/joint_optimize_hierarchy_ga.py",
        "layout_search.py",
        "hierarchy_optimizer.py",
        "threshold_optimizer.py",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_cached_records(
    records: Mapping[str, Mapping[str, object]],
    catalogue: ImplicitLayoutCatalogue,
    sampling_seed: int,
    settings: Mapping[str, object],
) -> float:
    """Validate a deterministic append-only prefix and return elapsed time."""

    if len(records) > len(catalogue):
        raise ValueError("Cached evaluations exceed the legal layout space.")
    by_rank: dict[int, tuple[str, Mapping[str, object]]] = {}
    for cached_id, record in records.items():
        if not _settings_match(record.get("settings"), settings):
            raise ValueError("An evaluation belongs to a different experiment.")
        rank = int(record.get("sample_rank", -1))
        if rank < 0 or rank >= len(records) or rank in by_rank:
            raise ValueError("Cached random-search sample ranks are invalid.")
        by_rank[rank] = (cached_id, record)
    if set(by_rank) != set(range(len(records))):
        raise ValueError("Cached evaluations are not a contiguous sample prefix.")

    prior_elapsed = -1.0
    expected_prefix = uniform_layout_prefix(catalogue, len(records), sampling_seed)
    for expected in expected_prefix:
        cached_id, record = by_rank[expected.sample_rank]
        if cached_id != expected.layout_id or record.get("layout_id") != expected.layout_id:
            raise ValueError("A cached evaluation does not match the saved sample order.")
        if int(record.get("space_index", -1)) != expected.space_index:
            raise ValueError("A cached evaluation has the wrong canonical space index.")
        if int(record.get("layout_index", -1)) != expected.sample_rank:
            raise ValueError("A cached evaluation has the wrong evaluation-order index.")
        expected_layout = _cascade_payload(
            cascade_from_genome(expected.genome, catalogue.space)
        )
        if record.get("layout") != expected_layout:
            raise ValueError("A cached evaluation has the wrong cascade payload.")
        elapsed = float(record.get("search_elapsed_seconds_at_completion", -1.0))
        if elapsed < 0.0 or elapsed < prior_elapsed:
            raise ValueError("Cached evaluation elapsed times are invalid.")
        prior_elapsed = elapsed
    return max(prior_elapsed, 0.0)


def _prefix_sha256(records: Mapping[str, Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    ordered = sorted(records.values(), key=lambda record: int(record["sample_rank"]))
    for record in ordered:
        digest.update(
            f"{record['sample_rank']}:{record['space_index']}:{record['layout_id']}\n".encode(
                "ascii"
            )
        )
    return digest.hexdigest()


def _holdout_metrics(
    winner: Mapping[str, object],
    sampled: SampledLayout,
    holdout_optimizer: HierarchyOptimizer,
    space: LayoutSpace,
    target_accuracy: float,
) -> dict[str, object]:
    validation = winner.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("Winning record has no validation metrics.")
    cascade = cascade_from_genome(sampled.genome, space)
    if cascade.initial == [cascade.detector]:
        metrics = _direct_detector_metrics(holdout_optimizer, cascade, target_accuracy)
        method = "direct_detector_holdout"
    else:
        thresholds = validation.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise ValueError("The winning validation policy has no thresholds.")
        options: dict[str, object] = {"strict_thresholds": True}
        if "active_slots" in validation:
            options["active_slots"] = validation["active_slots"]
        metrics = FixedLayoutThresholdEvaluator(holdout_optimizer, cascade).evaluate(
            thresholds, **options
        )
        method = "validation_pruned_policy_holdout_replay"
    result = dict(metrics)
    result.update(
        {
            "feasible": bool(float(result["accuracy"]) >= target_accuracy),
            "target_accuracy": float(target_accuracy),
            "method": method,
        }
    )
    return _compact_optimization(result)


def run_k1_random_search(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    time_budget_seconds: float = DEFAULT_TIME_BUDGET_SECONDS,
    iterations: int = DEFAULT_ITERATIONS,
    restarts: int = DEFAULT_SA_RESTARTS,
    quantile_points: int = DEFAULT_QUANTILE_POINTS,
    inner_seed: int = DEFAULT_SEED,
    sampling_seed: int = DEFAULT_SEED,
    split_seed: int = DEFAULT_SEED,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    max_layouts: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run deterministic-prefix random search with winner-only holdout replay."""

    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1 inclusive.")
    if time_budget_seconds <= 0.0:
        raise ValueError("time_budget_seconds must be positive.")
    if iterations < 1 or restarts < 1 or quantile_points < 1:
        raise ValueError("iterations, restarts, and quantile_points must be positive.")
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1).")
    if max_layouts is not None and max_layouts < 1:
        raise ValueError("max_layouts must be positive when provided.")

    payload = load_empirical_outcomes(outcomes)
    space = build_k1_layout_space(payload)
    catalogue = ImplicitLayoutCatalogue(space)
    layout_count = len(catalogue)
    if max_layouts is not None and max_layouts > layout_count:
        raise ValueError("max_layouts exceeds the legal layout space.")

    settings: dict[str, object] = {
        "schema_version": SETTINGS_SCHEMA_VERSION,
        "algorithm": ALGORITHM,
        "dataset": "m3n_vc/h24",
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "removed_candidates": [],
        "layout_grammar": "depth_one_K0_K1",
        "layout_space_size": layout_count,
        "sampling_seed": int(sampling_seed),
        "layout_sampler": {
            "schema_version": SAMPLER_SCHEMA_VERSION,
            "selection": "exact_uniform_without_replacement",
            "random_generator": "numpy.default_rng",
            "deduplication_key": "canonical_space_index",
            "materializes_full_space": False,
        },
        "fitness_implementation_sha256": _implementation_sha256(),
        "target_accuracy": float(target_accuracy),
        "target_accuracy_source": "explicit_cli_or_api_override",
        "time_budget_seconds": float(time_budget_seconds),
        "max_layouts": max_layouts,
        "split_strategy": split_strategy,
        "split_seed": int(split_seed),
        "holdout_fraction": float(holdout_fraction),
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "quantile_points_compatibility_argument": int(quantile_points),
        "inner_seed": int(inner_seed),
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
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    settings_path = output_dir / "settings.json"
    results_path = output_dir / "evaluations.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    summary_path = output_dir / "summary.json"
    if overwrite:
        for path in (settings_path, results_path, checkpoint_path, summary_path):
            path.unlink(missing_ok=True)
    if settings_path.exists():
        existing_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if not _settings_match(existing_settings, settings):
            raise ValueError("Existing random-search checkpoint has different settings.")
    else:
        _write_json_atomic(settings_path, settings)

    records = _load_jsonl(results_path)
    recorded_elapsed = _validate_cached_records(
        records, catalogue, sampling_seed, settings
    )
    prior_elapsed = recorded_elapsed
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not _settings_match(checkpoint.get("settings"), settings):
            raise ValueError("Existing random-search checkpoint has different settings.")
        checkpoint_elapsed = float(checkpoint.get("search_elapsed_seconds", 0.0))
        if checkpoint_elapsed < recorded_elapsed:
            raise ValueError("Checkpoint elapsed time predates its cached evaluations.")
        prior_elapsed = checkpoint_elapsed
        if checkpoint.get("status") == "complete" and summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))

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
    fitness = InnerAnnealingFitness(
        validation_optimizer,
        target_accuracy=target_accuracy,
        quantile_points=quantile_points,
        iterations=iterations,
        restarts=restarts,
        inner_seed=inner_seed,
        settings=settings,
    )

    started = perf_counter()
    new_evaluations = 0
    stop_reason = "layout_space_exhausted"
    completed_count = len(records)
    for sampled in uniform_layout_stream(catalogue, sampling_seed):
        if sampled.sample_rank < completed_count:
            continue
        elapsed = prior_elapsed + perf_counter() - started
        if elapsed >= time_budget_seconds:
            stop_reason = "time_budget_reached"
            break
        if max_layouts is not None and len(records) >= max_layouts:
            stop_reason = "max_layouts_reached"
            break

        indexed = IndexedLayout(
            sampled.sample_rank,
            sampled.layout_id,
            cascade_from_genome(sampled.genome, space),
        )
        evaluation_started = perf_counter()
        record = fitness(indexed)
        record["sample_rank"] = sampled.sample_rank
        record["space_index"] = sampled.space_index
        record["evaluation_wall_seconds"] = perf_counter() - evaluation_started
        record["search_elapsed_seconds_at_completion"] = (
            prior_elapsed + perf_counter() - started
        )
        with results_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
        records[sampled.layout_id] = record
        new_evaluations += 1

        elapsed = prior_elapsed + perf_counter() - started
        _write_json_atomic(
            checkpoint_path,
            {
                "status": "running",
                "settings": settings,
                "evaluated_layouts": len(records),
                "sample_prefix_sha256": _prefix_sha256(records),
                "search_elapsed_seconds": elapsed,
            },
        )
        if new_evaluations % 16 == 0:
            best = min(records.values(), key=_layout_selection_key)
            validation = best["validation"]
            assert isinstance(validation, Mapping)
            print(
                f"K1 random search: {len(records):,} layouts; "
                f"best={float(validation['expected_cost']):.3f} ms; "
                f"elapsed={elapsed:.1f}/{time_budget_seconds:.1f} s"
            )

    search_elapsed = prior_elapsed + perf_counter() - started
    if not records:
        raise RuntimeError("The time budget ended before any layout was evaluated.")
    winner = dict(min(records.values(), key=_layout_selection_key))
    winner_rank = int(winner["sample_rank"])
    winner_sampled = uniform_layout_prefix(
        catalogue, winner_rank + 1, sampling_seed
    )[winner_rank]
    holdout_optimizer = HierarchyOptimizer(
        holdout_payload,
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )
    winner["holdout"] = _holdout_metrics(
        winner, winner_sampled, holdout_optimizer, space, target_accuracy
    )
    total_elapsed = prior_elapsed + perf_counter() - started
    summary: dict[str, object] = {
        "settings": settings,
        "split": split,
        "stop_reason": stop_reason,
        "search_elapsed_seconds": search_elapsed,
        "total_elapsed_seconds_including_holdout": total_elapsed,
        "time_budget_overshoot_seconds": max(
            0.0, search_elapsed - time_budget_seconds
        ),
        "unique_layouts_evaluated": len(records),
        "new_evaluations_this_invocation": new_evaluations,
        "fraction_of_layout_space": len(records) / layout_count,
        "sample_prefix_sha256": _prefix_sha256(records),
        "evaluations": str(results_path.resolve()),
        "evaluations_sha256": _file_sha256(results_path),
        "winner": winner,
        "holdout_usage": "winner_only_after_validation_search",
    }
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(
        checkpoint_path,
        {
            "status": "complete",
            "settings": settings,
            "evaluated_layouts": len(records),
            "sample_prefix_sha256": _prefix_sha256(records),
            "search_elapsed_seconds": search_elapsed,
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY)
    parser.add_argument(
        "--time-budget-seconds", type=float, default=DEFAULT_TIME_BUDGET_SECONDS
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--restarts", type=int, default=DEFAULT_SA_RESTARTS)
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--inner-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sampling-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--split-strategy", default=DEFAULT_SPLIT_STRATEGY)
    parser.add_argument("--max-layouts", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run:
        payload = load_empirical_outcomes(args.outcomes)
        catalogue = ImplicitLayoutCatalogue(build_k1_layout_space(payload))
        print(
            json.dumps(
                {
                    "legal_layouts": len(catalogue),
                    "materialized_initial_blocks": catalogue.block_count,
                    "target_accuracy": args.target_accuracy,
                    "iterations_per_layout": args.iterations,
                    "restarts_per_layout": args.restarts,
                    "total_iterations_per_layout": args.iterations * args.restarts,
                    "sampling": "exact_uniform_without_replacement",
                },
                indent=2,
            )
        )
        return
    summary = run_k1_random_search(
        outcomes=args.outcomes,
        output_dir=args.output_dir,
        target_accuracy=args.target_accuracy,
        time_budget_seconds=args.time_budget_seconds,
        iterations=args.iterations,
        restarts=args.restarts,
        quantile_points=args.quantile_points,
        inner_seed=args.inner_seed,
        sampling_seed=args.sampling_seed,
        split_seed=args.split_seed,
        holdout_fraction=args.holdout_fraction,
        split_strategy=args.split_strategy,
        max_layouts=args.max_layouts,
        overwrite=args.overwrite,
    )
    winner = summary["winner"]
    assert isinstance(winner, Mapping)
    validation = winner["validation"]
    holdout = winner["holdout"]
    assert isinstance(validation, Mapping) and isinstance(holdout, Mapping)
    print(
        f"Winner {winner['layout_id']}: validation "
        f"{float(validation['accuracy']):.6f} / "
        f"{float(validation['expected_cost']):.3f} ms; holdout "
        f"{float(holdout['accuracy']):.6f} / "
        f"{float(holdout['expected_cost']):.3f} ms"
    )


if __name__ == "__main__":
    main()
