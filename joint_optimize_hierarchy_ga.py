"""Jointly optimize a K1-free hierarchy layout and its thresholds.

The outer optimizer is a constrained memetic genetic algorithm (GA).  A
genome contains the initial cascade and the two K0 branches. The fitness of
every previously unseen, non-detector-only genome is obtained by running the
*same* threshold optimizer used by :mod:`brute_force_k1_free_layouts`: 8,000
simulated-annealing iterations on 50-point confidence grids, followed by
coordinate descent. The detector-only topology is scored directly, as in the
brute force. Thus the approximation is only over which of the 5,545 layouts
are visited; a visited layout is evaluated identically to the exhaustive run.

The defaults reproduce the brute-force experimental contract:

* h24 empirical outcomes with K1 removed;
* paper Kdet (perfect, 10,000 ms);
* blocked-per-run 80/20 validation/holdout split;
* the Fig. 1 K3 validation-accuracy target;
* 50 confidence quantiles; and
* an 8,000-step inner anneal with seed 0.

Only validation outcomes participate in the GA. The holdout is not consulted
or evaluated, and the optional exhaustive reference is not read, until the
winning policy is frozen. Results are checkpointed by generation and layout
evaluations are cached by canonical layout id, so an interrupted run can be
resumed.

Examples
--------
Inspect the default budget and estimated runtime::

    python joint_optimize_hierarchy_ga.py --dry-run

Run the default 512-layout search::

    python joint_optimize_hierarchy_ga.py

Run the progress-annealed outer GA in its own checkpoint directory::

    python joint_optimize_hierarchy_ga.py --annealed-outer-schedule

Evaluate one generation concurrently on a multi-core node::

    python joint_optimize_hierarchy_ga.py --workers 8

Run independent outer-search repetitions without changing the split or the
inner annealer::

    python joint_optimize_hierarchy_ga.py --outer-seed 1 \
        --output-dir checkpoints/joint_ga_k1_free_h24_seed1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from brute_force_k1_free_layouts import (
    DEFAULT_HOLDOUT_FRACTION,
    DEFAULT_ITERATIONS,
    DEFAULT_OUTCOMES,
    DEFAULT_SEED,
    DEFAULT_SPLIT_STRATEGY,
    EXPECTED_LAYOUT_COUNT,
    FIG1_K3_TARGET_ACCURACY,
    REMOVED_CANDIDATES,
    IndexedLayout,
    _cascade_payload,
    _compact_optimization,
    _direct_detector_metrics,
    _without_candidates,
    enumerate_k1_free_layouts,
    layout_id,
)
from empirical_outcomes import load_empirical_outcomes
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


DEFAULT_OUTPUT_DIR = Path("checkpoints/joint_ga_k1_free_h24")
DEFAULT_BRUTE_FORCE_SUMMARY = Path(
    "checkpoints/brute_force_k1_free_h24/summary_shard_00000_of_00001.json"
)
DEFAULT_BRUTE_FORCE_RESULTS = Path(
    "checkpoints/brute_force_k1_free_h24/results_shard_00000_of_00001.jsonl"
)

DEFAULT_POPULATION_SIZE = 32
DEFAULT_GENERATIONS = 24
DEFAULT_EVALUATION_BUDGET = 512
DEFAULT_ELITE_COUNT = 4
DEFAULT_TOURNAMENT_SIZE = 2
DEFAULT_CROSSOVER_RATE = 0.80
DEFAULT_MUTATION_RATE = 0.80
DEFAULT_RANDOM_IMMIGRANT_RATE = 0.20
DEFAULT_COMPONENT_RESAMPLE_RATE = 0.30
DEFAULT_STAGNATION_GENERATIONS = 6
DEFAULT_MAX_RESTARTS = 3
DEFAULT_ANNEALED_OUTPUT_DIR = Path("checkpoints/joint_ga_annealed_k1_free_h24")

# Optional progress-based outer-GA schedule. This anneals only topology-search
# behavior; every visited layout still receives the same independent 8k inner
# threshold anneal. Values are intentionally kept explicit in checkpoints.
ANNEALED_OUTER_SCHEDULE = {
    "random_immigrant_rate": {"start": 0.40, "end": 0.05},
    "component_resample_rate": {"start": 0.60, "end": 0.10},
    "mutation_rate": {"start": 0.95, "end": 0.50},
    "crossover_rate": {"start": 0.60, "end": 0.90},
    "tournament_size": {"start": 2, "end": 4},
    "elite_count": {"start": 2, "end": 6},
}

# The completed exhaustive run is a better estimate than the older, single-
# layout microbenchmark retained in brute_force_k1_free_layouts.py.
OBSERVED_SECONDS_PER_LAYOUT = 17_066.2 / EXPECTED_LAYOUT_COUNT

ROUTER_ID = "K0"
DETECTOR_ID = "detector"
GLOBAL_IDS = ("K2", "K3")
INITIAL_IDS = (ROUTER_ID, *GLOBAL_IDS)
SPECIALISTS = {
    "coupe": ("K5", "K6"),
    "suv": ("K4",),
}
GROUP_ORDER = ("coupe", "suv")


@dataclass(frozen=True)
class TopologyGenome:
    """Hashable GA representation, excluding terminal detector stages."""

    initial: tuple[str, ...]
    coupe: tuple[str, ...] = ()
    suv: tuple[str, ...] = ()


@dataclass(frozen=True)
class OuterGAParameters:
    """Resolved topology-search controls for one generation."""

    elite_count: int
    tournament_size: int
    crossover_rate: float
    mutation_rate: float
    random_immigrant_rate: float
    component_resample_rate: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "elite_count": int(self.elite_count),
            "tournament_size": int(self.tournament_size),
            "crossover_rate": float(self.crossover_rate),
            "mutation_rate": float(self.mutation_rate),
            "random_immigrant_rate": float(self.random_immigrant_rate),
            "component_resample_rate": float(self.component_resample_rate),
        }


def _lerp(start: float, end: float, progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    if progress == 0.0:
        return float(start)
    if progress == 1.0:
        return float(end)
    return float(start + (end - start) * progress)


def outer_ga_parameters(
    progress: float,
    *,
    annealed: bool,
    elite_count: int = DEFAULT_ELITE_COUNT,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    random_immigrant_rate: float = DEFAULT_RANDOM_IMMIGRANT_RATE,
    component_resample_rate: float = DEFAULT_COMPONENT_RESAMPLE_RATE,
) -> OuterGAParameters:
    """Resolve fixed or linearly annealed controls at search progress [0, 1]."""

    if not annealed:
        return OuterGAParameters(
            elite_count=elite_count,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            random_immigrant_rate=random_immigrant_rate,
            component_resample_rate=component_resample_rate,
        )

    def scheduled(name: str) -> float:
        endpoints = ANNEALED_OUTER_SCHEDULE[name]
        return _lerp(endpoints["start"], endpoints["end"], progress)

    # Adding 0.5 implements intuitive nearest-integer interpolation rather
    # than Python's banker rounding at exact half steps.
    return OuterGAParameters(
        elite_count=int(scheduled("elite_count") + 0.5),
        tournament_size=int(scheduled("tournament_size") + 0.5),
        crossover_rate=scheduled("crossover_rate"),
        mutation_rate=scheduled("mutation_rate"),
        random_immigrant_rate=scheduled("random_immigrant_rate"),
        component_resample_rate=scheduled("component_resample_rate"),
    )


@dataclass(frozen=True)
class LayoutCatalogue:
    """The exact legal search space shared with the exhaustive optimizer."""

    entries: tuple[IndexedLayout, ...]
    by_id: Mapping[str, IndexedLayout]
    genome_to_id: Mapping[TopologyGenome, str]

    def entry_for_genome(self, genome: TopologyGenome) -> IndexedLayout:
        canonical = repair_genome(genome)
        try:
            return self.by_id[self.genome_to_id[canonical]]
        except KeyError as exc:  # pragma: no cover - defensive invariant
            raise ValueError(f"Genome is outside the legal catalogue: {canonical}") from exc

    def genome_for_id(self, candidate_layout_id: str) -> TopologyGenome:
        try:
            return genome_from_cascade(self.by_id[candidate_layout_id].cascade)
        except KeyError as exc:
            raise ValueError(f"Unknown layout id: {candidate_layout_id}") from exc


def genome_from_cascade(cascade: Cascade) -> TopologyGenome:
    """Convert a materialized cascade into the stable genetic representation."""

    initial = tuple(item for item in cascade.initial if item != cascade.detector)
    if ROUTER_ID not in initial:
        return TopologyGenome(initial)
    branches = {
        group: tuple(
            item
            for item in cascade.specialized.get((ROUTER_ID, group), ())
            if item != cascade.detector
        )
        for group in GROUP_ORDER
    }
    return TopologyGenome(initial, branches["coupe"], branches["suv"])


def cascade_from_genome(genome: TopologyGenome) -> Cascade:
    """Materialize a repaired genome using brute-force branch insertion order."""

    genome = repair_genome(genome)
    specialized: dict[tuple[str, str], list[str]] = {}
    if ROUTER_ID in genome.initial:
        # Insertion order matters to the seeded inner annealer's coordinate
        # order.  This is the same sorted order used by the enumerator.
        specialized[(ROUTER_ID, "coupe")] = [*genome.coupe, DETECTOR_ID]
        specialized[(ROUTER_ID, "suv")] = [*genome.suv, DETECTOR_ID]
    return Cascade(
        expected_cost=0.0,
        initial=[*genome.initial, DETECTOR_ID],
        specialized=specialized,
        detector=DETECTOR_ID,
    )


def _deduplicate_allowed(
    values: Iterable[str], allowed: Iterable[str]
) -> tuple[str, ...]:
    allowed_set = set(allowed)
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in allowed_set and value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)


def repair_genome(genome: TopologyGenome) -> TopologyGenome:
    """Project arbitrary chain edits back onto the exact K1-free grammar."""

    initial = _deduplicate_allowed(genome.initial, INITIAL_IDS)
    if ROUTER_ID not in initial:
        return TopologyGenome(initial)

    router_position = initial.index(ROUTER_ID)
    preceding_globals = set(initial[:router_position]) & set(GLOBAL_IDS)
    remaining_globals = tuple(
        candidate_id
        for candidate_id in GLOBAL_IDS
        if candidate_id not in preceding_globals
    )
    coupe = _deduplicate_allowed(
        genome.coupe,
        (*remaining_globals, *SPECIALISTS["coupe"]),
    )
    suv = _deduplicate_allowed(
        genome.suv,
        (*remaining_globals, *SPECIALISTS["suv"]),
    )
    return TopologyGenome(initial, coupe, suv)


def build_layout_catalogue() -> LayoutCatalogue:
    """Build and cross-check all 5,545 legal layouts once."""

    entries = tuple(enumerate_k1_free_layouts())
    if len(entries) != EXPECTED_LAYOUT_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_LAYOUT_COUNT:,} layouts, found {len(entries):,}."
        )
    by_id = {entry.layout_id: entry for entry in entries}
    genome_to_id = {
        genome_from_cascade(entry.cascade): entry.layout_id for entry in entries
    }
    if len(by_id) != len(entries) or len(genome_to_id) != len(entries):
        raise RuntimeError("The legal layout catalogue contains duplicate states.")

    # Catch any divergence between this representation and the authoritative
    # brute-force materialization before spending time on inner annealing.
    for genome, expected_id in genome_to_id.items():
        actual_id = layout_id(cascade_from_genome(genome))
        if actual_id != expected_id:
            raise RuntimeError(
                f"Genome round trip changed layout id {expected_id} to {actual_id}."
            )
    return LayoutCatalogue(entries, by_id, genome_to_id)


def topology_selection_key(
    record: Mapping[str, object], target_accuracy: float
) -> tuple[float, ...]:
    """Use exactly the exhaustive search's constrained lexicographic order."""

    validation = record["validation"]
    if not isinstance(validation, Mapping):
        raise TypeError("A fitness record must contain validation metrics.")
    accuracy = float(validation["accuracy"])
    cost = float(validation["expected_cost"])
    feasible = accuracy >= target_accuracy
    if feasible:
        return (0.0, cost, -accuracy, float(record["layout_index"]))
    return (1.0, -accuracy, cost, float(record["layout_index"]))


def _random_ordered_subset(
    allowed: Sequence[str], rng: np.random.Generator
) -> tuple[str, ...]:
    length = int(rng.integers(0, len(allowed) + 1))
    if length == 0:
        return ()
    selected = rng.choice(np.asarray(allowed, dtype=object), size=length, replace=False)
    return tuple(str(item) for item in selected)


def _edit_sequence(
    sequence: Sequence[str],
    allowed: Sequence[str],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """Apply one variable-length ordered-set mutation."""

    values = list(sequence)
    missing = [item for item in allowed if item not in values]
    operations: list[str] = []
    if missing:
        operations.append("insert")
    if values:
        operations.append("delete")
    if len(values) >= 2:
        operations.extend(("swap", "relocate"))
    if values and missing:
        operations.append("replace")
    operation = operations[int(rng.integers(0, len(operations)))]

    if operation == "insert":
        value = missing[int(rng.integers(0, len(missing)))]
        position = int(rng.integers(0, len(values) + 1))
        values.insert(position, value)
    elif operation == "delete":
        del values[int(rng.integers(0, len(values)))]
    elif operation == "swap":
        first, second = rng.choice(len(values), size=2, replace=False)
        values[int(first)], values[int(second)] = (
            values[int(second)],
            values[int(first)],
        )
    elif operation == "relocate":
        source = int(rng.integers(0, len(values)))
        value = values.pop(source)
        destination = int(rng.integers(0, len(values) + 1))
        values.insert(destination, value)
    elif operation == "replace":
        position = int(rng.integers(0, len(values)))
        values[position] = missing[int(rng.integers(0, len(missing)))]
    return tuple(values)


def mutate_genome(
    genome: TopologyGenome,
    rng: np.random.Generator,
    *,
    component_resample_rate: float = DEFAULT_COMPONENT_RESAMPLE_RATE,
) -> TopologyGenome:
    """Make one local edit or resample one complete topology component."""

    child = repair_genome(genome)
    if ROUTER_ID not in child.initial:
        component = "initial"
    else:
        component = ("initial", "coupe", "suv")[
            int(rng.choice(3, p=(0.40, 0.30, 0.30)))
        ]

    if component == "initial":
        edited = (
            _random_ordered_subset(INITIAL_IDS, rng)
            if rng.random() < component_resample_rate
            else _edit_sequence(child.initial, INITIAL_IDS, rng)
        )
        return repair_genome(
            TopologyGenome(edited, child.coupe, child.suv)
        )

    router_position = child.initial.index(ROUTER_ID)
    preceding = set(child.initial[:router_position]) & set(GLOBAL_IDS)
    allowed = tuple(
        item
        for item in (*GLOBAL_IDS, *SPECIALISTS[component])
        if item not in preceding
    )
    current = getattr(child, component)
    edited = (
        _random_ordered_subset(allowed, rng)
        if rng.random() < component_resample_rate
        else _edit_sequence(current, allowed, rng)
    )
    return repair_genome(
        TopologyGenome(
            child.initial,
            edited if component == "coupe" else child.coupe,
            edited if component == "suv" else child.suv,
        )
    )


def crossover_genomes(
    first: TopologyGenome,
    second: TopologyGenome,
    rng: np.random.Generator,
) -> TopologyGenome:
    """Recombine trunk/branch modules, with occasional ordered-set mixing."""

    def recombine(
        first_chain: tuple[str, ...], second_chain: tuple[str, ...]
    ) -> tuple[str, ...]:
        # Most exchanges retain a complete module, which preserves epistatic
        # routing behavior. A smaller fraction combines membership and order
        # within the module using a standard uniform ordered-set crossover.
        if rng.random() < 0.70:
            return first_chain if rng.random() < 0.5 else second_chain
        primary, secondary = (
            (first_chain, second_chain)
            if rng.random() < 0.5
            else (second_chain, first_chain)
        )
        included: set[str] = set()
        for item in dict.fromkeys((*primary, *secondary)):
            if item in primary and item in secondary:
                included.add(item)
            elif rng.random() < 0.5:
                included.add(item)
        return tuple(
            item
            for item in dict.fromkeys((*primary, *secondary))
            if item in included
        )

    return repair_genome(
        TopologyGenome(
            initial=recombine(first.initial, second.initial),
            coupe=recombine(first.coupe, second.coupe),
            suv=recombine(first.suv, second.suv),
        )
    )


def initial_population(
    catalogue: LayoutCatalogue,
    population_size: int,
    rng: np.random.Generator,
    *,
    extra_seeds: Sequence[TopologyGenome] = (),
) -> list[str]:
    """Create a diverse population from honest baselines and random layouts."""

    selected: list[str] = []
    selected_set: set[str] = set()

    def add(genome: TopologyGenome) -> None:
        repaired = repair_genome(genome)
        candidate_id = catalogue.genome_to_id.get(repaired)
        if candidate_id is not None and candidate_id not in selected_set:
            selected.append(candidate_id)
            selected_set.add(candidate_id)

    # K3 -> detector is the published pre-search reference. Everything else
    # is sampled uniformly; no topology learned from the exhaustive oracle is
    # injected into the approximate search.
    for seed in (TopologyGenome(("K3",)), *extra_seeds):
        if len(selected) >= population_size:
            break
        add(seed)

    # Uniform catalogue sampling prevents the hand-written seeds from
    # constraining discovery to one local basin.
    permutation = rng.permutation(len(catalogue.entries))
    for position in permutation:
        if len(selected) >= population_size:
            break
        candidate_id = catalogue.entries[int(position)].layout_id
        if candidate_id not in selected_set:
            selected.append(candidate_id)
            selected_set.add(candidate_id)
    if len(selected) != population_size:
        raise RuntimeError("Could not construct a full unique initial population.")
    return selected


def _tournament_parent(
    population: Sequence[str],
    records: Mapping[str, Mapping[str, object]],
    target_accuracy: float,
    tournament_size: int,
    rng: np.random.Generator,
) -> str:
    count = min(tournament_size, len(population))
    contestants = rng.choice(
        np.asarray(population, dtype=object), size=count, replace=False
    )
    return min(
        (str(item) for item in contestants),
        key=lambda candidate_id: topology_selection_key(
            records[candidate_id], target_accuracy
        ),
    )


def next_population(
    population: Sequence[str],
    records: Mapping[str, Mapping[str, object]],
    catalogue: LayoutCatalogue,
    rng: np.random.Generator,
    *,
    target_accuracy: float,
    population_size: int,
    elite_count: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    random_immigrant_rate: float,
    component_resample_rate: float = DEFAULT_COMPONENT_RESAMPLE_RATE,
    excluded_layout_ids: set[str] | None = None,
) -> list[str]:
    """Breed one unique population while retaining constrained elites."""

    ranked = sorted(
        population,
        key=lambda candidate_id: topology_selection_key(
            records[candidate_id], target_accuracy
        ),
    )
    next_ids = list(ranked[: min(elite_count, population_size)])
    next_set = set(next_ids)
    excluded = set() if excluded_layout_ids is None else set(excluded_layout_ids)
    all_ids = tuple(entry.layout_id for entry in catalogue.entries)

    attempts = 0
    max_attempts = max(1_000, population_size * 200)
    while len(next_ids) < population_size and attempts < max_attempts:
        attempts += 1
        if rng.random() < random_immigrant_rate:
            child_id = all_ids[int(rng.integers(0, len(all_ids)))]
        else:
            first_id = _tournament_parent(
                population, records, target_accuracy, tournament_size, rng
            )
            first = catalogue.genome_for_id(first_id)
            child = first
            if rng.random() < crossover_rate:
                second_id = _tournament_parent(
                    population, records, target_accuracy, tournament_size, rng
                )
                second = catalogue.genome_for_id(second_id)
                child = crossover_genomes(first, second, rng)
            if rng.random() < mutation_rate:
                child = mutate_genome(
                    child,
                    rng,
                    component_resample_rate=component_resample_rate,
                )
            child_id = catalogue.entry_for_genome(child).layout_id

        if child_id in next_set or child_id in excluded:
            continue
        next_ids.append(child_id)
        next_set.add(child_id)

    if len(next_ids) < population_size:
        # Near exhaustion, unbiased catalogue fill guarantees progress.
        for position in rng.permutation(len(all_ids)):
            child_id = all_ids[int(position)]
            if child_id in next_set or child_id in excluded:
                continue
            next_ids.append(child_id)
            next_set.add(child_id)
            if len(next_ids) == population_size:
                break
    if len(next_ids) != population_size:
        raise RuntimeError("The requested unique GA population cannot be filled.")
    return next_ids


def restart_population(
    records: Mapping[str, Mapping[str, object]],
    catalogue: LayoutCatalogue,
    rng: np.random.Generator,
    *,
    target_accuracy: float,
    population_size: int,
) -> list[str]:
    """Keep the global elite and refill uniformly from unseen layouts."""

    global_elite = str(_best_record(records, target_accuracy)["layout_id"])
    population = [global_elite]
    seen = set(records)
    for position in rng.permutation(len(catalogue.entries)):
        candidate_id = catalogue.entries[int(position)].layout_id
        if candidate_id in seen:
            continue
        population.append(candidate_id)
        if len(population) == population_size:
            break
    if len(population) != population_size:
        raise RuntimeError("There are too few unseen layouts for a GA restart.")
    return population


class InnerAnnealingFitness:
    """Exact brute-force-compatible validation fitness for one topology."""

    def __init__(
        self,
        optimizer: HierarchyOptimizer,
        *,
        target_accuracy: float,
        quantile_points: int,
        iterations: int,
        inner_seed: int,
        settings: Mapping[str, object],
    ) -> None:
        self.optimizer = optimizer
        self.target_accuracy = float(target_accuracy)
        self.quantile_points = int(quantile_points)
        self.iterations = int(iterations)
        self.inner_seed = int(inner_seed)
        self.settings = dict(settings)

    def __call__(self, indexed: IndexedLayout) -> dict[str, object]:
        cascade = indexed.cascade
        if cascade.initial == [cascade.detector]:
            metrics = _direct_detector_metrics(
                self.optimizer, cascade, self.target_accuracy
            )
        else:
            evaluator = FixedLayoutThresholdEvaluator(self.optimizer, cascade)
            metrics = optimize_fixed_layout_thresholds_simulated_annealing(
                evaluator,
                self.target_accuracy,
                quantile_points=self.quantile_points,
                n_iterations=self.iterations,
                random_seed=self.inner_seed,
                show_progress=False,
            )
        metrics = dict(metrics)
        metrics["feasible"] = bool(
            float(metrics["accuracy"]) >= self.target_accuracy
        )
        return {
            "layout_index": int(indexed.index),
            "layout_id": indexed.layout_id,
            "layout": _cascade_payload(cascade),
            "settings": dict(self.settings),
            "validation": _compact_optimization(metrics),
        }


def _load_jsonl(path: Path) -> dict[str, dict[str, object]]:
    """Load a cache and repair only an interrupted trailing JSONL record."""

    records: dict[str, dict[str, object]] = {}
    if not path.exists():
        return records
    truncate_at: int | None = None
    needs_final_newline = False
    with path.open("rb") as handle:
        line_number = 0
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line_number += 1
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                # A killed write may leave one partial final record. Silently
                # skipping it and then opening in append mode would concatenate
                # the next record onto corrupt bytes forever, so truncate it.
                if handle.read().strip():
                    raise ValueError(
                        f"Malformed non-final JSONL line {line_number} in {path}."
                    ) from exc
                truncate_at = line_start
                print(f"Removing incomplete JSONL line {line_number} in {path}")
                break
            records[str(record["layout_id"])] = record
            needs_final_newline = not raw_line.endswith(b"\n")

    if truncate_at is not None:
        with path.open("r+b") as handle:
            handle.truncate(truncate_at)
    elif needs_final_newline:
        # A complete JSON object without its final line ending is valid JSON,
        # but must be separated before future append-only checkpoint records.
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fitness_implementation_sha256() -> str:
    """Fingerprint source files that define cached layout fitness."""

    source_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "brute_force_k1_free_layouts.py",
        "hierarchy_optimizer.py",
        "joint_optimize_hierarchy_ga.py",
        "threshold_optimizer.py",
    ):
        path = source_dir / name
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _catalogue_sha256(catalogue: LayoutCatalogue) -> str:
    digest = hashlib.sha256()
    for entry in catalogue.entries:
        digest.update(f"{entry.index}:{entry.layout_id}\n".encode("ascii"))
    return digest.hexdigest()


def _settings_match(
    actual: object, expected: Mapping[str, object]
) -> bool:
    return isinstance(actual, Mapping) and dict(actual) == dict(expected)


def _evaluate_missing(
    population: Sequence[str],
    records: dict[str, dict[str, object]],
    catalogue: LayoutCatalogue,
    evaluate: Callable[[IndexedLayout], dict[str, object]],
    results_path: Path,
    *,
    workers: int,
) -> int:
    missing_ids = [candidate_id for candidate_id in population if candidate_id not in records]
    if not missing_ids:
        return 0
    entries = [catalogue.by_id[candidate_id] for candidate_id in missing_ids]
    completed = 0
    with results_path.open("a", encoding="utf-8", buffering=1) as handle:
        def save(iterator: Iterable[dict[str, object]]) -> None:
            nonlocal completed
            for record in tqdm(
                iterator,
                total=len(entries),
                desc="Inner-SA layout fitness",
            ):
                candidate_id = str(record["layout_id"])
                handle.write(json.dumps(record, sort_keys=True, default=float) + "\n")
                records[candidate_id] = record
                completed += 1

        if workers == 1:
            save(map(evaluate, entries))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                save(executor.map(evaluate, entries))
    return completed


def _best_record(
    records: Mapping[str, Mapping[str, object]], target_accuracy: float
) -> Mapping[str, object]:
    if not records:
        raise ValueError("No layouts have been evaluated.")
    return min(
        records.values(),
        key=lambda record: topology_selection_key(record, target_accuracy),
    )


def _history_item(
    generation: int,
    records: Mapping[str, Mapping[str, object]],
    target_accuracy: float,
    new_evaluations: int,
) -> dict[str, object]:
    best = _best_record(records, target_accuracy)
    validation = best["validation"]
    assert isinstance(validation, Mapping)
    return {
        "generation": int(generation),
        "unique_layouts_evaluated": int(len(records)),
        "new_layouts_evaluated": int(new_evaluations),
        "best_layout_id": str(best["layout_id"]),
        "best_layout_index": int(best["layout_index"]),
        "best_validation_accuracy": float(validation["accuracy"]),
        "best_validation_cost_ms": float(validation["expected_cost"]),
    }


def _pareto_archive(
    records: Mapping[str, Mapping[str, object]]
) -> list[dict[str, object]]:
    """Return validation cost/accuracy nondominated points for analysis."""

    points: list[tuple[float, float, Mapping[str, object]]] = []
    for record in records.values():
        validation = record["validation"]
        assert isinstance(validation, Mapping)
        points.append(
            (
                float(validation["expected_cost"]),
                float(validation["accuracy"]),
                record,
            )
        )
    points.sort(key=lambda item: (item[0], -item[1], int(item[2]["layout_index"])))
    archive: list[dict[str, object]] = []
    best_accuracy = -float("inf")
    for cost, accuracy, record in points:
        if accuracy <= best_accuracy:
            continue
        archive.append(
            {
                "layout_id": str(record["layout_id"]),
                "layout_index": int(record["layout_index"]),
                "validation_cost_ms": cost,
                "validation_accuracy": accuracy,
            }
        )
        best_accuracy = accuracy
    return archive


def _final_holdout(
    winner: Mapping[str, object],
    holdout_optimizer: HierarchyOptimizer,
    catalogue: LayoutCatalogue,
    target_accuracy: float,
) -> dict[str, object]:
    """Evaluate holdout exactly once, after validation selection is complete."""

    candidate_id = str(winner["layout_id"])
    cascade = catalogue.by_id[candidate_id].cascade
    validation = winner["validation"]
    assert isinstance(validation, Mapping)
    if cascade.initial == [cascade.detector]:
        metrics = _direct_detector_metrics(
            holdout_optimizer, cascade, target_accuracy
        )
    else:
        evaluator = FixedLayoutThresholdEvaluator(holdout_optimizer, cascade)
        thresholds = validation["thresholds"]
        assert isinstance(thresholds, Mapping)
        metrics = evaluator.evaluate(thresholds)
    metrics = dict(metrics)
    metrics["feasible"] = bool(float(metrics["accuracy"]) >= target_accuracy)
    return _compact_optimization(metrics)


def _read_brute_force_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def compare_with_exhaustive(
    winner: Mapping[str, object],
    *,
    target_accuracy: float,
    settings: Mapping[str, object],
    evaluated_layout_count: int,
    summary_path: Path | None,
    results_path: Path | None,
) -> dict[str, object] | None:
    """Benchmark a frozen winner; this function is never called by search."""

    if summary_path is None or not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reference = summary.get("best")
    if not isinstance(reference, Mapping):
        return None
    reference_settings = reference.get("settings", summary.get("settings", {}))
    exact_fields = (
        "detector_mode",
        "detector_cost_ms",
        "holdout_fraction",
        "iterations",
        "outcomes",
        "quantile_points",
        "removed_candidates",
        "split_strategy",
        "target_accuracy",
    )
    def settings_are_comparable(candidate: object) -> bool:
        if not isinstance(candidate, Mapping):
            return False
        same = all(
            candidate.get(field) == settings.get(field) for field in exact_fields
        )
        # The brute-force "seed" drives both splitting and inner SA; the GA
        # keeps those distinct so outer repetitions do not alter either one.
        return bool(
            same
            and candidate.get("seed") == settings.get("inner_seed")
            and candidate.get("seed") == settings.get("split_seed")
        )

    comparable = settings_are_comparable(reference_settings)
    completed_layouts = summary.get(
        "completed_layouts", summary.get("completed_unique_layouts")
    )
    reference_complete = bool(summary.get("complete")) or (
        completed_layouts == EXPECTED_LAYOUT_COUNT
        and summary.get("expected_total_layouts") == EXPECTED_LAYOUT_COUNT
    )
    comparison: dict[str, object] = {
        "reference_summary": str(summary_path.resolve()),
        "settings_comparable": bool(comparable),
        "reference_complete": bool(reference_complete),
        "comparison_available": bool(comparable and reference_complete),
    }
    if not comparable or not reference_complete:
        reasons: list[str] = []
        if not comparable:
            reasons.append("settings_mismatch")
        if not reference_complete:
            reasons.append("reference_is_not_complete_exhaustive_search")
        comparison["unavailable_reasons"] = reasons
        return comparison

    winner_validation = winner["validation"]
    reference_validation = reference["validation"]
    assert isinstance(winner_validation, Mapping)
    assert isinstance(reference_validation, Mapping)
    comparison.update({
        "exact_layout_recovered": str(winner["layout_id"])
        == str(reference["layout_id"]),
        "optimal_layout_id": str(reference["layout_id"]),
        "optimal_layout_index": int(reference["layout_index"]),
        "optimal_validation_cost_ms": float(reference_validation["expected_cost"]),
        "optimal_validation_accuracy": float(reference_validation["accuracy"]),
        "validation_cost_regret_ms": float(winner_validation["expected_cost"])
        - float(reference_validation["expected_cost"]),
        "validation_accuracy_difference": float(winner_validation["accuracy"])
        - float(reference_validation["accuracy"]),
    })

    if results_path is not None and results_path.exists():
        exhaustive_records = _read_brute_force_records(results_path)
        unique_ids = {str(record["layout_id"]) for record in exhaustive_records}
        indices = {int(record["layout_index"]) for record in exhaustive_records}
        results_complete = (
            len(exhaustive_records) == EXPECTED_LAYOUT_COUNT
            and len(unique_ids) == EXPECTED_LAYOUT_COUNT
            and indices == set(range(EXPECTED_LAYOUT_COUNT))
        )
        results_comparable = bool(exhaustive_records) and settings_are_comparable(
            exhaustive_records[0].get("settings")
        )
        comparison["reference_results"] = str(results_path.resolve())
        comparison["reference_results_complete"] = results_complete
        comparison["reference_results_settings_comparable"] = results_comparable
        if not results_complete or not results_comparable:
            return comparison
        ranked = sorted(
            exhaustive_records,
            key=lambda record: topology_selection_key(record, target_accuracy),
        )
        ranks = {
            str(record["layout_id"]): rank
            for rank, record in enumerate(ranked, start=1)
        }
        comparison["winner_exhaustive_rank"] = ranks.get(str(winner["layout_id"]))
        comparison["exhaustive_layout_count"] = len(ranked)

        # Equal-budget control: the same a-priori K3 seed plus a uniform sample
        # of legal layouts. It is computed from exhaustive validation records
        # only after the GA winner is frozen, and never feeds back into search.
        k3_reference = next(
            (
                record
                for record in exhaustive_records
                if record.get("layout", {}).get("initial")
                == ["K3", DETECTOR_ID]
                and not record.get("layout", {}).get("specialized")
            ),
            None,
        )
        if k3_reference is not None and evaluated_layout_count >= 1:
            k3_id = str(k3_reference["layout_id"])
            remaining = [
                record
                for record in exhaustive_records
                if str(record["layout_id"]) != k3_id
            ]
            control_rng = np.random.default_rng(int(settings["outer_seed"]))
            control_count = min(evaluated_layout_count - 1, len(remaining))
            positions = control_rng.choice(
                len(remaining), size=control_count, replace=False
            )
            control_records = [k3_reference] + [
                remaining[int(position)] for position in positions
            ]
            control_best = min(
                control_records,
                key=lambda record: topology_selection_key(
                    record, target_accuracy
                ),
            )
            control_validation = control_best["validation"]
            assert isinstance(control_validation, Mapping)
            comparison["uniform_random_control"] = {
                "source": "post_search_exhaustive_validation_records",
                "outer_seed": int(settings["outer_seed"]),
                "evaluated_layouts": len(control_records),
                "best_layout_id": str(control_best["layout_id"]),
                "best_exhaustive_rank": ranks[str(control_best["layout_id"])],
                "best_validation_cost_ms": float(
                    control_validation["expected_cost"]
                ),
                "validation_cost_regret_ms": float(
                    control_validation["expected_cost"]
                )
                - float(reference_validation["expected_cost"]),
            }
    return comparison


def _validate_arguments(
    *,
    target_accuracy: float,
    iterations: int,
    quantile_points: int,
    holdout_fraction: float,
    population_size: int,
    generations: int,
    evaluation_budget: int,
    elite_count: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    random_immigrant_rate: float,
    workers: int,
) -> None:
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be between 0 and 1.")
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if quantile_points < 2:
        raise ValueError("quantile_points must be at least 2.")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be strictly between 0 and 1.")
    if population_size < 2:
        raise ValueError("population_size must be at least 2.")
    if generations < 1:
        raise ValueError("generations must be at least 1.")
    if not population_size <= evaluation_budget <= EXPECTED_LAYOUT_COUNT:
        raise ValueError(
            "evaluation_budget must be at least population_size and no larger "
            f"than {EXPECTED_LAYOUT_COUNT:,}."
        )
    if not 1 <= elite_count < population_size:
        raise ValueError("elite_count must be in [1, population_size).")
    if tournament_size < 2:
        raise ValueError("tournament_size must be at least 2.")
    for name, value in (
        ("crossover_rate", crossover_rate),
        ("mutation_rate", mutation_rate),
        ("random_immigrant_rate", random_immigrant_rate),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")


def _search_settings(
    *,
    outcomes: Path,
    target_accuracy: float,
    iterations: int,
    quantile_points: int,
    inner_seed: int,
    split_seed: int,
    outer_seed: int,
    holdout_fraction: float,
    split_strategy: str,
    population_size: int,
    generations: int,
    evaluation_budget: int,
    elite_count: int,
    tournament_size: int,
    crossover_rate: float,
    mutation_rate: float,
    random_immigrant_rate: float,
    component_resample_rate: float,
    stagnation_generations: int,
    max_restarts: int,
    catalogue_sha256: str,
    annealed_outer_schedule: bool,
) -> dict[str, object]:
    return {
        "algorithm": (
            "annealed_constrained_memetic_genetic_algorithm"
            if annealed_outer_schedule
            else "constrained_memetic_genetic_algorithm"
        ),
        "fitness_cache_schema": 1,
        "fitness_implementation_sha256": _fitness_implementation_sha256(),
        "layout_catalogue_sha256": catalogue_sha256,
        "outcomes": str(outcomes.resolve()),
        "outcomes_sha256": _file_sha256(outcomes),
        "removed_candidates": sorted(REMOVED_CANDIDATES),
        "detector_mode": "paper",
        "detector_cost_ms": float(PAPER_DETECTOR_COST_MS),
        "target_accuracy": float(target_accuracy),
        "target_accuracy_source": (
            "fig1_K3_validation_baseline"
            if target_accuracy == FIG1_K3_TARGET_ACCURACY
            else "explicit"
        ),
        "iterations": int(iterations),
        "quantile_points": int(quantile_points),
        "inner_seed": int(inner_seed),
        "split_seed": int(split_seed),
        "outer_seed": int(outer_seed),
        "holdout_fraction": float(holdout_fraction),
        "split_strategy": split_strategy,
        "population_size": int(population_size),
        "generations": int(generations),
        "evaluation_budget": int(evaluation_budget),
        "elite_count": int(elite_count),
        "tournament_size": int(tournament_size),
        "crossover_rate": float(crossover_rate),
        "mutation_rate": float(mutation_rate),
        "random_immigrant_rate": float(random_immigrant_rate),
        "component_resample_rate": float(component_resample_rate),
        "stagnation_generations": int(stagnation_generations),
        "max_restarts": int(max_restarts),
        "outer_parameter_schedule": (
            "linear_annealed" if annealed_outer_schedule else "fixed"
        ),
        "annealed_outer_schedule": (
            ANNEALED_OUTER_SCHEDULE if annealed_outer_schedule else None
        ),
    }


def run_joint_search(
    *,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path | None = None,
    target_accuracy: float = FIG1_K3_TARGET_ACCURACY,
    iterations: int = DEFAULT_ITERATIONS,
    quantile_points: int = DEFAULT_QUANTILE_POINTS,
    inner_seed: int = DEFAULT_SEED,
    split_seed: int = DEFAULT_SEED,
    outer_seed: int = DEFAULT_SEED,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    split_strategy: str = DEFAULT_SPLIT_STRATEGY,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    evaluation_budget: int = DEFAULT_EVALUATION_BUDGET,
    elite_count: int = DEFAULT_ELITE_COUNT,
    tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    random_immigrant_rate: float = DEFAULT_RANDOM_IMMIGRANT_RATE,
    component_resample_rate: float = DEFAULT_COMPONENT_RESAMPLE_RATE,
    stagnation_generations: int = DEFAULT_STAGNATION_GENERATIONS,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    annealed_outer_schedule: bool = False,
    workers: int = 1,
    overwrite: bool = False,
    brute_force_summary: Path | None = DEFAULT_BRUTE_FORCE_SUMMARY,
    brute_force_results: Path | None = DEFAULT_BRUTE_FORCE_RESULTS,
) -> dict[str, object]:
    """Run or resume the memetic GA and evaluate its winner on holdout."""

    if output_dir is None:
        output_dir = (
            DEFAULT_ANNEALED_OUTPUT_DIR
            if annealed_outer_schedule
            else DEFAULT_OUTPUT_DIR
        )

    _validate_arguments(
        target_accuracy=target_accuracy,
        iterations=iterations,
        quantile_points=quantile_points,
        holdout_fraction=holdout_fraction,
        population_size=population_size,
        generations=generations,
        evaluation_budget=evaluation_budget,
        elite_count=elite_count,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        random_immigrant_rate=random_immigrant_rate,
        workers=workers,
    )
    if not 0.0 <= component_resample_rate <= 1.0:
        raise ValueError("component_resample_rate must be between 0 and 1.")
    if stagnation_generations < 1:
        raise ValueError("stagnation_generations must be at least 1.")
    if max_restarts < 0:
        raise ValueError("max_restarts cannot be negative.")
    if annealed_outer_schedule:
        maximum_elites = int(
            max(ANNEALED_OUTER_SCHEDULE["elite_count"].values())
        )
        if population_size <= maximum_elites:
            raise ValueError(
                "population_size must exceed the annealed schedule's maximum "
                f"elite count ({maximum_elites})."
            )
    catalogue = build_layout_catalogue()
    settings = _search_settings(
        outcomes=outcomes,
        target_accuracy=target_accuracy,
        iterations=iterations,
        quantile_points=quantile_points,
        inner_seed=inner_seed,
        split_seed=split_seed,
        outer_seed=outer_seed,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        population_size=population_size,
        generations=generations,
        evaluation_budget=evaluation_budget,
        elite_count=elite_count,
        tournament_size=tournament_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        random_immigrant_rate=random_immigrant_rate,
        component_resample_rate=component_resample_rate,
        stagnation_generations=stagnation_generations,
        max_restarts=max_restarts,
        catalogue_sha256=_catalogue_sha256(catalogue),
        annealed_outer_schedule=annealed_outer_schedule,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "evaluations.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    summary_path = output_dir / "summary.json"
    if overwrite:
        for path in (results_path, checkpoint_path, summary_path):
            path.unlink(missing_ok=True)

    records = _load_jsonl(results_path)
    for record in records.values():
        if not _settings_match(record.get("settings"), settings):
            raise ValueError(
                f"{results_path} contains a different experiment. Use another "
                "output directory or pass --overwrite."
            )

    payload = _without_candidates(
        load_empirical_outcomes(outcomes), REMOVED_CANDIDATES
    )
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
    evaluate = InnerAnnealingFitness(
        validation_optimizer,
        target_accuracy=target_accuracy,
        quantile_points=quantile_points,
        iterations=iterations,
        inner_seed=inner_seed,
        settings=settings,
    )

    rng = np.random.default_rng(outer_seed)
    history: list[dict[str, object]] = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not _settings_match(checkpoint.get("settings"), settings):
            raise ValueError(
                f"{checkpoint_path} belongs to a different experiment. Use "
                "another output directory or pass --overwrite."
            )
        if checkpoint.get("status") == "complete" and summary_path.exists():
            completed_summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
            completed_winner = completed_summary.get("winner")
            if isinstance(completed_winner, Mapping):
                # Reporting inputs are intentionally outside the fitness-cache
                # settings. Refresh them without rerunning any inner anneals.
                completed_summary["exhaustive_comparison"] = (
                    compare_with_exhaustive(
                        completed_winner,
                        target_accuracy=target_accuracy,
                        settings=settings,
                        evaluated_layout_count=int(
                            completed_summary.get(
                                "unique_layouts_evaluated", len(records)
                            )
                        ),
                        summary_path=brute_force_summary,
                        results_path=brute_force_results,
                    )
                )
                _write_json_atomic(summary_path, completed_summary)
            return completed_summary
        generation = int(checkpoint["next_generation"])
        population = [str(item) for item in checkpoint["next_population"]]
        history = list(checkpoint.get("history", []))
        stagnant_generations = int(checkpoint.get("stagnant_generations", 0))
        restart_count = int(checkpoint.get("restart_count", 0))
        rng.bit_generator.state = checkpoint["rng_state"]
    else:
        population = initial_population(
            catalogue,
            population_size,
            rng,
        )
        generation = 0
        stagnant_generations = 0
        restart_count = 0
        _write_json_atomic(
            checkpoint_path,
            {
                "status": "running",
                "settings": settings,
                "next_generation": generation,
                "next_population": population,
                "rng_state": rng.bit_generator.state,
                "history": history,
                "stagnant_generations": stagnant_generations,
                "restart_count": restart_count,
            },
        )

    print(
        f"Memetic GA: population={population_size}, generations<={generations}, "
        f"unique-layout budget={evaluation_budget}, workers={workers}"
    )
    print(
        "Inner fitness: K1-free, paper Kdet=10000 ms, "
        f"{iterations:,}-step SA, {quantile_points} quantiles, inner seed={inner_seed}"
    )
    print(
        "Outer parameter schedule: "
        f"{'linear annealed' if annealed_outer_schedule else 'fixed'}"
    )
    started = perf_counter()
    stop_reason = "generation_limit"

    while generation < generations:
        remaining_budget = evaluation_budget - len(records)
        missing = [item for item in population if item not in records]
        if len(missing) > remaining_budget:
            # This can occur only with a custom budget that does not align to
            # population/elitism. Keep the best cached members and a prefix of
            # unseen proposals; no topology is evaluated beyond the cap.
            cached = [item for item in population if item in records]
            population = [*cached, *missing[:remaining_budget]]

        new_evaluations = _evaluate_missing(
            population,
            records,
            catalogue,
            evaluate,
            results_path,
            workers=workers,
        )
        if not population:
            break
        item = _history_item(
            generation, records, target_accuracy, new_evaluations
        )
        previous_best_id = (
            str(history[-1]["best_layout_id"]) if history else None
        )
        if previous_best_id is None or previous_best_id != item["best_layout_id"]:
            stagnant_generations = 0
        else:
            stagnant_generations += 1
        item["stagnant_generations"] = stagnant_generations
        item["restart_after_generation"] = False
        # These fields describe breeding for the *next* generation. They stay
        # null on the terminal history item, where no parameters are applied.
        item["next_population_schedule_progress"] = None
        item["next_population_parameters"] = None
        item["next_population_strategy"] = None
        history.append(item)
        print(
            f"Generation {generation:02d}: unique={len(records):,}, "
            f"best={item['best_validation_cost_ms']:.3f} ms at "
            f"accuracy={item['best_validation_accuracy']:.6f}"
        )

        if len(records) >= evaluation_budget:
            stop_reason = "evaluation_budget"
            break
        if generation + 1 >= generations:
            stop_reason = "generation_limit"
            break

        schedule_progress = min(1.0, len(records) / evaluation_budget)
        generation_parameters = outer_ga_parameters(
            schedule_progress,
            annealed=annealed_outer_schedule,
            elite_count=elite_count,
            tournament_size=tournament_size,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            random_immigrant_rate=random_immigrant_rate,
            component_resample_rate=component_resample_rate,
        )
        item["next_population_schedule_progress"] = schedule_progress
        remaining_unique = evaluation_budget - len(records)
        if (
            stagnant_generations >= stagnation_generations
            and restart_count < max_restarts
        ):
            # A restart retains one cached global elite, whereas ordinary
            # breeding retains the scheduled number of cached elites.
            desired_size = min(population_size, 1 + remaining_unique)
            population = restart_population(
                records,
                catalogue,
                rng,
                target_accuracy=target_accuracy,
                population_size=desired_size,
            )
            restart_count += 1
            stagnant_generations = 0
            item["restart_after_generation"] = True
            item["next_population_strategy"] = "restart"
        else:
            desired_size = min(
                population_size,
                generation_parameters.elite_count + remaining_unique,
            )
            population = next_population(
                population,
                records,
                catalogue,
                rng,
                target_accuracy=target_accuracy,
                population_size=desired_size,
                elite_count=min(
                    generation_parameters.elite_count,
                    desired_size - 1,
                ),
                tournament_size=generation_parameters.tournament_size,
                crossover_rate=generation_parameters.crossover_rate,
                mutation_rate=generation_parameters.mutation_rate,
                random_immigrant_rate=(
                    generation_parameters.random_immigrant_rate
                ),
                component_resample_rate=(
                    generation_parameters.component_resample_rate
                ),
                excluded_layout_ids=set(records),
            )
            item["next_population_strategy"] = "breed"
            item["next_population_parameters"] = (
                generation_parameters.as_dict()
            )
        generation += 1
        _write_json_atomic(
            checkpoint_path,
            {
                "status": "running",
                "settings": settings,
                "next_generation": generation,
                "next_population": population,
                "rng_state": rng.bit_generator.state,
                "history": history,
                "stagnant_generations": stagnant_generations,
                "restart_count": restart_count,
            },
        )

    winner = dict(_best_record(records, target_accuracy))
    # Constructing the holdout evaluator is deliberately deferred until the
    # validation-only search has frozen its winner.
    holdout_optimizer = HierarchyOptimizer(
        holdout_payload,
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )
    holdout = _final_holdout(
        winner, holdout_optimizer, catalogue, target_accuracy
    )
    winner["holdout"] = holdout
    winner["holdout_feasible"] = bool(holdout["feasible"])

    # Ground truth is deliberately consulted only now, after validation has
    # selected and frozen the GA winner.
    exhaustive_comparison = compare_with_exhaustive(
        winner,
        target_accuracy=target_accuracy,
        settings=settings,
        evaluated_layout_count=len(records),
        summary_path=brute_force_summary,
        results_path=brute_force_results,
    )
    elapsed = perf_counter() - started
    summary: dict[str, object] = {
        "settings": settings,
        "split": split,
        "stop_reason": stop_reason,
        "generations_completed": len(history),
        "restart_count": restart_count,
        "unique_layouts_evaluated": len(records),
        "layout_space_size": EXPECTED_LAYOUT_COUNT,
        "fraction_of_layout_space": len(records) / EXPECTED_LAYOUT_COUNT,
        "elapsed_seconds_this_invocation": elapsed,
        "winner": winner,
        "history": history,
        "pareto_archive": _pareto_archive(records),
        "exhaustive_comparison": exhaustive_comparison,
        "holdout_usage": "winner_only_after_validation_search",
    }
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(
        checkpoint_path,
        {
            "status": "complete",
            "settings": settings,
            "next_generation": generation,
            "next_population": population,
            "rng_state": rng.bit_generator.state,
            "history": history,
            "stagnant_generations": stagnant_generations,
            "restart_count": restart_count,
        },
    )
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Joint K1-free layout/threshold optimization with a memetic GA "
            "and brute-force-compatible inner annealing."
        )
    )
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory. Defaults to separate fixed/annealed "
            "directories based on --annealed-outer-schedule."
        ),
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=FIG1_K3_TARGET_ACCURACY,
        help="Hard validation constraint; defaults to the Fig. 1 K3 baseline.",
    )
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS
    )
    parser.add_argument("--inner-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--outer-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION
    )
    parser.add_argument(
        "--split-strategy",
        choices=("blocked_per_run", "random_per_run"),
        default=DEFAULT_SPLIT_STRATEGY,
    )
    parser.add_argument(
        "--population-size", type=int, default=DEFAULT_POPULATION_SIZE
    )
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument(
        "--evaluation-budget", type=int, default=DEFAULT_EVALUATION_BUDGET
    )
    parser.add_argument("--elite-count", type=int, default=DEFAULT_ELITE_COUNT)
    parser.add_argument(
        "--tournament-size", type=int, default=DEFAULT_TOURNAMENT_SIZE
    )
    parser.add_argument(
        "--crossover-rate", type=float, default=DEFAULT_CROSSOVER_RATE
    )
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE)
    parser.add_argument(
        "--random-immigrant-rate",
        type=float,
        default=DEFAULT_RANDOM_IMMIGRANT_RATE,
    )
    parser.add_argument(
        "--component-resample-rate",
        type=float,
        default=DEFAULT_COMPONENT_RESAMPLE_RATE,
        help="Share of mutations that redraw a whole trunk/branch component.",
    )
    parser.add_argument(
        "--stagnation-generations",
        type=int,
        default=DEFAULT_STAGNATION_GENERATIONS,
        help="Restart after this many generations without a new global best.",
    )
    parser.add_argument("--max-restarts", type=int, default=DEFAULT_MAX_RESTARTS)
    parser.add_argument(
        "--annealed-outer-schedule",
        action="store_true",
        help=(
            "Linearly shift the outer GA from broad exploration to stronger "
            "selection/local refinement. The inner 8k threshold SA is unchanged."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Concurrent layout evaluations within a generation. Threaded "
            "workers share the read-only empirical table."
        ),
    )
    parser.add_argument(
        "--brute-force-summary",
        type=Path,
        default=DEFAULT_BRUTE_FORCE_SUMMARY,
        help="Optional post-search ground-truth comparison; never used as fitness.",
    )
    parser.add_argument(
        "--brute-force-results",
        type=Path,
        default=DEFAULT_BRUTE_FORCE_RESULTS,
        help="Optional post-search file used to report the winner's exact rank.",
    )
    parser.add_argument(
        "--no-brute-force-comparison",
        action="store_true",
        help="Do not read exhaustive outputs, even after search.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing GA checkpoint and evaluation cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the budget/runtime estimate without loading outcomes.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.dry_run:
        maximum_from_generations = args.population_size + max(
            0, args.generations - 1
        ) * (args.population_size - 1)
        expected_evaluations = min(args.evaluation_budget, maximum_from_generations)
        iteration_scale = args.iterations / DEFAULT_ITERATIONS
        sequential_seconds = (
            expected_evaluations
            * OBSERVED_SECONDS_PER_LAYOUT
            * iteration_scale
        )
        print(f"Legal K1-free layouts: {EXPECTED_LAYOUT_COUNT:,}")
        print(
            f"Planned unique evaluations: up to {expected_evaluations:,} "
            f"({expected_evaluations / EXPECTED_LAYOUT_COUNT:.2%} of the space)"
        )
        print(
            "Outer parameter schedule: "
            f"{'linear annealed' if args.annealed_outer_schedule else 'fixed'}"
        )
        print(
            f"Observed sequential estimate: {sequential_seconds / 60.0:.1f} min "
            f"({OBSERVED_SECONDS_PER_LAYOUT:.3f} s/layout at 8k SA, scaled "
            f"linearly to {args.iterations:,} iterations)"
        )
        if args.workers > 1:
            print(
                f"Idealized {args.workers}-worker lower bound: "
                f"{sequential_seconds / args.workers / 60.0:.1f} min; actual "
                "time depends on CPU/memory contention and generation barriers."
            )
        return

    comparison_summary = (
        None if args.no_brute_force_comparison else args.brute_force_summary
    )
    comparison_results = (
        None if args.no_brute_force_comparison else args.brute_force_results
    )
    output_dir = args.output_dir or (
        DEFAULT_ANNEALED_OUTPUT_DIR
        if args.annealed_outer_schedule
        else DEFAULT_OUTPUT_DIR
    )
    summary = run_joint_search(
        outcomes=args.outcomes,
        output_dir=output_dir,
        target_accuracy=args.target_accuracy,
        iterations=args.iterations,
        quantile_points=args.quantile_points,
        inner_seed=args.inner_seed,
        split_seed=args.split_seed,
        outer_seed=args.outer_seed,
        holdout_fraction=args.holdout_fraction,
        split_strategy=args.split_strategy,
        population_size=args.population_size,
        generations=args.generations,
        evaluation_budget=args.evaluation_budget,
        elite_count=args.elite_count,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        random_immigrant_rate=args.random_immigrant_rate,
        component_resample_rate=args.component_resample_rate,
        stagnation_generations=args.stagnation_generations,
        max_restarts=args.max_restarts,
        annealed_outer_schedule=args.annealed_outer_schedule,
        workers=args.workers,
        overwrite=args.overwrite,
        brute_force_summary=comparison_summary,
        brute_force_results=comparison_results,
    )
    winner = summary["winner"]
    validation = winner["validation"]
    holdout = winner["holdout"]
    print(
        f"Winner {winner['layout_id']}: validation "
        f"{validation['accuracy']:.6f} / {validation['expected_cost']:.3f} ms; "
        f"holdout {holdout['accuracy']:.6f} / {holdout['expected_cost']:.3f} ms"
    )
    comparison = summary.get("exhaustive_comparison")
    if isinstance(comparison, Mapping) and comparison.get(
        "comparison_available"
    ):
        print(
            f"Exhaustive rank={comparison.get('winner_exhaustive_rank')}; "
            f"validation regret={comparison['validation_cost_regret_ms']:.3f} ms; "
            f"exact winner={comparison['exact_layout_recovered']}"
        )
    elif isinstance(comparison, Mapping):
        print(
            "Exhaustive comparison unavailable: "
            f"{comparison.get('unavailable_reasons', 'incomplete results file')}"
        )


if __name__ == "__main__":
    main()
