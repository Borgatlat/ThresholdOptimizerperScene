"""Dataset-neutral memetic search over depth-one hierarchy layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from layout_search import (
    LayoutSpace,
    TopologyGenome,
    crossover_genomes,
    layout_id,
    mutate_genome,
    random_genome,
    repair_genome,
)


FitnessRecord = Mapping[str, object]
FitnessFunction = Callable[[TopologyGenome], FitnessRecord]


@dataclass(frozen=True)
class MemeticSearchConfig:
    population_size: int = 32
    generations: int = 24
    evaluation_budget: int = 512
    elite_count: int = 4
    tournament_size: int = 2
    crossover_rate: float = 0.80
    mutation_rate: float = 0.80
    random_immigrant_rate: float = 0.20
    component_resample_rate: float = 0.30
    random_seed: int = 0
    allow_cached_reentry: bool = True


@dataclass(frozen=True)
class MemeticSearchResult:
    best_genome: TopologyGenome
    best_record: FitnessRecord
    records: Mapping[str, FitnessRecord]
    genomes: Mapping[str, TopologyGenome]
    generations_completed: int


def topology_selection_key(
    record: FitnessRecord, target_accuracy: float
) -> tuple[float, ...]:
    validation = record.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("Fitness records require validation metrics.")
    accuracy = float(validation["accuracy"])
    cost = float(validation["expected_cost"])
    layout_index = float(record.get("layout_index", 0))
    if accuracy >= target_accuracy:
        return (0.0, cost, -accuracy, layout_index)
    return (1.0, -accuracy, cost, layout_index)


def _tournament_parent(
    population: Sequence[str],
    records: Mapping[str, FitnessRecord],
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


def create_starting_population(
    space: LayoutSpace,
    population_size: int,
    rng: np.random.Generator,
    *,
    seeds: Sequence[TopologyGenome] = (),
) -> tuple[list[str], dict[str, TopologyGenome]]:
    """Create a unique population without requiring an enumerable catalogue."""

    selected: list[str] = []
    genomes: dict[str, TopologyGenome] = {}

    def add(genome: TopologyGenome) -> None:
        canonical = repair_genome(genome, space)
        candidate_id = layout_id(canonical, space)
        if candidate_id not in genomes:
            selected.append(candidate_id)
            genomes[candidate_id] = canonical

    for seed in seeds:
        if len(selected) == population_size:
            break
        add(seed)

    attempts = 0
    max_attempts = max(1_000, population_size * 500)
    while len(selected) < population_size and attempts < max_attempts:
        attempts += 1
        add(random_genome(space, rng))
    if len(selected) != population_size:
        raise RuntimeError("Could not construct a full unique starting population.")
    return selected, genomes


def run_memetic_search(
    space: LayoutSpace,
    evaluate: FitnessFunction,
    target_accuracy: float,
    *,
    config: MemeticSearchConfig = MemeticSearchConfig(),
    seeds: Sequence[TopologyGenome] = (),
) -> MemeticSearchResult:
    """Run a generic unique-population layout search with cached fitness."""

    rng = np.random.default_rng(config.random_seed)
    population, genomes = create_starting_population(
        space, config.population_size, rng, seeds=seeds
    )
    records: dict[str, FitnessRecord] = {}
    generations_completed = 0

    for generation in range(config.generations):
        pending_ids = [
            candidate_id
            for candidate_id in population
            if candidate_id not in records
        ][: max(0, config.evaluation_budget - len(records))]
        evaluate_many = getattr(evaluate, "evaluate_many", None)
        if callable(evaluate_many) and pending_ids:
            batch = evaluate_many([genomes[candidate_id] for candidate_id in pending_ids])
            if len(batch) != len(pending_ids):
                raise RuntimeError("Batch fitness returned the wrong number of records.")
            for candidate_id, item in zip(pending_ids, batch, strict=True):
                record = dict(item)
                record.setdefault("layout_id", candidate_id)
                records[candidate_id] = record
        else:
            for candidate_id in pending_ids:
                record = dict(evaluate(genomes[candidate_id]))
                record.setdefault("layout_id", candidate_id)
                records[candidate_id] = record
        evaluated_population = [item for item in population if item in records]
        if not evaluated_population:
            break
        generations_completed = generation + 1
        if len(records) >= config.evaluation_budget:
            break

        ranked = sorted(
            evaluated_population,
            key=lambda item: topology_selection_key(records[item], target_accuracy),
        )
        desired_size = config.population_size
        next_population = ranked[: min(config.elite_count, desired_size)]
        next_set = set(next_population)
        attempts = 0
        max_attempts = max(2_000, desired_size * 500)
        while len(next_population) < desired_size and attempts < max_attempts:
            attempts += 1
            if rng.random() < config.random_immigrant_rate:
                child = random_genome(space, rng)
            else:
                first_id = _tournament_parent(
                    evaluated_population,
                    records,
                    target_accuracy,
                    config.tournament_size,
                    rng,
                )
                child = genomes[first_id]
                if rng.random() < config.crossover_rate:
                    second_id = _tournament_parent(
                        evaluated_population,
                        records,
                        target_accuracy,
                        config.tournament_size,
                        rng,
                    )
                    child = crossover_genomes(
                        child, genomes[second_id], space, rng
                    )
                if rng.random() < config.mutation_rate:
                    child = mutate_genome(
                        child,
                        space,
                        rng,
                        component_resample_rate=config.component_resample_rate,
                    )
            child = repair_genome(child, space)
            child_id = layout_id(child, space)
            if child_id in next_set:
                continue
            if child_id in records and not config.allow_cached_reentry:
                continue
            genomes.setdefault(child_id, child)
            next_population.append(child_id)
            next_set.add(child_id)
        if len(next_population) != desired_size:
            raise RuntimeError("Could not breed a full unique population.")
        population = next_population

    best_id = min(
        records,
        key=lambda item: topology_selection_key(records[item], target_accuracy),
    )
    return MemeticSearchResult(
        best_genome=genomes[best_id],
        best_record=records[best_id],
        records=records,
        genomes=genomes,
        generations_completed=generations_completed,
    )
