"""Generic depth-one hierarchy genomes and evolutionary operators."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from cascade_profile import HierarchyProfile
from hierarchy_optimizer import Cascade


BranchKey = tuple[str, str]


@dataclass(frozen=True)
class TopologyGenome:
    """Hashable variable-module genome for a depth-one hierarchy."""

    initial: tuple[str, ...]
    branches: tuple[tuple[str, str, tuple[str, ...]], ...] = ()

    @property
    def branch_map(self) -> dict[BranchKey, tuple[str, ...]]:
        return {
            (router_id, group): chain
            for router_id, group, chain in self.branches
        }


@dataclass(frozen=True)
class LayoutSpace:
    """Legal classifier grammar derived entirely from an outcome profile."""

    profile: HierarchyProfile
    global_ids: tuple[str, ...]
    router_ids: tuple[str, ...]
    specialized_by_group: Mapping[str, tuple[str, ...]]
    detector_id: str

    @classmethod
    def from_candidates(
        cls,
        profile: HierarchyProfile,
        candidates: pd.DataFrame,
        detector_id: str,
    ) -> "LayoutSpace":
        indexed = candidates.set_index("id", drop=False)
        return cls(
            profile=profile,
            global_ids=tuple(
                str(item) for item in indexed[indexed["kind"] == "global"].index
            ),
            router_ids=tuple(
                str(item)
                for item in indexed[indexed["kind"] == "identifier"].index
            ),
            specialized_by_group={
                group: tuple(
                    str(item)
                    for item in indexed[
                        (indexed["kind"] == "specialized")
                        & (indexed["group"] == group)
                    ].index
                )
                for group in profile.group_ids
            },
            detector_id=str(detector_id),
        )

    @property
    def initial_ids(self) -> tuple[str, ...]:
        return (*self.router_ids, *self.global_ids)

    def allowed_branch_ids(
        self, initial: Sequence[str], router_id: str, group: str
    ) -> tuple[str, ...]:
        router_position = tuple(initial).index(router_id)
        already_evaluated = set(initial[:router_position]) & set(self.global_ids)
        return tuple(
            candidate_id
            for candidate_id in (
                *self.global_ids,
                *self.specialized_by_group.get(group, ()),
            )
            if candidate_id not in already_evaluated
        )


def _deduplicate_allowed(
    sequence: Sequence[str], allowed: Sequence[str]
) -> tuple[str, ...]:
    allowed_set = set(allowed)
    return tuple(dict.fromkeys(item for item in sequence if item in allowed_set))


def repair_genome(genome: TopologyGenome, space: LayoutSpace) -> TopologyGenome:
    """Project arbitrary edits onto the dynamic depth-one grammar."""

    initial = _deduplicate_allowed(genome.initial, space.initial_ids)
    supplied = genome.branch_map
    branches: list[tuple[str, str, tuple[str, ...]]] = []
    for router_id in initial:
        if router_id not in space.router_ids:
            continue
        for group in space.profile.group_ids:
            allowed = space.allowed_branch_ids(initial, router_id, group)
            chain = _deduplicate_allowed(supplied.get((router_id, group), ()), allowed)
            branches.append((router_id, group, chain))
    return TopologyGenome(initial=initial, branches=tuple(branches))


def cascade_from_genome(genome: TopologyGenome, space: LayoutSpace) -> Cascade:
    canonical = repair_genome(genome, space)
    return Cascade(
        expected_cost=0.0,
        initial=[*canonical.initial, space.detector_id],
        specialized={
            (router_id, group): [*chain, space.detector_id]
            for router_id, group, chain in canonical.branches
        },
        detector=space.detector_id,
    )


def genome_from_cascade(cascade: Cascade, space: LayoutSpace) -> TopologyGenome:
    initial = tuple(
        item for item in cascade.initial if item != cascade.detector
    )
    branches = tuple(
        (
            router_id,
            group,
            tuple(item for item in chain if item != cascade.detector),
        )
        for (router_id, group), chain in cascade.specialized.items()
    )
    return repair_genome(TopologyGenome(initial, branches), space)


def layout_id(genome: TopologyGenome, space: LayoutSpace) -> str:
    canonical = repair_genome(genome, space)
    encoded = json.dumps(
        {
            "initial": canonical.initial,
            "branches": canonical.branches,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _random_ordered_subset(
    allowed: Sequence[str], rng: np.random.Generator
) -> tuple[str, ...]:
    length = int(rng.integers(0, len(allowed) + 1))
    if not length:
        return ()
    values = rng.choice(np.asarray(allowed, dtype=object), size=length, replace=False)
    return tuple(str(item) for item in values)


def random_genome(space: LayoutSpace, rng: np.random.Generator) -> TopologyGenome:
    initial = _random_ordered_subset(space.initial_ids, rng)
    branches: list[tuple[str, str, tuple[str, ...]]] = []
    for router_id in initial:
        if router_id not in space.router_ids:
            continue
        for group in space.profile.group_ids:
            allowed = space.allowed_branch_ids(initial, router_id, group)
            branches.append(
                (router_id, group, _random_ordered_subset(allowed, rng))
            )
    return repair_genome(TopologyGenome(initial, tuple(branches)), space)


def _edit_sequence(
    sequence: Sequence[str], allowed: Sequence[str], rng: np.random.Generator
) -> tuple[str, ...]:
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
    if not operations:
        return tuple(values)
    operation = str(rng.choice(operations))
    if operation == "insert":
        value = str(rng.choice(missing))
        values.insert(int(rng.integers(0, len(values) + 1)), value)
    elif operation == "delete":
        del values[int(rng.integers(0, len(values)))]
    elif operation == "swap":
        first, second = rng.choice(len(values), size=2, replace=False)
        values[int(first)], values[int(second)] = values[int(second)], values[int(first)]
    elif operation == "relocate":
        value = values.pop(int(rng.integers(0, len(values))))
        values.insert(int(rng.integers(0, len(values) + 1)), value)
    else:
        values[int(rng.integers(0, len(values)))] = str(rng.choice(missing))
    return tuple(values)


def mutate_genome(
    genome: TopologyGenome,
    space: LayoutSpace,
    rng: np.random.Generator,
    *,
    component_resample_rate: float = 0.30,
    initial_component_weight: float = 0.40,
) -> TopologyGenome:
    child = repair_genome(genome, space)
    branch_keys = [
        (router_id, group) for router_id, group, _ in child.branches
    ]
    if not branch_keys:
        component: str | BranchKey = "initial"
    else:
        branch_weight = (1.0 - initial_component_weight) / len(branch_keys)
        components: list[str | BranchKey] = ["initial", *branch_keys]
        weights = [initial_component_weight, *([branch_weight] * len(branch_keys))]
        component = components[int(rng.choice(len(components), p=weights))]

    resample = rng.random() < component_resample_rate
    if component == "initial":
        edited = (
            _random_ordered_subset(space.initial_ids, rng)
            if resample
            else _edit_sequence(child.initial, space.initial_ids, rng)
        )
        return repair_genome(TopologyGenome(edited, child.branches), space)

    router_id, group = component
    allowed = space.allowed_branch_ids(child.initial, router_id, group)
    branches = child.branch_map
    current = branches[(router_id, group)]
    branches[(router_id, group)] = (
        _random_ordered_subset(allowed, rng)
        if resample
        else _edit_sequence(current, allowed, rng)
    )
    return repair_genome(
        TopologyGenome(
            child.initial,
            tuple(
                (branch_router, branch_group, chain)
                for (branch_router, branch_group), chain in branches.items()
            ),
        ),
        space,
    )


def _recombine_chain(
    first: tuple[str, ...],
    second: tuple[str, ...],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    if rng.random() < 0.70:
        return first if rng.random() < 0.5 else second
    primary, secondary = (first, second) if rng.random() < 0.5 else (second, first)
    included: set[str] = set()
    for item in dict.fromkeys((*primary, *secondary)):
        if (item in primary and item in secondary) or rng.random() < 0.5:
            included.add(item)
    return tuple(
        item for item in dict.fromkeys((*primary, *secondary)) if item in included
    )


def crossover_genomes(
    first: TopologyGenome,
    second: TopologyGenome,
    space: LayoutSpace,
    rng: np.random.Generator,
) -> TopologyGenome:
    first = repair_genome(first, space)
    second = repair_genome(second, space)
    initial = _recombine_chain(first.initial, second.initial, rng)
    first_branches = first.branch_map
    second_branches = second.branch_map
    branches = tuple(
        (
            router_id,
            group,
            _recombine_chain(
                first_branches.get((router_id, group), ()),
                second_branches.get((router_id, group), ()),
                rng,
            ),
        )
        for router_id in space.router_ids
        for group in space.profile.group_ids
    )
    return repair_genome(TopologyGenome(initial, branches), space)
