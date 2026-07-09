"""Hierarchical IDK cascade optimizer -- faithful implementation of Algorithm 1
(EXPAND / EXPAND') from "Timely Classification of Hierarchical Classes"
(Baruah, Burns, Abdelzaher, Hu; RTSS 2025).

This consumes the payload produced by cascade/empirical_outcomes.py: every
classifier's accept/reject + prediction on the SAME shared set of samples,
so probabilities are computed by masking the joint per-sample outcome
arrays (Section III-B's measurement-based methodology) rather than assuming
classifier independence. Two samples where K0 and K1 both fail together
(e.g. a shared confounder like weather) are captured correctly; an
independence-assuming optimizer would miss that correlation.

Mapping from the paper to this code:
  EXPAND(S)                  -> ExpandTable indexed by `rejected_initial`
  EXPAND'(S, I_l, T, K_h)     -> ExpandPrimeTable indexed by
                                 (rejected_initial, group, rejected_specialized, router)
  K_phi (global classifiers)  -> candidates with kind == "global"
  K_I (identifier classifiers)-> candidates with kind == "identifier"
  K_l (specialized classifiers)-> candidates with kind == "specialized", group == l
  K_det                       -> the `detector` entry in the payload

Kdet handling
-------------
The paper assumes Kdet is deterministic: it never IDKs and is always
correct (footnote 1: "a deterministic classifier is added as the final
stage... If this deterministic classifier also fails... the system
registers a fault"). The trained Kdet checkpoint in this repo is NOT
faithful to that assumption (p_correct ~0.94, cost not dramatically higher
than the other classifiers), which causes the DP to route to Kdet far more
often than the paper's model intends, starving the optimizer of reasons to
chain through more classifiers first.

`detector_mode="paper"` (default) replaces the trained Kdet with the
paper-faithful synthetic fallback you asked for: cost/wcet forced to a
large constant (default 10_000 ms) and treated as always-correct. This is
not a hack -- it is *removing* a deviation from the paper's own assumption,
not adding a new one. `detector_mode="trained"` keeps the registry's real
Kdet cost/behavior if you want to compare the two directly.

Note: this DP only optimizes for expected classification TIME (Section
IV-A/B). It does not yet fold in Kdet's accuracy or a hard deadline D
(Section IV-C) into the objective -- see optimize_empirical_hierarchy()
docstring for what synthesize() returns and what it does not account for.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from empirical_outcomes import DEFAULT_OUTPUT_PATH, load_empirical_outcomes

PAPER_DETECTOR_COST_MS = 10_000.0


@dataclass
class Cascade:
    expected_cost: float
    initial: list[str]
    specialized: dict[tuple[str, str], list[str]]  # (router_id, group) -> chain
    detector: str


class HierarchyOptimizer:
    """EXPAND / EXPAND' DP over a shared empirical outcome table.

    `rejected_initial` / `rejected_specialized` are always passed as sorted
    tuples so lru_cache sees identical keys for identical sets regardless of
    insertion order (Python tuples are order-sensitive, sets are not
    hashable -- sorted tuples give us both: hashable AND order-independent).
    """

    def __init__(
        self,
        payload: dict,
        detector_mode: str = "paper",
        detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
    ):
        self.candidates = payload["candidates"].set_index("id", drop=False)
        self.outcomes = payload["outcomes"]
        self.labels = payload["labels"]
        self.sample_count = len(self.labels)

        self.global_ids = tuple(self.candidates[self.candidates["kind"] == "global"].index)
        self.identifier_ids = tuple(self.candidates[self.candidates["kind"] == "identifier"].index)
        self.initial_ids = tuple(self.global_ids + self.identifier_ids)

        self.groups = tuple(
            sorted(g for g in self.candidates["group"].dropna().unique())
        )
        self.specialized_by_group = {
            group: tuple(
                self.candidates[
                    (self.candidates["kind"] == "specialized") & (self.candidates["group"] == group)
                ].index
            )
            for group in self.groups
        }

        # router (identifier) -> which raw prediction value means "this group"
        # K0/K1 predict into INTERMEDIATE_CLASS_NAMES order; group names here
        # ("suv", "coupe") must match that same shared label schema produced
        # by cascade/empirical_outcomes.py's _map_intermediate.
        from utils.labels import INTERMEDIATE_CLASS_NAMES

        self._group_to_intermediate_idx = {
            name: idx for idx, name in enumerate(INTERMEDIATE_CLASS_NAMES) if name in self.groups
        }

        self.accepted = {}
        self.prediction = {}
        for candidate_id, group_df in self.outcomes.groupby("candidate_id", sort=False):
            ordered = group_df.sort_values("sample_id")
            self.accepted[candidate_id] = ordered["accepted"].to_numpy(dtype=bool)
            self.prediction[candidate_id] = ordered["prediction"].to_numpy(dtype=int)

        self.detector_id = "detector"
        if detector_mode == "paper":
            # Paper's Kdet: deterministic, always correct, dominant fallback
            # cost. We don't have a per-sample "is Kdet correct" outcome
            # array because Kdet never IDKs in this dataset's logging
            # convention -- so we model it the same way the paper does:
            # a constant cost, p_idk = 0, and accuracy folded in separately
            # (see synthesize()'s docstring note on accuracy bookkeeping).
            self.detector_cost = float(detector_cost_ms)
        elif detector_mode == "trained":
            self.detector_cost = float(payload["detector"]["cost"])
        else:
            raise ValueError(f"Unknown detector_mode: {detector_mode!r}")

        self._expand_next: dict[tuple, str] = {}
        self._expand_prime_next: dict[tuple, str] = {}

        # lru_cache is per-instance-unsafe across different HierarchyOptimizer
        # objects sharing a class method unless bound per-instance; binding
        # here (rather than decorating the method directly) avoids cache
        # collisions if you build two optimizers (e.g. "paper" vs "trained"
        # detector_mode) in the same process.
        self.expand = lru_cache(maxsize=None)(self._expand_impl)
        self.expand_prime = lru_cache(maxsize=None)(self._expand_prime_impl)

    def _cost(self, candidate_id: str) -> float:
        return float(self.candidates.loc[candidate_id, "cost"])

    def _idk_mask(self, candidate_id: str) -> np.ndarray:
        return ~self.accepted[candidate_id]

    def _eligible_initial(self, rejected_initial: tuple[str, ...]) -> np.ndarray:
        mask = np.ones(self.sample_count, dtype=bool)
        for candidate_id in rejected_initial:
            mask &= self._idk_mask(candidate_id)
        return mask

    def _eligible_specialized(
        self,
        rejected_initial: tuple[str, ...],
        group: str,
        rejected_specialized: tuple[str, ...],
        router_id: str,
    ) -> np.ndarray:
        mask = self._eligible_initial(rejected_initial)
        mask &= self.accepted[router_id]
        mask &= self.prediction[router_id] == self._group_to_intermediate_idx[group]
        for candidate_id in rejected_specialized:
            mask &= self._idk_mask(candidate_id)
        return mask

    def _initial_probs(
        self, rejected_initial: tuple[str, ...], candidate_id: str
    ) -> tuple[float, dict[str, float]]:
        eligible = self._eligible_initial(rejected_initial)
        denominator = int(eligible.sum())
        if denominator == 0:
            return 1.0, {group: 0.0 for group in self.groups}

        idk_probability = float((eligible & self._idk_mask(candidate_id)).sum()) / denominator

        group_probabilities = {group: 0.0 for group in self.groups}
        if candidate_id in self.identifier_ids:
            for group in self.groups:
                group_idx = self._group_to_intermediate_idx[group]
                group_mask = (
                    eligible & self.accepted[candidate_id] & (self.prediction[candidate_id] == group_idx)
                )
                group_probabilities[group] = float(group_mask.sum()) / denominator

        return idk_probability, group_probabilities

    def _specialized_idk_prob(
        self,
        rejected_initial: tuple[str, ...],
        group: str,
        rejected_specialized: tuple[str, ...],
        router_id: str,
        candidate_id: str,
    ) -> float:
        eligible = self._eligible_specialized(rejected_initial, group, rejected_specialized, router_id)
        denominator = int(eligible.sum())
        if denominator == 0:
            return 1.0
        return float((eligible & self._idk_mask(candidate_id)).sum()) / denominator

    def _expand_impl(self, rejected_initial: tuple[str, ...]) -> float:
        rejected_set = set(rejected_initial)
        remaining = [c for c in self.initial_ids if c not in rejected_set]

        best_cost = self.detector_cost
        best_next = self.detector_id

        if remaining:
            for candidate_id in remaining:
                idk_probability, group_probabilities = self._initial_probs(rejected_initial, candidate_id)
                cost = self._cost(candidate_id) + idk_probability * self.expand(
                    tuple(sorted(rejected_set | {candidate_id}))
                )

                if candidate_id in self.identifier_ids:
                    for group, group_probability in group_probabilities.items():
                        if group_probability == 0.0:
                            continue
                        cost += group_probability * self.expand_prime(
                            rejected_initial, group, tuple(), candidate_id
                        )

                if cost < best_cost:
                    best_cost = cost
                    best_next = candidate_id

        self._expand_next[rejected_initial] = best_next
        return best_cost

    def _expand_prime_impl(
        self,
        rejected_initial: tuple[str, ...],
        group: str,
        rejected_specialized: tuple[str, ...],
        router_id: str,
    ) -> float:
        rejected_initial_set = set(rejected_initial)
        rejected_specialized_set = set(rejected_specialized)

        remaining_globals = [c for c in self.global_ids if c not in rejected_initial_set]
        remaining_specialized = [
            c for c in self.specialized_by_group.get(group, ()) if c not in rejected_specialized_set
        ]

        best_cost = self.detector_cost
        best_next = self.detector_id

        for candidate_id in remaining_globals + remaining_specialized:
            idk_probability = self._specialized_idk_prob(
                rejected_initial, group, rejected_specialized, router_id, candidate_id
            )
            cost = self._cost(candidate_id)

            if candidate_id in self.global_ids:
                next_rejected_initial = tuple(sorted(rejected_initial_set | {candidate_id}))
                cost += idk_probability * self.expand_prime(
                    next_rejected_initial, group, rejected_specialized, router_id
                )
            else:
                next_rejected_specialized = tuple(sorted(rejected_specialized_set | {candidate_id}))
                cost += idk_probability * self.expand_prime(
                    rejected_initial, group, next_rejected_specialized, router_id
                )

            if cost < best_cost:
                best_cost = cost
                best_next = candidate_id

        key = (rejected_initial, group, rejected_specialized, router_id)
        self._expand_prime_next[key] = best_next
        return best_cost

    def synthesize(self) -> Cascade:
        """Build the initial cascade + one specialized cascade per
        (identifier, group) pair by walking the `.next` pointers populated
        during the DP (mirrors paper Algorithm 2).

        Note on accuracy: this returns the cascade that minimizes EXPECTED
        TIME only. It does not weight by Kdet's accuracy (paper assumes
        Kdet is always correct, so this doesn't matter there). If you run
        with detector_mode="trained" (the real, imperfect Kdet), the
        resulting `expected_cost` is still a pure time figure -- track
        end-to-end accuracy separately by replaying this cascade against
        ground truth, the same way the friend's runtime cascade builder
        does for the ImageNet repo.
        """
        expected_cost = self.expand(tuple())
        initial: list[str] = []
        specialized: dict[tuple[str, str], list[str]] = {}
        rejected_initial: tuple[str, ...] = tuple()

        while True:
            next_id = self._expand_next.get(rejected_initial, self.detector_id)
            initial.append(next_id)
            if next_id == self.detector_id:
                break

            if next_id in self.identifier_ids:
                for group in self.groups:
                    specialized[(next_id, group)] = self._synthesize_specialized(
                        rejected_initial, group, next_id
                    )

            rejected_initial = tuple(sorted(set(rejected_initial) | {next_id}))

        return Cascade(
            expected_cost=expected_cost,
            initial=initial,
            specialized=specialized,
            detector=self.detector_id,
        )

    def _synthesize_specialized(
        self, rejected_initial: tuple[str, ...], group: str, router_id: str
    ) -> list[str]:
        chain: list[str] = []
        rejected_specialized: tuple[str, ...] = tuple()

        while True:
            key = (rejected_initial, group, rejected_specialized, router_id)
            next_id = self._expand_prime_next.get(key, self.detector_id)
            chain.append(next_id)
            if next_id == self.detector_id:
                break
            if next_id in self.global_ids:
                rejected_initial = tuple(sorted(set(rejected_initial) | {next_id}))
            else:
                rejected_specialized = tuple(sorted(set(rejected_specialized) | {next_id}))

        return chain

    def describe(self, candidate_id: str) -> dict:
        if candidate_id == self.detector_id:
            return {"id": self.detector_id, "kind": "detector", "cost": self.detector_cost}
        row = self.candidates.loc[candidate_id]
        return {
            "id": candidate_id,
            "kind": row["kind"],
            "group": row["group"],
            "cost": row["cost"],
            "wcet": row["wcet"],
            "threshold": row["threshold"],
        }


def optimize_empirical_hierarchy(
    path: str | Path = DEFAULT_OUTPUT_PATH,
    detector_mode: str = "paper",
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
) -> tuple[HierarchyOptimizer, Cascade]:
    payload = load_empirical_outcomes(path)
    optimizer = HierarchyOptimizer(payload, detector_mode=detector_mode, detector_cost_ms=detector_cost_ms)
    cascade = optimizer.synthesize()
    return optimizer, cascade


def print_cascade(
    path: str | Path = DEFAULT_OUTPUT_PATH,
    detector_mode: str = "paper",
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
) -> tuple[HierarchyOptimizer, Cascade]:
    optimizer, cascade = optimize_empirical_hierarchy(path, detector_mode, detector_cost_ms)
    print(f"detector_mode={detector_mode} (cost={optimizer.detector_cost}ms)")
    print(f"expected_cost: {cascade.expected_cost:.4f} ms")
    print("initial:")
    for candidate_id in cascade.initial:
        print("  ", optimizer.describe(candidate_id))
    print("specialized:")
    for (router_id, group), chain in cascade.specialized.items():
        print(f"  {router_id} group={group}: {chain}")
    return optimizer, cascade


def compare_kdet_costs(
    path: str | Path = DEFAULT_OUTPUT_PATH,
    detector_costs_ms: list[float] = (10.85, 100, 250, 1000, 10_000),
) -> dict[float, Cascade]:
    """Print h24's DP-optimal cascade structure side by side for several
    candidate synthetic Kdet costs, so you can eyeball how much richer the
    chain gets as the assumed Kdet cost increases -- instead of debating
    it in the abstract or committing to one value blind.

    10.85 is included as a reference point: that's the REAL measured Kdet
    cost (detector_mode="trained"), i.e. what the DP would choose if Kdet
    weren't artificially inflated at all -- usually a very short chain,
    useful as a baseline to see how much the synthetic values are actually
    buying you in chain depth.
    """
    results: dict[float, Cascade] = {}
    for cost in detector_costs_ms:
        print(f"\n{'=' * 60}")
        print(f"detector_cost_ms = {cost}")
        print(f"{'=' * 60}")
        optimizer, cascade = optimize_empirical_hierarchy(path, "paper", cost)
        print(f"expected_cost: {cascade.expected_cost:.4f} ms")
        print("initial:")
        for candidate_id in cascade.initial:
            print("  ", optimizer.describe(candidate_id))
        print("specialized:")
        for (router_id, group), chain in cascade.specialized.items():
            print(f"  {router_id} group={group}: {chain}")
        results[cost] = cascade

    print(f"\n{'-' * 60}")
    print("SUMMARY: chain length (initial, excluding detector) per Kdet cost")
    print(f"{'-' * 60}")
    for cost, cascade in results.items():
        initial_len = len([c for c in cascade.initial if c != cascade.detector])
        spec_lens = {k: len(v) for k, v in cascade.specialized.items()}
        print(f"  cost={cost:>8}ms: initial_len={initial_len}, specialized_lens={spec_lens}, "
              f"expected_cost={cascade.expected_cost:.2f}ms")

    return results


if __name__ == "__main__":
    print_cascade()