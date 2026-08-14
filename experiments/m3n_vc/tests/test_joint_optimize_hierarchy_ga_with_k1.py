from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from experiments.m3n_vc.joint_optimize_hierarchy_ga_with_k1 import (
    build_k1_layout_space,
    legal_layout_count,
)
from layout_search import (
    TopologyGenome,
    cascade_from_genome,
    crossover_genomes,
    mutate_genome,
    repair_genome,
)


class K1EnabledGenomeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        payload = {
            "candidates": pd.DataFrame(
                [
                    {"id": "K0", "kind": "identifier", "group": None},
                    {"id": "K1", "kind": "identifier", "group": None},
                    {"id": "K2", "kind": "global", "group": None},
                    {"id": "K3", "kind": "global", "group": None},
                    {"id": "K4", "kind": "specialized", "group": "suv"},
                    {"id": "K5", "kind": "specialized", "group": "coupe"},
                    {"id": "K6", "kind": "specialized", "group": "coupe"},
                ]
            )
        }
        cls.space = build_k1_layout_space(payload)

    def test_full_layout_count_includes_two_router_modules(self) -> None:
        self.assertEqual(legal_layout_count(self.space), 11_589_085)
        self.assertEqual(self.space.router_ids, ("K0", "K1"))

    def test_two_router_genome_materializes_all_branches(self) -> None:
        genome = repair_genome(
            TopologyGenome(
                initial=("K0", "K3", "K1", "K2"),
                branches=(
                    ("K0", "coupe", ("K6", "K3")),
                    ("K0", "suv", ("K4", "K2")),
                    ("K1", "coupe", ("K5", "K2", "K3")),
                    ("K1", "suv", ("K4", "K2", "K3")),
                ),
            ),
            self.space,
        )
        cascade = cascade_from_genome(genome, self.space)
        self.assertEqual(
            set(cascade.specialized),
            {
                ("K0", "coupe"),
                ("K0", "suv"),
                ("K1", "coupe"),
                ("K1", "suv"),
            },
        )
        # K3 is before K1, so it cannot be repeated on a K1 branch.
        self.assertNotIn("K3", genome.branch_map[("K1", "coupe")])

    def test_mutation_and_crossover_preserve_dynamic_grammar(self) -> None:
        rng = np.random.default_rng(18)
        first = TopologyGenome(("K0", "K1", "K3"))
        second = TopologyGenome(("K2", "K1", "K0"))
        for _ in range(200):
            first = mutate_genome(first, self.space, rng)
            child = crossover_genomes(first, second, self.space, rng)
            self.assertEqual(child, repair_genome(child, self.space))
            self.assertTrue(set(child.initial) <= set(self.space.initial_ids))


if __name__ == "__main__":
    unittest.main()
