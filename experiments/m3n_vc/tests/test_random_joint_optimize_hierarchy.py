from __future__ import annotations

import unittest

from experiments.m3n_vc.brute_force_k1_free_layouts import (
    EXPECTED_LAYOUT_COUNT,
    enumerate_k1_free_layouts,
)
from experiments.m3n_vc.random_joint_optimize_hierarchy import uniform_layout_order
from experiments.m3n_vc.random_joint_optimize_hierarchy import (
    _validate_cached_records,
)


class RandomJointLayoutOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layouts = tuple(enumerate_k1_free_layouts())

    def test_uniform_order_is_complete_unique_and_deterministic(self) -> None:
        first = uniform_layout_order(self.layouts, 17)
        second = uniform_layout_order(self.layouts, 17)

        self.assertEqual(len(first), EXPECTED_LAYOUT_COUNT)
        self.assertEqual(
            [layout.layout_id for layout in first],
            [layout.layout_id for layout in second],
        )
        self.assertEqual(len({layout.layout_id for layout in first}), len(first))

    def test_uniform_order_changes_with_seed_and_never_contains_k1(self) -> None:
        first = uniform_layout_order(self.layouts, 17)
        second = uniform_layout_order(self.layouts, 18)

        self.assertNotEqual(first[0].layout_id, second[0].layout_id)
        self.assertTrue(
            all(
                "K1" not in layout.cascade.initial
                and all("K1" not in chain for chain in layout.cascade.specialized.values())
                for layout in first
            )
        )

    def test_cached_records_must_be_a_monotonic_prefix(self) -> None:
        order = uniform_layout_order(self.layouts, 17)
        settings = {"experiment": "test"}
        records = {}
        for rank, layout in enumerate(order[:2]):
            records[layout.layout_id] = {
                "layout_id": layout.layout_id,
                "layout_index": layout.index,
                "sample_rank": rank,
                "search_elapsed_seconds_at_completion": float(rank + 1),
                "settings": settings,
            }
        self.assertEqual(_validate_cached_records(records, order, settings), 2.0)

        records[order[1].layout_id]["sample_rank"] = 2
        with self.assertRaisesRegex(ValueError, "sample order"):
            _validate_cached_records(records, order, settings)


if __name__ == "__main__":
    unittest.main()
