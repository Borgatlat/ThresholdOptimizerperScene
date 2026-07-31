from __future__ import annotations

import unittest

from brute_force_k1_free_layouts import (
    EXPECTED_LAYOUT_COUNT,
    enumerate_k1_free_layouts,
)


class K1FreeLayoutEnumerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layouts = list(enumerate_k1_free_layouts())

    def test_enumerates_expected_number_of_unique_layouts(self) -> None:
        self.assertEqual(len(self.layouts), EXPECTED_LAYOUT_COUNT)
        self.assertEqual(
            len({layout.layout_id for layout in self.layouts}),
            EXPECTED_LAYOUT_COUNT,
        )
        self.assertEqual(
            [layout.index for layout in self.layouts],
            list(range(EXPECTED_LAYOUT_COUNT)),
        )

    def test_never_contains_removed_router(self) -> None:
        for indexed in self.layouts:
            cascade = indexed.cascade
            self.assertNotIn("K1", cascade.initial)
            for chain in cascade.specialized.values():
                self.assertNotIn("K1", chain)

    def test_globals_never_repeat_on_one_execution_path(self) -> None:
        globals_ = {"K2", "K3"}
        for indexed in self.layouts:
            cascade = indexed.cascade
            if "K0" not in cascade.initial:
                continue
            router_position = cascade.initial.index("K0")
            preceding_globals = globals_ & set(
                cascade.initial[:router_position]
            )
            for chain in cascade.specialized.values():
                self.assertTrue(preceding_globals.isdisjoint(chain))
                non_detector = [
                    candidate_id
                    for candidate_id in chain
                    if candidate_id != cascade.detector
                ]
                self.assertEqual(len(non_detector), len(set(non_detector)))

    def test_contains_fig1_single_global_reference(self) -> None:
        matches = [
            indexed
            for indexed in self.layouts
            if indexed.cascade.initial == ["K3", "detector"]
            and not indexed.cascade.specialized
        ]
        self.assertEqual(len(matches), 1)

    def test_only_detector_layout_is_unique(self) -> None:
        matches = [
            indexed
            for indexed in self.layouts
            if indexed.cascade.initial == ["detector"]
        ]
        self.assertEqual(len(matches), 1)


if __name__ == "__main__":
    unittest.main()
