from __future__ import annotations

import unittest

from experiments.cifar100.labels import (
    CIFAR100_PROFILE,
    COARSE_LABEL_NAMES,
    COARSE_TO_FINE_INDICES,
    COARSE_TO_FINE_NAMES,
    FINE_LABEL_NAMES,
    FINE_TO_COARSE_INDEX,
    cifar100_profile,
    fine_to_coarse,
    specialist_global_to_local,
    specialist_local_to_global,
)


class CIFAR100LabelTests(unittest.TestCase):
    def test_official_mapping_is_complete_disjoint_and_five_per_group(self) -> None:
        self.assertEqual(len(FINE_LABEL_NAMES), 100)
        self.assertEqual(len(COARSE_LABEL_NAMES), 20)
        self.assertEqual(tuple(COARSE_TO_FINE_NAMES), COARSE_LABEL_NAMES)
        self.assertTrue(all(len(values) == 5 for values in COARSE_TO_FINE_INDICES))
        flattened = [index for values in COARSE_TO_FINE_INDICES for index in values]
        self.assertEqual(sorted(flattened), list(range(100)))
        self.assertEqual(len(FINE_TO_COARSE_INDEX), 100)

    def test_known_official_groups_and_orders(self) -> None:
        self.assertEqual(
            COARSE_TO_FINE_NAMES["aquatic_mammals"],
            ("beaver", "dolphin", "otter", "seal", "whale"),
        )
        self.assertEqual(
            COARSE_TO_FINE_NAMES["vehicles_1"],
            ("bicycle", "bus", "motorcycle", "pickup_truck", "train"),
        )
        self.assertEqual(fine_to_coarse("apple"), 4)
        self.assertEqual(fine_to_coarse("worm"), 13)

    def test_specialist_local_global_mapping_round_trips(self) -> None:
        for coarse_index, fine_indices in enumerate(COARSE_TO_FINE_INDICES):
            for local_index, global_index in enumerate(fine_indices):
                self.assertEqual(
                    specialist_global_to_local(coarse_index, global_index),
                    local_index,
                )
                self.assertEqual(
                    specialist_local_to_global(coarse_index, local_index),
                    global_index,
                )

    def test_specialist_rejects_out_of_group_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside specialist group"):
            specialist_global_to_local("aquatic_mammals", "apple")
        with self.assertRaisesRegex(ValueError, "outside"):
            specialist_local_to_global("aquatic_mammals", 5)

    def test_profile_uses_official_shared_label_spaces(self) -> None:
        profile = cifar100_profile()
        self.assertIs(profile, CIFAR100_PROFILE)
        self.assertEqual(profile.global_classes, FINE_LABEL_NAMES)
        self.assertEqual(profile.router_outputs, COARSE_LABEL_NAMES)
        self.assertEqual(tuple(profile.groups), COARSE_LABEL_NAMES)
        self.assertEqual(profile.max_router_depth, 1)


if __name__ == "__main__":
    unittest.main()
