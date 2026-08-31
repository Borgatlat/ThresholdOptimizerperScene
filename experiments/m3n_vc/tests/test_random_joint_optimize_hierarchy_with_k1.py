from __future__ import annotations

import copy
import itertools
import unittest

import pandas as pd

from experiments.m3n_vc.brute_force_k1_free_layouts import _cascade_payload
from experiments.m3n_vc.joint_optimize_hierarchy_ga_with_k1 import (
    M3N_PROFILE,
    build_k1_layout_space,
    legal_layout_count,
)
from experiments.m3n_vc.random_joint_optimize_hierarchy_with_k1 import (
    ALGORITHM,
    ImplicitLayoutCatalogue,
    SETTINGS_SCHEMA_VERSION,
    _implementation_sha256,
    _ordered_subset_at,
    _ordered_subset_count,
    _prefix_sha256,
    _validate_cached_records,
    uniform_layout_prefix,
)
from layout_search import (
    LayoutSpace,
    cascade_from_genome,
    layout_id,
    repair_genome,
)


def _full_space() -> LayoutSpace:
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
    return build_k1_layout_space(payload)


def _small_space() -> LayoutSpace:
    return LayoutSpace(
        profile=M3N_PROFILE,
        global_ids=("G",),
        router_ids=("R",),
        specialized_by_group={"coupe": (), "suv": ()},
        detector_id="detector",
    )


class OrderedSubsetUnrankingTests(unittest.TestCase):
    def test_unranking_matches_length_then_permutation_order(self) -> None:
        values = ("a", "b", "c", "d")
        expected = tuple(
            sequence
            for length in range(len(values) + 1)
            for sequence in itertools.permutations(values, length)
        )
        actual = tuple(
            _ordered_subset_at(values, index) for index in range(len(expected))
        )
        self.assertEqual(actual, expected)
        self.assertEqual(_ordered_subset_count(len(values)), len(expected))

    def test_unranking_rejects_out_of_range_indices(self) -> None:
        with self.assertRaises(IndexError):
            _ordered_subset_at(("a",), -1)
        with self.assertRaises(IndexError):
            _ordered_subset_at(("a",), 3)


class ImplicitCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.space = _full_space()
        cls.catalogue = ImplicitLayoutCatalogue(cls.space)

    def test_full_space_count_without_materializing_layouts(self) -> None:
        self.assertEqual(len(self.catalogue), 11_589_085)
        self.assertEqual(len(self.catalogue), legal_layout_count(self.space))
        self.assertEqual(self.catalogue.block_count, 65)

    def test_small_space_index_is_a_complete_bijection(self) -> None:
        space = _small_space()
        catalogue = ImplicitLayoutCatalogue(space)
        genomes = tuple(catalogue.genome_at(index) for index in range(len(catalogue)))
        ids = {layout_id(genome, space) for genome in genomes}

        self.assertEqual(len(catalogue), 11)
        self.assertEqual(len(ids), len(catalogue))
        self.assertTrue(all(genome == repair_genome(genome, space) for genome in genomes))

    def test_boundaries_and_random_entries_are_canonical(self) -> None:
        indices = (0, 1, 17, 1_000, len(self.catalogue) - 1)
        for index in indices:
            genome = self.catalogue.genome_at(index)
            self.assertEqual(genome, repair_genome(genome, self.space))
            sampled = self.catalogue.sampled_layout(index, 4)
            self.assertEqual(sampled.space_index, index)
            self.assertEqual(sampled.layout_id, layout_id(genome, self.space))
        with self.assertRaises(IndexError):
            self.catalogue.genome_at(-1)
        with self.assertRaises(IndexError):
            self.catalogue.genome_at(len(self.catalogue))

    def test_uniform_prefix_is_deterministic_and_without_replacement(self) -> None:
        first = uniform_layout_prefix(self.catalogue, 512, random_seed=9)
        second = uniform_layout_prefix(self.catalogue, 512, random_seed=9)
        different = uniform_layout_prefix(self.catalogue, 512, random_seed=10)

        first_identity = tuple((item.space_index, item.layout_id) for item in first)
        self.assertEqual(
            first_identity,
            tuple((item.space_index, item.layout_id) for item in second),
        )
        self.assertNotEqual(
            first_identity,
            tuple((item.space_index, item.layout_id) for item in different),
        )
        self.assertEqual(len({item.space_index for item in first}), len(first))
        self.assertEqual(len({item.layout_id for item in first}), len(first))
        self.assertEqual([item.sample_rank for item in first], list(range(512)))

    def test_prefix_count_validation(self) -> None:
        self.assertEqual(uniform_layout_prefix(self.catalogue, 0, 0), ())
        with self.assertRaises(ValueError):
            uniform_layout_prefix(self.catalogue, -1, 0)
        with self.assertRaises(ValueError):
            uniform_layout_prefix(self.catalogue, len(self.catalogue) + 1, 0)


class PrefixCheckpointValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.space = _small_space()
        self.catalogue = ImplicitLayoutCatalogue(self.space)
        self.settings = {"experiment": "unit-test"}
        self.records = self._records(4)

    def _records(self, count: int) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for sampled in uniform_layout_prefix(self.catalogue, count, random_seed=3):
            record: dict[str, object] = {
                "layout_index": sampled.sample_rank,
                "space_index": sampled.space_index,
                "layout_id": sampled.layout_id,
                "layout": _cascade_payload(
                    cascade_from_genome(sampled.genome, self.space)
                ),
                "sample_rank": sampled.sample_rank,
                "search_elapsed_seconds_at_completion": sampled.sample_rank + 0.5,
                "settings": dict(self.settings),
            }
            result[sampled.layout_id] = record
        return result

    def test_valid_prefix_returns_last_elapsed_time(self) -> None:
        self.assertEqual(
            _validate_cached_records(
                self.records, self.catalogue, 3, self.settings
            ),
            3.5,
        )

    def test_validation_rejects_order_index_layout_and_settings_changes(self) -> None:
        corruptions = (
            ("sample_rank", 8, "sample ranks"),
            ("space_index", -9, "space index"),
            ("layout_index", -9, "evaluation-order index"),
            ("layout", {"initial": []}, "cascade payload"),
            ("settings", {"experiment": "other"}, "different experiment"),
        )
        first_id = next(iter(self.records))
        for field, value, message in corruptions:
            with self.subTest(field=field):
                records = copy.deepcopy(self.records)
                records[first_id][field] = value
                with self.assertRaisesRegex(ValueError, message):
                    _validate_cached_records(records, self.catalogue, 3, self.settings)

    def test_validation_rejects_nonmonotonic_elapsed_time(self) -> None:
        records = copy.deepcopy(self.records)
        by_rank = {
            int(record["sample_rank"]): record for record in records.values()
        }
        by_rank[2]["search_elapsed_seconds_at_completion"] = 0.1
        with self.assertRaisesRegex(ValueError, "elapsed times"):
            _validate_cached_records(records, self.catalogue, 3, self.settings)

    def test_prefix_hash_uses_sample_order_not_mapping_order(self) -> None:
        reversed_records = dict(reversed(tuple(self.records.items())))
        self.assertEqual(_prefix_sha256(self.records), _prefix_sha256(reversed_records))


class ContractTests(unittest.TestCase):
    def test_public_contract_identifiers_and_fingerprint(self) -> None:
        self.assertEqual(
            SETTINGS_SCHEMA_VERSION, "random-joint-layout-search-with-k1/v1"
        )
        self.assertEqual(
            ALGORITHM,
            "uniform_random_implicit_layout_sampling_without_replacement",
        )
        fingerprint = _implementation_sha256()
        self.assertEqual(len(fingerprint), 64)
        int(fingerprint, 16)


if __name__ == "__main__":
    unittest.main()
