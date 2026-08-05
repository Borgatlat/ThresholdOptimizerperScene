from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.cifar100.data import (
    CASCADE_VALIDATION_PER_FINE_CLASS,
    CIFAR100SplitIndices,
    CIFAR100TrainingDataset,
    MODEL_SELECTION_PER_FINE_CLASS,
    TRAIN_PER_FINE_CLASS,
    build_dataset_view,
    build_convnext_evaluation_transform,
    generate_stratified_splits,
    load_split_bundle,
    load_training_dataset,
    save_split_bundle,
    validate_official_training_labels,
)
from experiments.cifar100.labels import (
    COARSE_LABEL_NAMES,
    FINE_LABEL_NAMES,
    FINE_TO_COARSE_INDEX,
)


class _IndexDataset:
    def __init__(self, fine_targets: np.ndarray) -> None:
        self.targets = fine_targets.tolist()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return index, self.targets[index]


class CIFAR100DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fine = np.repeat(np.arange(100, dtype=np.int64), 500)
        cls.coarse = np.asarray(FINE_TO_COARSE_INDEX, dtype=np.int64)[cls.fine]

    def test_exact_deterministic_stratified_counts_and_disjointness(self) -> None:
        first = generate_stratified_splits(self.fine, seed=71)
        repeated = generate_stratified_splits(self.fine, seed=71)
        other_seed = generate_stratified_splits(self.fine, seed=72)

        for name in first.by_name:
            self.assertTrue(np.array_equal(first.by_name[name], repeated.by_name[name]))
        self.assertFalse(np.array_equal(first.train, other_seed.train))

        expected_per_class = {
            "train": TRAIN_PER_FINE_CLASS,
            "model_selection": MODEL_SELECTION_PER_FINE_CLASS,
            "cascade_validation": CASCADE_VALIDATION_PER_FINE_CLASS,
        }
        combined = []
        for name, indices in first.by_name.items():
            counts = np.bincount(self.fine[indices], minlength=100)
            self.assertTrue(np.all(counts == expected_per_class[name]))
            combined.append(indices)
        all_indices = np.concatenate(combined)
        self.assertEqual(len(np.unique(all_indices)), 50_000)
        self.assertTrue(np.array_equal(np.sort(all_indices), np.arange(50_000)))

    def test_split_npz_and_manifest_round_trip_with_checksums(self) -> None:
        splits = generate_stratified_splits(self.fine, seed=19)
        with tempfile.TemporaryDirectory() as directory:
            npz_path, manifest_path = save_split_bundle(
                splits,
                directory,
                fine_targets=self.fine,
                coarse_targets=self.coarse,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            loaded = load_split_bundle(
                npz_path,
                fine_targets=self.fine,
                coarse_targets=self.coarse,
            )

        self.assertFalse(manifest["official_test_used"])
        self.assertEqual(manifest["counts"]["train"], 42_500)
        self.assertEqual(manifest["counts"]["model_selection"], 2_500)
        self.assertEqual(manifest["counts"]["cascade_validation"], 5_000)
        for name in splits.by_name:
            self.assertTrue(np.array_equal(splits.by_name[name], loaded.by_name[name]))

    def test_raw_coarse_mapping_validation_rejects_one_bad_row(self) -> None:
        validate_official_training_labels(self.fine, self.coarse)
        corrupted = self.coarse.copy()
        corrupted[0] = (corrupted[0] + 1) % 20
        with self.assertRaisesRegex(ValueError, "official hierarchy"):
            validate_official_training_labels(self.fine, corrupted)

    def test_specialist_view_filters_group_and_maps_local_targets(self) -> None:
        source = CIFAR100TrainingDataset(
            base_dataset=_IndexDataset(self.fine),
            fine_targets=self.fine,
            coarse_targets=self.coarse,
        )
        all_indices = np.arange(50_000)
        view = build_dataset_view(
            source,
            all_indices,
            target_mode="specialist",
            group="aquatic_mammals",
            transform=lambda value: value,
        )

        self.assertEqual(len(view), 2_500)
        observed_targets = {view[index][1] for index in range(len(view))}
        self.assertEqual(observed_targets, set(range(5)))
        self.assertTrue(np.all(self.coarse[view.indices] == 0))
        for index in (0, len(view) // 2, len(view) - 1):
            image, _ = view[index]
            self.assertEqual(image, view.source_index(index))

    def test_loader_constructs_only_training_partition_and_checks_raw_labels(self) -> None:
        calls: list[dict] = []
        fine = self.fine
        coarse = self.coarse

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "cifar-100-python"
            raw.mkdir()
            with (raw / "train").open("wb") as stream:
                pickle.dump(
                    {"fine_labels": fine.tolist(), "coarse_labels": coarse.tolist()},
                    stream,
                )
            with (raw / "meta").open("wb") as stream:
                pickle.dump(
                    {
                        "fine_label_names": list(FINE_LABEL_NAMES),
                        "coarse_label_names": list(COARSE_LABEL_NAMES),
                    },
                    stream,
                )

            class FakeCIFAR100(_IndexDataset):
                base_folder = "cifar-100-python"
                train_list = (("train", "unused-md5"),)
                meta = {"filename": "meta"}

                def __init__(
                    self,
                    *,
                    root,
                    train,
                    transform,
                    target_transform,
                    download,
                ) -> None:
                    calls.append(
                        {
                            "root": root,
                            "train": train,
                            "transform": transform,
                            "target_transform": target_transform,
                            "download": download,
                        }
                    )
                    if train is not True:
                        raise AssertionError("Official test split must never be constructed.")
                    super().__init__(fine)
                    self.root = root
                    self.classes = list(FINE_LABEL_NAMES)

                def _check_integrity(self) -> bool:
                    return True

            loaded = load_training_dataset(
                root,
                download=False,
                dataset_class=FakeCIFAR100,
            )

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["train"], True)
        self.assertFalse(calls[0]["download"])
        self.assertEqual(len(loaded), 50_000)
        self.assertTrue(np.array_equal(loaded.coarse_targets, coarse))

    def test_split_bundle_dataclass_rejects_non_vector_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            CIFAR100SplitIndices(
                train=np.zeros((1, 1), dtype=np.int64),
                model_selection=np.array([], dtype=np.int64),
                cascade_validation=np.array([], dtype=np.int64),
            )

    def test_convnext_preprocessing_produces_224px_tensor(self) -> None:
        image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
        tensor = build_convnext_evaluation_transform()(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertTrue(np.isfinite(tensor.numpy()).all())


if __name__ == "__main__":
    unittest.main()
