"""Official-training-only CIFAR-100 loading and deterministic split support."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

from experiments.cifar100.labels import (
    CIFAR100_PROFILE,
    COARSE_LABEL_NAMES,
    FINE_LABEL_NAMES,
    FINE_TO_COARSE_INDEX,
    coarse_index,
    specialist_global_to_local,
)


DEFAULT_SPLIT_SEED = 2025
SPLIT_SCHEMA_VERSION = "cifar100-splits/v1"
SPLIT_FILE_STEM = "cifar100_split_indices"
TRAIN_PER_FINE_CLASS = 425
MODEL_SELECTION_PER_FINE_CLASS = 25
CASCADE_VALIDATION_PER_FINE_CLASS = 50
OFFICIAL_TRAIN_SAMPLES_PER_FINE_CLASS = 500
OFFICIAL_TRAIN_SAMPLE_COUNT = 50_000

# Statistics of the official 50,000-image training set, with pixel values in
# [0, 1]. They are shared by all CIFAR-100 roles.
CIFAR100_MEAN = (0.50707516, 0.48654887, 0.44091784)
CIFAR100_STD = (0.26733429, 0.25643846, 0.27615047)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CONVNEXT_INPUT_SIZE = 224


@dataclass(frozen=True)
class CIFAR100TrainingDataset:
    """A torchvision training dataset plus validated official coarse labels."""

    base_dataset: Dataset
    fine_targets: np.ndarray
    coarse_targets: np.ndarray
    raw_files: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        fine = _readonly_int_array(self.fine_targets, "fine_targets")
        coarse = _readonly_int_array(self.coarse_targets, "coarse_targets")
        if len(fine) != len(coarse) or len(fine) != len(self.base_dataset):
            raise ValueError("Dataset, fine targets, and coarse targets differ in length.")
        object.__setattr__(self, "fine_targets", fine)
        object.__setattr__(self, "coarse_targets", coarse)
        object.__setattr__(
            self, "raw_files", tuple(Path(path) for path in self.raw_files)
        )

    def __len__(self) -> int:
        return len(self.fine_targets)


@dataclass(frozen=True)
class CIFAR100SplitIndices:
    """Exact source indices for the three partitions of official training data."""

    train: np.ndarray
    model_selection: np.ndarray
    cascade_validation: np.ndarray
    seed: int = DEFAULT_SPLIT_SEED

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("Split seed must be an integer.")
        for name in ("train", "model_selection", "cascade_validation"):
            object.__setattr__(self, name, _readonly_int_array(getattr(self, name), name))

    @property
    def by_name(self) -> dict[str, np.ndarray]:
        return {
            "train": self.train,
            "model_selection": self.model_selection,
            "cascade_validation": self.cascade_validation,
        }


class CIFAR100DatasetView(Dataset):
    """A transform/target-specific view over immutable official train rows."""

    def __init__(
        self,
        source: CIFAR100TrainingDataset,
        indices: Sequence[int] | np.ndarray,
        *,
        target_mode: str,
        group: int | str | None,
        transform: Any,
    ) -> None:
        if target_mode not in {"fine", "coarse", "specialist"}:
            raise ValueError("target_mode must be 'fine', 'coarse', or 'specialist'.")
        if target_mode == "specialist" and group is None:
            raise ValueError("A specialist dataset view requires a coarse group.")
        if target_mode != "specialist" and group is not None:
            raise ValueError("group is valid only for a specialist dataset view.")

        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("Dataset-view indices must be one-dimensional.")
        if len(selected) and (selected.min() < 0 or selected.max() >= len(source)):
            raise ValueError("Dataset-view indices are outside the source dataset.")

        self.group_index = None if group is None else coarse_index(group)
        if self.group_index is not None:
            selected = selected[source.coarse_targets[selected] == self.group_index]
        selected = np.array(selected, dtype=np.int64, copy=True)
        selected.setflags(write=False)

        self.source = source
        self.indices = selected
        self.target_mode = target_mode
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Any, int]:
        source_index = int(self.indices[item])
        sample = self.source.base_dataset[source_index]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise ValueError("The base CIFAR-100 dataset must return (image, target).")
        image, returned_fine = sample[0], int(sample[1])
        expected_fine = int(self.source.fine_targets[source_index])
        if returned_fine != expected_fine:
            raise ValueError("The base dataset target differs from validated raw labels.")
        if self.transform is not None:
            image = self.transform(image)

        if self.target_mode == "fine":
            target = expected_fine
        elif self.target_mode == "coarse":
            target = int(self.source.coarse_targets[source_index])
        else:
            target = specialist_global_to_local(int(self.group_index), expected_fine)
        return image, target

    def source_index(self, item: int) -> int:
        """Return the stable official-training index for a view row."""

        return int(self.indices[item])


def _readonly_int_array(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    array = np.array(array, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    normalized = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(json.dumps(normalized.shape).encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def _pickle_field(payload: dict[Any, Any], name: str) -> Any:
    if name in payload:
        return payload[name]
    encoded = name.encode("ascii")
    if encoded in payload:
        return payload[encoded]
    raise ValueError(f"Raw CIFAR-100 pickle is missing {name!r}.")


def _decode_names(values: Sequence[str | bytes]) -> tuple[str, ...]:
    return tuple(
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    )


def validate_official_training_labels(
    fine_targets: Sequence[int] | np.ndarray,
    coarse_targets: Sequence[int] | np.ndarray,
    *,
    require_full_training_set: bool = True,
) -> None:
    """Validate raw coarse labels against the canonical hierarchy mapping."""

    fine = np.asarray(fine_targets, dtype=np.int64)
    coarse = np.asarray(coarse_targets, dtype=np.int64)
    if fine.ndim != 1 or coarse.ndim != 1 or len(fine) != len(coarse):
        raise ValueError("Fine and coarse target arrays must be one-dimensional peers.")
    if len(fine) and (fine.min() < 0 or fine.max() >= len(FINE_LABEL_NAMES)):
        raise ValueError("Raw data contain an invalid CIFAR-100 fine label.")
    if len(coarse) and (
        coarse.min() < 0 or coarse.max() >= len(COARSE_LABEL_NAMES)
    ):
        raise ValueError("Raw data contain an invalid CIFAR-100 coarse label.")

    expected_coarse = np.asarray(FINE_TO_COARSE_INDEX, dtype=np.int64)[fine]
    mismatch = np.flatnonzero(coarse != expected_coarse)
    if len(mismatch):
        index = int(mismatch[0])
        raise ValueError(
            "Raw CIFAR-100 coarse mapping disagrees with the official hierarchy "
            f"at source index {index}."
        )

    if require_full_training_set:
        if len(fine) != OFFICIAL_TRAIN_SAMPLE_COUNT:
            raise ValueError(
                f"Expected {OFFICIAL_TRAIN_SAMPLE_COUNT} official training rows, "
                f"found {len(fine)}."
            )
        fine_counts = np.bincount(fine, minlength=len(FINE_LABEL_NAMES))
        if not np.all(fine_counts == OFFICIAL_TRAIN_SAMPLES_PER_FINE_CLASS):
            raise ValueError("Every official fine class must have 500 training rows.")
        coarse_counts = np.bincount(coarse, minlength=len(COARSE_LABEL_NAMES))
        if not np.all(coarse_counts == 2_500):
            raise ValueError("Every official coarse class must have 2,500 training rows.")


def _raw_training_paths(dataset: Dataset) -> tuple[Path, ...]:
    root = Path(str(getattr(dataset, "root")))
    base_folder = str(getattr(dataset, "base_folder"))
    train_list = getattr(dataset, "train_list")
    paths = tuple(root / base_folder / str(entry[0]) for entry in train_list)
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError("The verified CIFAR-100 raw training pickle is missing.")
    return paths


def _raw_meta_path(dataset: Dataset) -> Path:
    root = Path(str(getattr(dataset, "root")))
    base_folder = str(getattr(dataset, "base_folder"))
    meta = getattr(dataset, "meta", {})
    filename = meta.get("filename", "meta")
    path = root / base_folder / str(filename)
    if not path.is_file():
        raise FileNotFoundError("The verified CIFAR-100 metadata pickle is missing.")
    return path


def load_training_dataset(
    root: str | Path,
    download: bool = False,
    *,
    dataset_class: type = CIFAR100,
) -> CIFAR100TrainingDataset:
    """Load and validate only the official CIFAR-100 training split.

    torchvision performs its built-in MD5 integrity check during construction.
    This adapter additionally checks the raw fine/coarse mapping and metadata.
    It intentionally has no API for constructing the official test split.
    """

    dataset = dataset_class(
        root=str(Path(root)),
        train=True,
        transform=None,
        target_transform=None,
        download=bool(download),
    )
    integrity_check = getattr(dataset, "_check_integrity", None)
    if callable(integrity_check) and not bool(integrity_check()):
        raise ValueError("torchvision reports failed CIFAR-100 dataset integrity.")

    raw_paths = _raw_training_paths(dataset)
    fine_parts: list[np.ndarray] = []
    coarse_parts: list[np.ndarray] = []
    for path in raw_paths:
        with path.open("rb") as stream:
            payload = pickle.load(stream, encoding="latin1")
        fine_parts.append(np.asarray(_pickle_field(payload, "fine_labels"), dtype=np.int64))
        coarse_parts.append(
            np.asarray(_pickle_field(payload, "coarse_labels"), dtype=np.int64)
        )
    fine = np.concatenate(fine_parts)
    coarse = np.concatenate(coarse_parts)

    torchvision_targets = np.asarray(getattr(dataset, "targets"), dtype=np.int64)
    if not np.array_equal(fine, torchvision_targets):
        raise ValueError("torchvision targets differ from the raw official fine labels.")

    meta_path = _raw_meta_path(dataset)
    with meta_path.open("rb") as stream:
        meta_payload = pickle.load(stream, encoding="latin1")
    fine_names = _decode_names(_pickle_field(meta_payload, "fine_label_names"))
    coarse_names = _decode_names(_pickle_field(meta_payload, "coarse_label_names"))
    if fine_names != FINE_LABEL_NAMES or coarse_names != COARSE_LABEL_NAMES:
        raise ValueError("Raw CIFAR-100 metadata label order is not the official order.")
    dataset_classes = getattr(dataset, "classes", None)
    if dataset_classes is not None and tuple(str(item) for item in dataset_classes) != FINE_LABEL_NAMES:
        raise ValueError("torchvision class names differ from official raw metadata.")

    validate_official_training_labels(fine, coarse)
    return CIFAR100TrainingDataset(
        base_dataset=dataset,
        fine_targets=fine,
        coarse_targets=coarse,
        raw_files=(*raw_paths, meta_path),
    )


def generate_stratified_splits(
    fine_targets: Sequence[int] | np.ndarray,
    seed: int = DEFAULT_SPLIT_SEED,
) -> CIFAR100SplitIndices:
    """Make the exact 425/25/50 per-fine-class deterministic partition."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Split seed must be an integer.")
    fine = np.asarray(fine_targets, dtype=np.int64)
    if fine.ndim != 1 or len(fine) != OFFICIAL_TRAIN_SAMPLE_COUNT:
        raise ValueError("Splits require all 50,000 official training targets.")
    if len(fine) and (fine.min() < 0 or fine.max() >= len(FINE_LABEL_NAMES)):
        raise ValueError("Fine targets contain an invalid class index.")
    counts = np.bincount(fine, minlength=len(FINE_LABEL_NAMES))
    if not np.all(counts == OFFICIAL_TRAIN_SAMPLES_PER_FINE_CLASS):
        raise ValueError("Splits require exactly 500 rows for every fine class.")

    rng = np.random.default_rng(seed)
    train: list[np.ndarray] = []
    model_selection: list[np.ndarray] = []
    cascade_validation: list[np.ndarray] = []
    for fine_index in range(len(FINE_LABEL_NAMES)):
        shuffled = rng.permutation(np.flatnonzero(fine == fine_index))
        train.append(shuffled[:TRAIN_PER_FINE_CLASS])
        model_selection.append(
            shuffled[
                TRAIN_PER_FINE_CLASS : TRAIN_PER_FINE_CLASS
                + MODEL_SELECTION_PER_FINE_CLASS
            ]
        )
        cascade_validation.append(
            shuffled[
                TRAIN_PER_FINE_CLASS + MODEL_SELECTION_PER_FINE_CLASS :
            ]
        )

    result = CIFAR100SplitIndices(
        train=np.sort(np.concatenate(train)),
        model_selection=np.sort(np.concatenate(model_selection)),
        cascade_validation=np.sort(np.concatenate(cascade_validation)),
        seed=seed,
    )
    validate_split_bundle(result, fine)
    return result


def validate_split_bundle(
    bundle: CIFAR100SplitIndices,
    fine_targets: Sequence[int] | np.ndarray,
    coarse_targets: Sequence[int] | np.ndarray | None = None,
) -> None:
    """Check exact sizes, disjointness, coverage, and fine stratification."""

    fine = np.asarray(fine_targets, dtype=np.int64)
    if fine.ndim != 1 or len(fine) != OFFICIAL_TRAIN_SAMPLE_COUNT:
        raise ValueError("Split validation requires 50,000 fine targets.")
    if coarse_targets is not None:
        validate_official_training_labels(fine, coarse_targets)

    expected_sizes = {
        "train": TRAIN_PER_FINE_CLASS * len(FINE_LABEL_NAMES),
        "model_selection": MODEL_SELECTION_PER_FINE_CLASS * len(FINE_LABEL_NAMES),
        "cascade_validation": CASCADE_VALIDATION_PER_FINE_CLASS
        * len(FINE_LABEL_NAMES),
    }
    all_indices: list[np.ndarray] = []
    per_class_expected = {
        "train": TRAIN_PER_FINE_CLASS,
        "model_selection": MODEL_SELECTION_PER_FINE_CLASS,
        "cascade_validation": CASCADE_VALIDATION_PER_FINE_CLASS,
    }
    for name, indices in bundle.by_name.items():
        if len(indices) != expected_sizes[name]:
            raise ValueError(f"Split {name!r} has the wrong number of rows.")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"Split {name!r} repeats source indices.")
        if len(indices) and (indices.min() < 0 or indices.max() >= len(fine)):
            raise ValueError(f"Split {name!r} contains an invalid source index.")
        class_counts = np.bincount(fine[indices], minlength=len(FINE_LABEL_NAMES))
        if not np.all(class_counts == per_class_expected[name]):
            raise ValueError(f"Split {name!r} is not exactly fine-label stratified.")
        all_indices.append(indices)

    combined = np.concatenate(all_indices)
    if len(np.unique(combined)) != OFFICIAL_TRAIN_SAMPLE_COUNT:
        raise ValueError("CIFAR-100 split indices overlap.")
    if not np.array_equal(np.sort(combined), np.arange(OFFICIAL_TRAIN_SAMPLE_COUNT)):
        raise ValueError("CIFAR-100 split indices do not cover official training data.")


def save_split_bundle(
    bundle: CIFAR100SplitIndices,
    output_dir: str | Path,
    *,
    fine_targets: Sequence[int] | np.ndarray,
    coarse_targets: Sequence[int] | np.ndarray,
    source_files: Iterable[str | Path] = (),
) -> tuple[Path, Path]:
    """Save exact indices to NPZ and a readable checksummed JSON manifest."""

    fine = np.asarray(fine_targets, dtype=np.int64)
    coarse = np.asarray(coarse_targets, dtype=np.int64)
    validate_split_bundle(bundle, fine, coarse)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    npz_path = destination / f"{SPLIT_FILE_STEM}.npz"
    manifest_path = destination / f"{SPLIT_FILE_STEM}.json"
    np.savez_compressed(
        npz_path,
        train=bundle.train,
        model_selection=bundle.model_selection,
        cascade_validation=bundle.cascade_validation,
        seed=np.asarray(bundle.seed, dtype=np.int64),
    )

    source_entries = []
    for source_file in source_files:
        path = Path(source_file)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot checksum missing source file: {path}")
        source_entries.append(
            {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        )
    manifest = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "dataset_id": CIFAR100_PROFILE.dataset_id,
        "profile_fingerprint": CIFAR100_PROFILE.fingerprint,
        "source_partition": "official_train",
        "official_test_used": False,
        "seed": bundle.seed,
        "rng": "numpy.random.Generator(PCG64)",
        "counts": {name: int(len(indices)) for name, indices in bundle.by_name.items()},
        "per_fine_class": {
            "train": TRAIN_PER_FINE_CLASS,
            "model_selection": MODEL_SELECTION_PER_FINE_CLASS,
            "cascade_validation": CASCADE_VALIDATION_PER_FINE_CLASS,
        },
        "checksums": {
            "indices_npz_sha256": _sha256_file(npz_path),
            "fine_targets_sha256": _sha256_array(fine),
            "coarse_targets_sha256": _sha256_array(coarse),
            "arrays": {
                name: _sha256_array(indices) for name, indices in bundle.by_name.items()
            },
        },
        "files": {
            "indices": npz_path.name,
            "sources": source_entries,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return npz_path, manifest_path


def load_split_bundle(
    npz_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    fine_targets: Sequence[int] | np.ndarray | None = None,
    coarse_targets: Sequence[int] | np.ndarray | None = None,
) -> CIFAR100SplitIndices:
    """Load a split bundle and verify its manifest and optional source labels."""

    source = Path(npz_path)
    manifest_source = (
        source.with_suffix(".json") if manifest_path is None else Path(manifest_path)
    )
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("Unsupported CIFAR-100 split manifest schema.")
    if manifest.get("dataset_id") != CIFAR100_PROFILE.dataset_id:
        raise ValueError("Split manifest is for a different dataset.")
    if manifest.get("profile_fingerprint") != CIFAR100_PROFILE.fingerprint:
        raise ValueError("Split manifest hierarchy fingerprint has changed.")
    expected_npz = manifest.get("checksums", {}).get("indices_npz_sha256")
    if expected_npz != _sha256_file(source):
        raise ValueError("CIFAR-100 split NPZ checksum mismatch.")

    with np.load(source, allow_pickle=False) as values:
        bundle = CIFAR100SplitIndices(
            train=values["train"],
            model_selection=values["model_selection"],
            cascade_validation=values["cascade_validation"],
            seed=int(values["seed"]),
        )
    if bundle.seed != int(manifest["seed"]):
        raise ValueError("Split seed differs between NPZ and manifest.")
    expected_arrays = manifest.get("checksums", {}).get("arrays", {})
    for name, indices in bundle.by_name.items():
        if expected_arrays.get(name) != _sha256_array(indices):
            raise ValueError(f"CIFAR-100 split array checksum mismatch: {name}")

    if fine_targets is not None:
        fine = np.asarray(fine_targets, dtype=np.int64)
        if manifest["checksums"].get("fine_targets_sha256") != _sha256_array(fine):
            raise ValueError("Fine-target checksum differs from the split manifest.")
        if coarse_targets is not None:
            coarse = np.asarray(coarse_targets, dtype=np.int64)
            if manifest["checksums"].get("coarse_targets_sha256") != _sha256_array(coarse):
                raise ValueError("Coarse-target checksum differs from the split manifest.")
        validate_split_bundle(bundle, fine, coarse_targets)
    elif coarse_targets is not None:
        raise ValueError("coarse_targets cannot be checked without fine_targets.")
    else:
        expected_sizes = {
            "train": TRAIN_PER_FINE_CLASS * len(FINE_LABEL_NAMES),
            "model_selection": MODEL_SELECTION_PER_FINE_CLASS * len(FINE_LABEL_NAMES),
            "cascade_validation": CASCADE_VALIDATION_PER_FINE_CLASS
            * len(FINE_LABEL_NAMES),
        }
        if any(
            len(bundle.by_name[name]) != expected_size
            for name, expected_size in expected_sizes.items()
        ):
            raise ValueError("Loaded split bundle has incorrect partition sizes.")
        combined = np.concatenate(tuple(bundle.by_name.values()))
        if not np.array_equal(np.sort(combined), np.arange(OFFICIAL_TRAIN_SAMPLE_COUNT)):
            raise ValueError("Loaded split bundle does not cover official training indices.")
    return bundle


def build_training_transform() -> Compose:
    """Canonical 32x32 CIFAR augmentation followed by official normalization."""

    return Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def build_evaluation_transform() -> Compose:
    """Deterministic 32x32 CIFAR evaluation preprocessing."""

    return Compose([ToTensor(), Normalize(CIFAR100_MEAN, CIFAR100_STD)])


def build_convnext_evaluation_transform() -> Compose:
    """Official 224px ImageNet preprocessing for the pretrained endpoint."""

    return Compose(
        [
            Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            CenterCrop(CONVNEXT_INPUT_SIZE),
            ToTensor(),
            Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataset_view(
    base: CIFAR100TrainingDataset,
    indices: Sequence[int] | np.ndarray,
    target_mode: str = "fine",
    group: int | str | None = None,
    train: bool = False,
    *,
    transform: Any | None = None,
) -> CIFAR100DatasetView:
    """Build a split-specific fine, coarse, or five-way specialist view."""

    resolved_transform = (
        transform
        if transform is not None
        else (build_training_transform() if train else build_evaluation_transform())
    )
    return CIFAR100DatasetView(
        base,
        indices,
        target_mode=target_mode,
        group=group,
        transform=resolved_transform,
    )


__all__ = [
    "CASCADE_VALIDATION_PER_FINE_CLASS",
    "CIFAR100DatasetView",
    "CIFAR100TrainingDataset",
    "CIFAR100SplitIndices",
    "CIFAR100_MEAN",
    "CIFAR100_STD",
    "CONVNEXT_INPUT_SIZE",
    "DEFAULT_SPLIT_SEED",
    "MODEL_SELECTION_PER_FINE_CLASS",
    "TRAIN_PER_FINE_CLASS",
    "build_dataset_view",
    "build_convnext_evaluation_transform",
    "build_evaluation_transform",
    "build_training_transform",
    "generate_stratified_splits",
    "load_split_bundle",
    "load_training_dataset",
    "save_split_bundle",
    "validate_official_training_labels",
    "validate_split_bundle",
]
