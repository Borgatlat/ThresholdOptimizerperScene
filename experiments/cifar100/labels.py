"""Official CIFAR-100 fine/coarse labels and hierarchy mappings.

The label orders below are the orders stored in the official
``cifar-100-python/meta`` file.  Cascade predictions use the fine-label order
as their global output space and the coarse-label order as their router output
space.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

from cascade_profile import HierarchyProfile


FINE_LABEL_NAMES: tuple[str, ...] = (
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
)

COARSE_LABEL_NAMES: tuple[str, ...] = (
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
)

COARSE_TO_FINE_NAMES: Mapping[str, tuple[str, ...]] = {
    "aquatic_mammals": ("beaver", "dolphin", "otter", "seal", "whale"),
    "fish": ("aquarium_fish", "flatfish", "ray", "shark", "trout"),
    "flowers": ("orchid", "poppy", "rose", "sunflower", "tulip"),
    "food_containers": ("bottle", "bowl", "can", "cup", "plate"),
    "fruit_and_vegetables": (
        "apple",
        "mushroom",
        "orange",
        "pear",
        "sweet_pepper",
    ),
    "household_electrical_devices": (
        "clock",
        "keyboard",
        "lamp",
        "telephone",
        "television",
    ),
    "household_furniture": ("bed", "chair", "couch", "table", "wardrobe"),
    "insects": ("bee", "beetle", "butterfly", "caterpillar", "cockroach"),
    "large_carnivores": ("bear", "leopard", "lion", "tiger", "wolf"),
    "large_man-made_outdoor_things": (
        "bridge",
        "castle",
        "house",
        "road",
        "skyscraper",
    ),
    "large_natural_outdoor_scenes": (
        "cloud",
        "forest",
        "mountain",
        "plain",
        "sea",
    ),
    "large_omnivores_and_herbivores": (
        "camel",
        "cattle",
        "chimpanzee",
        "elephant",
        "kangaroo",
    ),
    "medium_mammals": (
        "fox",
        "porcupine",
        "possum",
        "raccoon",
        "skunk",
    ),
    "non-insect_invertebrates": ("crab", "lobster", "snail", "spider", "worm"),
    "people": ("baby", "boy", "girl", "man", "woman"),
    "reptiles": ("crocodile", "dinosaur", "lizard", "snake", "turtle"),
    "small_mammals": ("hamster", "mouse", "rabbit", "shrew", "squirrel"),
    "trees": ("maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"),
    "vehicles_1": ("bicycle", "bus", "motorcycle", "pickup_truck", "train"),
    "vehicles_2": ("lawn_mower", "rocket", "streetcar", "tank", "tractor"),
}

FINE_NAME_TO_INDEX = {name: index for index, name in enumerate(FINE_LABEL_NAMES)}
COARSE_NAME_TO_INDEX = {
    name: index for index, name in enumerate(COARSE_LABEL_NAMES)
}
COARSE_TO_FINE_INDICES: tuple[tuple[int, ...], ...] = tuple(
    tuple(FINE_NAME_TO_INDEX[name] for name in COARSE_TO_FINE_NAMES[coarse_name])
    for coarse_name in COARSE_LABEL_NAMES
)

_fine_to_coarse: list[int | None] = [None] * len(FINE_LABEL_NAMES)
_fine_to_local: list[int | None] = [None] * len(FINE_LABEL_NAMES)
for _coarse_index, _fine_indices in enumerate(COARSE_TO_FINE_INDICES):
    for _local_index, _fine_index in enumerate(_fine_indices):
        if _fine_to_coarse[_fine_index] is not None:
            raise RuntimeError(f"Fine label {_fine_index} appears in multiple groups.")
        _fine_to_coarse[_fine_index] = _coarse_index
        _fine_to_local[_fine_index] = _local_index
if any(value is None for value in _fine_to_coarse):
    raise RuntimeError("The official coarse mapping does not cover every fine label.")

FINE_TO_COARSE_INDEX: tuple[int, ...] = tuple(int(value) for value in _fine_to_coarse)
FINE_TO_SPECIALIST_LOCAL_INDEX: tuple[int, ...] = tuple(
    int(value) for value in _fine_to_local
)

CIFAR100_PROFILE = HierarchyProfile(
    dataset_id="cifar100/official",
    global_classes=FINE_LABEL_NAMES,
    groups={
        coarse_name: COARSE_TO_FINE_NAMES[coarse_name]
        for coarse_name in COARSE_LABEL_NAMES
    },
    router_outputs=COARSE_LABEL_NAMES,
    split_group_column="true_fine_label",
)


def cifar100_profile() -> HierarchyProfile:
    """Return the immutable official CIFAR-100 hierarchy profile."""

    return CIFAR100_PROFILE


def fine_index(label: int | str) -> int:
    """Return an official fine-label index from an index or label name."""

    if isinstance(label, bool):
        raise ValueError("Boolean values are not fine-label indices.")
    if isinstance(label, Integral):
        resolved = int(label)
        if 0 <= resolved < len(FINE_LABEL_NAMES):
            return resolved
        raise ValueError(f"Fine-label index is outside [0, 99]: {label}")
    try:
        return FINE_NAME_TO_INDEX[str(label)]
    except KeyError as exc:
        raise ValueError(f"Unknown CIFAR-100 fine label: {label!r}") from exc


def coarse_index(group: int | str) -> int:
    """Return an official coarse-label index from an index or group name."""

    if isinstance(group, bool):
        raise ValueError("Boolean values are not coarse-label indices.")
    if isinstance(group, Integral):
        resolved = int(group)
        if 0 <= resolved < len(COARSE_LABEL_NAMES):
            return resolved
        raise ValueError(f"Coarse-label index is outside [0, 19]: {group}")
    try:
        return COARSE_NAME_TO_INDEX[str(group)]
    except KeyError as exc:
        raise ValueError(f"Unknown CIFAR-100 coarse label: {group!r}") from exc


def specialist_global_to_local(group: int | str, label: int | str) -> int:
    """Map a member fine label to its five-way specialist output index."""

    group_index = coarse_index(group)
    global_index = fine_index(label)
    if FINE_TO_COARSE_INDEX[global_index] != group_index:
        raise ValueError(
            f"Fine label {FINE_LABEL_NAMES[global_index]!r} is outside specialist "
            f"group {COARSE_LABEL_NAMES[group_index]!r}."
        )
    return FINE_TO_SPECIALIST_LOCAL_INDEX[global_index]


def specialist_local_to_global(group: int | str, local_index: int) -> int:
    """Map a five-way specialist output index to the global fine-label index."""

    group_index = coarse_index(group)
    if isinstance(local_index, bool) or not isinstance(local_index, Integral):
        raise ValueError("Specialist local index must be an integer in [0, 4].")
    resolved_local = int(local_index)
    if not 0 <= resolved_local < 5:
        raise ValueError(f"Specialist local index is outside [0, 4]: {local_index}")
    return COARSE_TO_FINE_INDICES[group_index][resolved_local]


def fine_to_coarse(label: int | str) -> int:
    """Return the official coarse index for a fine index or name."""

    return FINE_TO_COARSE_INDEX[fine_index(label)]


__all__ = [
    "CIFAR100_PROFILE",
    "COARSE_LABEL_NAMES",
    "COARSE_NAME_TO_INDEX",
    "COARSE_TO_FINE_INDICES",
    "COARSE_TO_FINE_NAMES",
    "FINE_LABEL_NAMES",
    "FINE_NAME_TO_INDEX",
    "FINE_TO_COARSE_INDEX",
    "FINE_TO_SPECIALIST_LOCAL_INDEX",
    "cifar100_profile",
    "coarse_index",
    "fine_index",
    "fine_to_coarse",
    "specialist_global_to_local",
    "specialist_local_to_global",
]
