"""Hierarchical label mappings for Ki classifiers (K0–K6)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GlobalLabel(str, Enum):
    GLE350 = "gle350"
    CX30 = "cx30"
    MUSTANG = "mustang"
    MIATA = "miata"
    BACKGROUND = "background"


class IntermediateLabel(str, Enum):
    SUV = "suv"
    COUPE = "coupe"
    BACKGROUND = "background"


# run0/run1 -> gle350, run2/run3 -> cx30, ...
RUN_NUMBER_TO_GLOBAL: dict[int, GlobalLabel] = {
    0: GlobalLabel.GLE350,
    1: GlobalLabel.GLE350,
    2: GlobalLabel.CX30,
    3: GlobalLabel.CX30,
    4: GlobalLabel.MUSTANG,
    5: GlobalLabel.MUSTANG,
    6: GlobalLabel.MIATA,
    7: GlobalLabel.MIATA,
    8: GlobalLabel.BACKGROUND,
    9: GlobalLabel.BACKGROUND,
}

GLOBAL_TO_INTERMEDIATE: dict[GlobalLabel, IntermediateLabel] = {
    GlobalLabel.GLE350: IntermediateLabel.SUV,
    GlobalLabel.CX30: IntermediateLabel.SUV,
    GlobalLabel.MUSTANG: IntermediateLabel.COUPE,
    GlobalLabel.MIATA: IntermediateLabel.COUPE,
    GlobalLabel.BACKGROUND: IntermediateLabel.BACKGROUND,
}


def run_id_to_global(run_id: str) -> GlobalLabel:
    """Map run id string (e.g. run3) to global vehicle label."""
    run_number = int(str(run_id).removeprefix("run"))
    return RUN_NUMBER_TO_GLOBAL[run_number]


def global_to_intermediate(global_label: GlobalLabel) -> IntermediateLabel:
    return GLOBAL_TO_INTERMEDIATE[global_label]


GLOBAL_CLASS_NAMES = [label.value for label in GlobalLabel]
INTERMEDIATE_CLASS_NAMES = [label.value for label in IntermediateLabel]
SUV_SPECIALIZED_NAMES = [GlobalLabel.GLE350.value, GlobalLabel.CX30.value]
COUPE_SPECIALIZED_NAMES = [GlobalLabel.MUSTANG.value, GlobalLabel.MIATA.value]


@dataclass(frozen=True)
class KiSpec:
    """Training configuration for one classifier in the cascade."""

    name: str
    level: str
    class_names: list[str]
    modality: str  # "mic", "both"
    subset: str  # "all", "suv", "coupe"


KI_REGISTRY: dict[str, KiSpec] = {
    "K0": KiSpec("K0", "intermediate", INTERMEDIATE_CLASS_NAMES, "both", "all"),
    "K1": KiSpec("K1", "intermediate", INTERMEDIATE_CLASS_NAMES, "both", "all"),
    "K2": KiSpec("K2", "global", GLOBAL_CLASS_NAMES, "both", "all"),
    "K3": KiSpec("K3", "global", GLOBAL_CLASS_NAMES, "both", "all"),
    "K4": KiSpec("K4", "specialized_suv", SUV_SPECIALIZED_NAMES, "mic", "suv"),
    "K5": KiSpec("K5", "specialized_coupe", COUPE_SPECIALIZED_NAMES, "mic", "coupe"),
    "K6": KiSpec("K6", "specialized_coupe", COUPE_SPECIALIZED_NAMES, "both", "coupe"),
    "Kdet": KiSpec("Kdet", "deterministic", GLOBAL_CLASS_NAMES, "both", "all") #both beacuas its global 
}

# Paper Section V-A (RTSS 2025): required confidence H_i for IDK deferral.
# Global Ki: 0.90; intermediate + specialized Ki: 0.95 (≤10% cumulative error).
PAPER_THRESHOLD_HI_BY_LEVEL: dict[str, float] = {
    "intermediate": 0.95,
    "global": 0.90,
    "specialized_suv": 0.95,
    "specialized_coupe": 0.95,
}


def is_deterministic_ki(ki_name: str) -> bool:
    """True for Kdet — never defers, always returns a base class."""
    return KI_REGISTRY[ki_name].level == "deterministic"


def threshold_hi_for_ki(ki_name: str) -> float | None:
    """Return paper-calibrated H_i for one Ki classifier, or None for Kdet."""
    if is_deterministic_ki(ki_name):
        return None
    level = KI_REGISTRY[ki_name].level
    return PAPER_THRESHOLD_HI_BY_LEVEL[level]


def label_to_index(label: str, class_names: list[str]) -> int:
    return class_names.index(label)


def metadata_row_labels(run_id: str) -> dict[str, str]:
    """Build global / intermediate / specialized string labels for one segment."""
    global_label = run_id_to_global(run_id)
    intermediate = global_to_intermediate(global_label)
    specialized = global_label.value if global_label != GlobalLabel.BACKGROUND else ""
    return {
        "global_label": global_label.value,
        "intermediate_label": intermediate.value,
        "specialized_label": specialized,
    }
