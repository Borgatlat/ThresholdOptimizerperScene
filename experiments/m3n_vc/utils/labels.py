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
    """Map run id string (e.g. run3) to global vehicle label.

    WARNING: this uses a HARDCODED run-number convention (run0/1=gle350,
    run2/3=cx30, run4/5=mustang, run6/7=miata, run8/9=background) that only
    holds for h24, because h24 happened to record vehicles in that fixed
    order. It does NOT generalize to other M3N-VC scenes -- e.g. a06 has no
    Mustang runs at all and i22 has no GLE350 runs, so their run-number
    sequences cannot follow this same pattern. For any scene other than
    h24, use `load_scene_run_labels()` below, which reads the real
    ground-truth label from that scene's own run_ids.parquet instead of
    guessing from the run number.
    """
    run_number = int(str(run_id).removeprefix("run"))
    return RUN_NUMBER_TO_GLOBAL[run_number]


# Aliases seen (or plausibly seen) in different scenes' run_ids.parquet
# `label` column. M3N-VC's README abbreviates targets as C/G/M/X; extend
# this if a scene's actual label column uses a different string and
# load_scene_run_labels raises with the unmapped values it found.
_SCENE_LABEL_ALIASES: dict[str, GlobalLabel] = {
    "c": GlobalLabel.CX30, "cx30": GlobalLabel.CX30, "cx-30": GlobalLabel.CX30,
    "g": GlobalLabel.GLE350, "gle350": GlobalLabel.GLE350, "gle-350": GlobalLabel.GLE350,
    "m": GlobalLabel.MUSTANG, "mustang": GlobalLabel.MUSTANG,
    "x": GlobalLabel.MIATA, "mx5": GlobalLabel.MIATA, "mx-5": GlobalLabel.MIATA,
    "miata": GlobalLabel.MIATA,
    "none": GlobalLabel.BACKGROUND, "background": GlobalLabel.BACKGROUND,
    "bg": GlobalLabel.BACKGROUND, "": GlobalLabel.BACKGROUND,
}


def load_scene_run_labels(scene_dir: str) -> dict[str, GlobalLabel]:
    """Read the REAL per-run ground-truth label for a scene from its own
    run_ids.parquet (columns: run_id, label, set, start_time, end_time,
    length -- per the M3N-VC README), instead of guessing from the run
    number. Use this for every scene, including h24, going forward -- it's
    the correct source of truth; run_id_to_global's hardcoded table should
    be treated as h24-only legacy behavior.

    Raises with the actual unmapped label strings found, rather than
    silently mislabeling, if a scene uses label text not covered by
    _SCENE_LABEL_ALIASES -- extend that dict once you see the real values.
    """
    import pandas as pd
    from pathlib import Path

    run_ids_path = Path(scene_dir) / "run_ids.parquet"
    df = pd.read_parquet(run_ids_path)
    if "run_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"{run_ids_path} missing expected 'run_id'/'label' columns; "
            f"found: {list(df.columns)}"
        )

    result: dict[str, GlobalLabel] = {}
    unmapped: set[str] = set()
    skipped_multitarget: list[str] = []
    for _, row in df.iterrows():
        raw_label = row["label"]

        # Some scenes (i22 per the M3N-VC README: "multi-target (2)") have
        # runs with more than one vehicle present simultaneously -- `label`
        # comes back as an array/list of names for those rows instead of a
        # single string. Our GlobalLabel schema is single-vehicle-per-run;
        # there's no principled single answer for "the" vehicle in a
        # multi-target run, so skip those runs explicitly rather than
        # picking one name arbitrarily or crashing on the array repr.
        if hasattr(raw_label, "__len__") and not isinstance(raw_label, str):
            if len(raw_label) != 1:
                skipped_multitarget.append(f"{row['run_id']}: {list(raw_label)}")
                continue
            raw_label = raw_label[0]

        raw = str(raw_label).strip().lower()
        mapped = _SCENE_LABEL_ALIASES.get(raw)
        if mapped is None:
            unmapped.add(str(raw_label))
            continue

        # run_ids.parquet stores run_id as a bare int/string (e.g. 0, 1, 2),
        # but every mic/geo filename -- and therefore every segment's
        # metadata built from it -- uses the "run<N>" prefix (e.g. "run0",
        # from files like run0_rs1_mic.parquet, see process_data.py's
        # _file_metadata). Normalize here so lookups by "run0" actually hit,
        # instead of comparing "run0" against a bare "0" and never matching.
        run_id_str = str(row["run_id"]).strip()
        if not run_id_str.startswith("run"):
            run_id_str = f"run{run_id_str}"
        result[run_id_str] = mapped

    if skipped_multitarget:
        print(
            f"  NOTE: {run_ids_path} has {len(skipped_multitarget)} multi-target "
            f"run(s), skipped (no single-vehicle ground truth to assign): "
            f"{skipped_multitarget}"
        )

    if unmapped:
        raise ValueError(
            f"{run_ids_path}: found label value(s) not in _SCENE_LABEL_ALIASES: "
            f"{sorted(unmapped)}. Add the correct mapping to _SCENE_LABEL_ALIASES "
            f"in utils/labels.py before trusting any downstream result for this scene."
        )
    return result


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


def metadata_row_labels(run_id: str, run_labels: dict[str, "GlobalLabel"] | None = None) -> dict[str, str]:
    """Build global / intermediate / specialized string labels for one segment.

    run_labels: if given, must be the dict returned by load_scene_run_labels()
    for the CURRENT scene -- looks up the real per-scene ground truth. If
    None (legacy default, h24 only), falls back to run_id_to_global's
    hardcoded run-number table. Always pass run_labels explicitly for any
    scene other than h24.
    """
    if run_labels is not None:
        if run_id not in run_labels:
            raise KeyError(f"{run_id} not found in this scene's run_ids.parquet labels")
        global_label = run_labels[run_id]
    else:
        global_label = run_id_to_global(run_id)
    intermediate = global_to_intermediate(global_label)
    specialized = global_label.value if global_label != GlobalLabel.BACKGROUND else ""
    return {
        "global_label": global_label.value,
        "intermediate_label": intermediate.value,
        "specialized_label": specialized,
    }