"""Run every Ki (K0-K6) + Kdet over one shared evaluation set and log
per-sample outcomes, so the hierarchy optimizer can build the paper's
empirical joint-probability tables (RTSS 2025, Section III-B) instead of
assuming classifier independence.

Why this exists
----------------
training/trainer.py already runs each Ki over a validation loader and
computes softmax probabilities every epoch (see compute_p_idk, run_epoch).
But it only keeps aggregate metrics (accuracy, confusion matrix) and throws
the per-sample predictions away. It also evaluates each Ki on a *different*
val subset (see utils/splits.py: SUV_VAL_RUNS / COUPE_VAL_RUNS differ from
DEFAULT_VAL_RUNS), so even if predictions were kept, K0's outputs and K4's
outputs would not refer to the same set of rows.

The paper's joint table (Table II / III) requires every classifier's
outcome on the *same* input. This script fixes both gaps:
  1. Evaluates every Ki + Kdet on one shared row set (all rows whose
     run_id is in a held-out split, regardless of which Ki "owns" that
     split during training).
  2. Maps every Ki's raw class index back to a shared label schema so
     outcomes line up across intermediate / global / specialized levels.
  3. Logs (sample_id, candidate_id, accepted, prediction, confidence)
     per row instead of collapsing into accuracy/F1.

Output schema (mirrors empirical_outcomes.py from the friend's ImageNet
repo, so the optimizer code shares the same shape):

    payload = {
        "labels": DataFrame[sample_id, true_global_label, true_intermediate_label],
        "candidates": DataFrame[id, kind, group, name, threshold, cost, wcet],
        "detector": {...},
        "outcomes": DataFrame[sample_id, candidate_id, accepted, prediction, confidence],
    }

`prediction` is an integer index into a SHARED label space per row "kind":
  - intermediate predictions index into INTERMEDIATE_CLASS_NAMES (suv/coupe/background)
  - global / specialized / detector predictions index into GLOBAL_CLASS_NAMES
    (gle350/cx30/mustang/miata/background)
`prediction == -1` means IDK (not accepted).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cascade.loader import load_cascade_models
from training.trainer import KiDataset, load_spectrogram_cache
from utils.classifier_registry import ClassifierRegistry
from utils.labels import (
    GLOBAL_CLASS_NAMES,
    INTERMEDIATE_CLASS_NAMES,
    KI_REGISTRY,
    is_deterministic_ki,
    threshold_hi_for_ki,
)

DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_REGISTRY_PATH = Path("checkpoints/classifier_registry.json")
DEFAULT_OUTPUT_PATH = Path("checkpoints/empirical_outcomes.pkl")


@dataclass(frozen=True)
class CandidateMeta:
    id: str
    kind: str  # "intermediate" | "global" | "specialized_suv" | "specialized_coupe" | "detector"
    name: str
    threshold: float | None
    cost: float
    wcet: float


def _shared_eval_mask(metadata: pd.DataFrame, eval_runs: set[str] | None) -> np.ndarray:
    """Rows used for the shared outcome log.

    Default: union of every split currently used anywhere in utils/splits.py
    (DEFAULT_VAL_RUNS | SUV_VAL_RUNS | COUPE_VAL_RUNS) so background, suv, and
    coupe samples are all represented for every classifier -- including
    classifiers asked to score inputs *outside* their own training subset.
    Per the paper (Sec III-B, footnote 4): "we include all inputs and record
    the true outputs, including those outside Iℓ, to accurately capture the
    classifier's behavior in deployment."
    """
    if eval_runs is None:
        from utils.splits import COUPE_VAL_RUNS, DEFAULT_VAL_RUNS, SUV_VAL_RUNS

        eval_runs = DEFAULT_VAL_RUNS | SUV_VAL_RUNS | COUPE_VAL_RUNS

    run_ids = metadata["run_id"].astype(str)
    return run_ids.isin(eval_runs).to_numpy()


def _build_shared_dataset(
    mic: np.ndarray,
    geo: np.ndarray,
    mask: np.ndarray,
) -> KiDataset:
    """One dataset, both modalities, used for every Ki (each model just reads
    the modality tensor(s) it needs; KiDataset always carries both)."""
    dummy_labels = np.zeros(int(mask.sum()), dtype=np.int64)
    return KiDataset(mic[mask], geo[mask], dummy_labels, modality="both", augment=False)


def _predict_logits(model: torch.nn.Module, batch, modality: str, device: torch.device) -> torch.Tensor:
    # The shared dataset is always built with modality="both" (see
    # _build_shared_dataset), so every batch is a (mic, geo, label) triple
    # regardless of which modality this particular Ki actually needs.
    mic, geo, _ = batch
    if modality == "mic":
        return model(mic.to(device, non_blocking=True))
    return model(mic.to(device, non_blocking=True), geo.to(device, non_blocking=True))


def _map_intermediate(class_idx: np.ndarray, class_names: list[str]) -> np.ndarray:
    """K0/K1 raw class index -> index into INTERMEDIATE_CLASS_NAMES (identity
    today since K0/K1 are already trained on that exact ordering, but kept
    explicit in case class_names order ever changes)."""
    name_to_shared = {name: INTERMEDIATE_CLASS_NAMES.index(name) for name in class_names}
    lookup = np.array([name_to_shared[name] for name in class_names])
    return lookup[class_idx]


def _map_global(class_idx: np.ndarray, class_names: list[str]) -> np.ndarray:
    """K2/K3/K4/K5/K6/Kdet raw class index -> index into GLOBAL_CLASS_NAMES.
    Needed because K4 only has 2 classes (gle350, cx30) and K5/K6 only have
    2 (mustang, miata) -- their local index 0/1 must be remapped to the
    shared 5-way global schema before outcomes can be compared/joined."""
    name_to_shared = {name: GLOBAL_CLASS_NAMES.index(name) for name in class_names}
    lookup = np.array([name_to_shared[name] for name in class_names])
    return lookup[class_idx]


@torch.inference_mode()
def _run_one_classifier(
    ki_name: str,
    model: torch.nn.Module,
    class_names: list[str],
    modality: str,
    threshold: float | None,
    dataset: KiDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (accepted, prediction, confidence) arrays, one row per sample,
    in the SAME row order as `dataset` (i.e. same order for every Ki)."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    accepted_chunks, prediction_chunks, confidence_chunks = [], [], []
    is_intermediate = KI_REGISTRY[ki_name].level == "intermediate"

    for batch in loader:
        logits = _predict_logits(model, batch, modality, device)
        probs = torch.softmax(logits, dim=1)
        confidence, class_idx = probs.max(dim=1)
        confidence = confidence.cpu().numpy()
        class_idx = class_idx.cpu().numpy()

        if is_deterministic_ki(ki_name):
            accepted = np.ones_like(confidence, dtype=bool)  # Kdet never IDKs (paper footnote 1)
        else:
            accepted = confidence >= threshold

        if is_intermediate:
            shared_idx = _map_intermediate(class_idx, class_names)
        else:
            shared_idx = _map_global(class_idx, class_names)

        prediction = np.where(accepted, shared_idx, -1)

        accepted_chunks.append(accepted)
        prediction_chunks.append(prediction)
        confidence_chunks.append(confidence)

    return (
        np.concatenate(accepted_chunks),
        np.concatenate(prediction_chunks),
        np.concatenate(confidence_chunks),
    )


def _candidate_kind(ki_name: str) -> str:
    level = KI_REGISTRY[ki_name].level
    if level == "intermediate":
        return "identifier"
    if level == "global":
        return "global"
    if level.startswith("specialized"):
        return "specialized"
    return "detector"


def _candidate_group(ki_name: str) -> str | None:
    level = KI_REGISTRY[ki_name].level
    if level == "specialized_suv":
        return "suv"
    if level == "specialized_coupe":
        return "coupe"
    return None


def collect_empirical_outcomes(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    eval_runs: set[str] | None = None,
    batch_size: int = 64,
) -> dict:
    """Run K0-K6 + Kdet over one shared eval set and save per-sample outcomes.

    Requires datasets/processed/h24_paired_{mic,geo}.npy + h24_metadata.parquet
    to already exist (run process_data.save_h24_paired_arrays() first), and
    checkpoints/{Ki}.pt + classifier_registry.json to already exist (existing
    trained checkpoints -- no retraining happens here).
    """
    processed_dir = Path(processed_dir)
    checkpoint_dir = Path(checkpoint_dir)
    registry_path = Path(registry_path)

    mic, geo, metadata = load_spectrogram_cache(processed_dir)
    mask = _shared_eval_mask(metadata, eval_runs)
    if mask.sum() == 0:
        raise ValueError(
            "No rows matched the shared eval split. Check utils/splits.py run "
            "ids against h24_metadata.parquet's run_id column."
        )

    dataset = _build_shared_dataset(mic, geo, mask)
    eval_metadata = metadata.loc[mask].reset_index(drop=True)

    models, registry, device = load_cascade_models(checkpoint_dir, registry_path)

    metadata_rows: list[dict] = []
    outcome_frames: list[pd.DataFrame] = []
    sample_ids = list(range(len(eval_metadata)))

    for ki_name, model in models.items():
        spec = KI_REGISTRY[ki_name]
        rec = registry.get(ki_name)
        class_names = rec.class_names if rec is not None else spec.class_names
        threshold = threshold_hi_for_ki(ki_name)

        print(f"Running {ki_name} over {len(sample_ids)} shared rows...")
        accepted, prediction, confidence = _run_one_classifier(
            ki_name, model, class_names, spec.modality, threshold, dataset, device, batch_size
        )

        cost = float(rec.runtime_ms) if rec is not None and rec.runtime_ms is not None else float("nan")
        wcet = float(rec.wcet_ms) if rec is not None and rec.wcet_ms is not None else float("nan")

        metadata_rows.append(
            CandidateMeta(
                id=ki_name,
                kind=_candidate_kind(ki_name),
                name=ki_name,
                threshold=threshold,
                cost=cost,
                wcet=wcet,
            ).__dict__
            | {"group": _candidate_group(ki_name)}
        )
        outcome_frames.append(
            pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "candidate_id": ki_name,
                    "accepted": accepted,
                    "prediction": prediction,
                    "confidence": confidence,
                }
            )
        )

    labels_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "true_global_label": eval_metadata["global_label"].astype(str),
            "true_intermediate_label": eval_metadata["intermediate_label"].astype(str),
            "run_id": eval_metadata["run_id"].astype(str),
        }
    )

    det_rec = registry.get("Kdet")
    detector_meta = {
        "id": "Kdet",
        "kind": "detector",
        "name": "Kdet",
        "cost": float(det_rec.runtime_ms) if det_rec and det_rec.runtime_ms is not None else float("nan"),
        "wcet": float(det_rec.wcet_ms) if det_rec and det_rec.wcet_ms is not None else float("nan"),
        "p_correct": float(det_rec.p_correct) if det_rec else None,
    }

    payload = {
        "labels": labels_df,
        "candidates": pd.DataFrame(metadata_rows),
        "detector": detector_meta,
        "outcomes": pd.concat(outcome_frames, ignore_index=True),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(payload, output_path)
    print(f"Saved empirical outcomes -> {output_path} ({len(sample_ids)} shared rows)")
    return payload


def load_empirical_outcomes(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict:
    return pd.read_pickle(Path(path))


if __name__ == "__main__":
    collect_empirical_outcomes()
