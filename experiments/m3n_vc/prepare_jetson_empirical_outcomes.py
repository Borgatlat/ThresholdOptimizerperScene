"""Rewrite a cached empirical-outcomes packet with Jetson runtime metadata.

This keeps the per-sample predictions intact, swaps the candidate costs to the
Jetson-profiled runtime_ms / wcet_ms values, and optionally turns Kdet into the
paper's perfect 10,000 ms oracle detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from cascade_profile import profile_from_payload
from empirical_outcomes import load_empirical_outcomes, save_empirical_outcomes
from experiments.m3n_vc.checkpoint_paths import (
    file_fingerprint,
    resolve_registry_checkpoint,
)
from experiments.m3n_vc.utils.classifier_registry import ClassifierRegistry
from experiments.m3n_vc.utils.labels import GLOBAL_CLASS_NAMES, threshold_hi_for_ki


DEFAULT_SOURCE_OUTCOMES = Path("checkpoints/empirical_outcomes_h24_with_run9.pkl")
DEFAULT_REGISTRY_PATH = Path("checkpoints/classifier_registry_jetson_nano.json")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_OUTPUT_PATH = Path("checkpoints/empirical_outcomes_h24_jetson_nano.pkl")
DEFAULT_PAPER_DETECTOR_COST_MS = 10_000.0
MODEL_IDS = tuple(f"K{index}" for index in range(7))
EXPECTED_RUNS = {"run1", "run3", "run5", "run7", "run9"}


def _fingerprints_match(
    expected: Mapping[str, object], actual: Mapping[str, object]
) -> bool:
    try:
        return (
            str(expected.get("sha256")) == str(actual.get("sha256"))
            and int(expected.get("size_bytes", -1))
            == int(actual.get("size_bytes", -1))
        )
    except (TypeError, ValueError):
        return False


def _finite_nonnegative(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_costs(
    registry: ClassifierRegistry,
    *,
    paper_detector: bool,
    paper_detector_cost_ms: float,
) -> dict[str, dict[str, float]]:
    paper_detector_cost_ms = _finite_nonnegative(
        paper_detector_cost_ms, "Paper detector cost"
    )
    costs: dict[str, dict[str, float]] = {}
    registry_rows = registry.to_dataframe()
    for candidate_id in registry_rows["name"].astype(str):
        record = registry.get(candidate_id)
        if record is None:
            continue
        if candidate_id == "Kdet" and paper_detector:
            costs[candidate_id] = {
                "cost": float(paper_detector_cost_ms),
                "wcet": float(paper_detector_cost_ms),
            }
            continue
        costs[candidate_id] = {
            "cost": _finite_nonnegative(
                record.runtime_ms, f"{candidate_id} runtime_ms"
            ),
            "wcet": _finite_nonnegative(
                record.wcet_ms, f"{candidate_id} wcet_ms"
            ),
        }
    return costs


def _update_outcomes_for_paper_detector(
    payload: dict[str, object],
    *,
    paper_detector_cost_ms: float,
) -> None:
    labels = payload["labels"]
    outcomes = payload["outcomes"]
    detector = payload.get("detector")
    if not isinstance(labels, pd.DataFrame) or not isinstance(outcomes, pd.DataFrame):
        raise ValueError("labels and outcomes must be DataFrames.")

    true_labels = labels.set_index("sample_id")["true_global_label"].astype(str)
    global_index = {
        name: index
        for index, name in enumerate(payload["profile"]["global_classes"])  # type: ignore[index]
    }
    detector_mask = outcomes["candidate_id"].astype(str) == "Kdet"
    if detector_mask.any():
        sample_ids = outcomes.loc[detector_mask, "sample_id"]
        rewired = sample_ids.map(true_labels).map(global_index)
        if rewired.isna().any():
            raise ValueError("Paper detector encountered an unknown true label.")
        outcomes.loc[detector_mask, "accepted"] = True
        outcomes.loc[detector_mask, "prediction"] = rewired.astype(int).to_numpy()
        outcomes.loc[detector_mask, "confidence"] = 1.0

    payload["detector"] = {
        "id": "Kdet",
        "kind": "detector",
        "name": "Kdet",
        "cost": float(paper_detector_cost_ms),
        "wcet": float(paper_detector_cost_ms),
        "p_correct": 1.0,
    }


def prepare_jetson_empirical_outcomes(
    source_outcomes: str | Path = DEFAULT_SOURCE_OUTCOMES,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    paper_detector: bool = True,
    paper_detector_cost_ms: float = DEFAULT_PAPER_DETECTOR_COST_MS,
    overwrite: bool = False,
) -> dict[str, object]:
    """Rewrite a cached outcomes packet so the optimizer sees Jetson costs."""

    source_outcomes = Path(source_outcomes).expanduser().resolve()
    registry_path = Path(registry_path).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if source_outcomes == output_path:
        raise ValueError("Source and output paths must differ.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    payload = load_empirical_outcomes(source_outcomes)
    profile = profile_from_payload(payload)
    if tuple(profile.global_classes) != tuple(GLOBAL_CLASS_NAMES):
        raise ValueError(
            "Source global-class order does not match the M3N-VC H24 models."
        )
    registry = ClassifierRegistry.load(registry_path)
    candidates = payload.get("candidates")
    labels = payload.get("labels")
    if not isinstance(candidates, pd.DataFrame) or not isinstance(
        labels, pd.DataFrame
    ):
        raise ValueError("Empirical outcomes are missing candidates or labels.")
    if "run_id" not in labels.columns:
        raise ValueError("Source labels have no run_id column.")
    run_ids = set(labels["run_id"].astype(str))
    if run_ids != EXPECTED_RUNS:
        raise ValueError(
            f"Expected H24 runs {sorted(EXPECTED_RUNS)}, found {sorted(run_ids)}."
        )

    costs = _candidate_costs(
        registry,
        paper_detector=paper_detector,
        paper_detector_cost_ms=paper_detector_cost_ms,
    )
    candidate_ids = candidates["id"].astype(str)
    expected_candidates = {*MODEL_IDS, "Kdet"}
    if set(candidate_ids) != expected_candidates:
        raise ValueError(
            f"Expected candidates {sorted(expected_candidates)}, "
            f"found {sorted(set(candidate_ids))}."
        )
    missing = sorted(set(candidate_ids) - set(costs))
    if missing:
        raise ValueError(
            "Jetson registry is missing runtime metadata for: " + ", ".join(missing)
        )

    candidates = candidates.copy(deep=True)
    candidates["id"] = candidate_ids
    candidates["cost"] = candidate_ids.map(
        lambda candidate_id: costs[candidate_id]["cost"]
    ).astype(float)
    if "wcet" not in candidates.columns:
        candidates["wcet"] = np.nan
    candidates["wcet"] = candidate_ids.map(
        lambda candidate_id: costs[candidate_id]["wcet"]
    ).astype(float)
    for model_id in MODEL_IDS:
        expected_threshold = threshold_hi_for_ki(model_id)
        actual_threshold = float(
            candidates.loc[candidates["id"] == model_id, "threshold"].iloc[0]
        )
        if expected_threshold is None or not math.isclose(
            actual_threshold,
            float(expected_threshold),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"Source {model_id} threshold does not match its canonical threshold."
            )
    payload["candidates"] = candidates

    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    runtime_profile = registry_payload.get("runtime_profile")
    profiled_checkpoints = (
        runtime_profile.get("model_checkpoints")
        if isinstance(runtime_profile, Mapping)
        else None
    )
    if not isinstance(profiled_checkpoints, Mapping):
        raise ValueError(
            "Jetson registry has no checkpoint fingerprints. Run "
            "profile_models_jetson.py from the JetsonNanoBench branch first."
        )

    model_checkpoints: dict[str, dict[str, str | int]] = {}
    for candidate_id in MODEL_IDS:
        record = registry.get(candidate_id)
        if record is None:
            raise ValueError(f"Jetson registry has no {candidate_id} record.")
        checkpoint_path = resolve_registry_checkpoint(
            record.checkpoint,
            candidate_id,
            checkpoint_dir,
            registry_path=registry_path,
        )
        current = file_fingerprint(checkpoint_path)
        profiled = profiled_checkpoints.get(candidate_id)
        if not isinstance(profiled, Mapping) or not _fingerprints_match(
            profiled, current
        ):
            raise ValueError(
                f"{candidate_id} checkpoint differs from the Jetson runtime profile. "
                "Re-run profile_models_jetson.py with the current weights."
            )
        model_checkpoints[candidate_id] = current

    source_collection = payload.get("collection")
    collection = (
        dict(source_collection) if isinstance(source_collection, Mapping) else {}
    )
    runtime_manifest = {
        "schema_version": "jetson-empirical-outcomes/v1",
        "source_outcomes": str(source_outcomes),
        "source_outcomes_sha256": _sha256(source_outcomes),
        "source_registry": str(registry_path),
        "source_registry_sha256": _sha256(registry_path),
        "paper_detector": bool(paper_detector),
        "paper_detector_cost_ms": float(paper_detector_cost_ms),
        "candidate_costs_ms": {
            candidate_id: {
                "cost": costs[candidate_id]["cost"],
                "wcet": costs[candidate_id]["wcet"],
            }
            for candidate_id in candidate_ids
        },
        "model_checkpoints": model_checkpoints,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_inference_performed": False,
    }
    runtime_manifest["conversion_sha256"] = hashlib.sha256(
        json.dumps(runtime_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    collection.update(
        {
            "registry": str(registry_path),
            "registry_sha256": runtime_manifest["source_registry_sha256"],
            "checkpoint_dir": str(checkpoint_dir),
            "model_checkpoints": model_checkpoints,
            "paper_detector": bool(paper_detector),
            "detector_behavior": (
                "perfect_oracle_non_sleeping" if paper_detector else collection.get("detector_behavior")
            ),
            "runtime_profile": runtime_manifest,
            "scene": "h24",
            "eval_runs": sorted(run_ids),
            "cascade_partition_protocol": {
                "optimization": "first 80% within each of run1,run3,run5,run7,run9",
                "testing": "last 20% within each run",
                "split_strategy": "blocked_per_run",
                "holdout_fraction": 0.20,
            },
            "device": "reused_cached_predictions",
        }
    )
    payload["collection"] = collection
    payload["detector_status"] = "available"

    if paper_detector:
        _update_outcomes_for_paper_detector(
            payload, paper_detector_cost_ms=paper_detector_cost_ms
        )
    else:
        detector = payload.get("detector")
        if isinstance(detector, dict):
            detector["cost"] = float(costs["Kdet"]["cost"])
            detector["wcet"] = float(costs["Kdet"]["wcet"])

    save_empirical_outcomes(payload, output_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a cached empirical-outcomes packet for Jetson runtime costs."
    )
    parser.add_argument(
        "--source-outcomes", type=Path, default=DEFAULT_SOURCE_OUTCOMES
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--paper-detector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite Kdet as the paper's perfect 10,000 ms detector.",
    )
    parser.add_argument(
        "--paper-detector-cost-ms",
        type=float,
        default=DEFAULT_PAPER_DETECTOR_COST_MS,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.paper_detector_cost_ms < 0:
        parser.error("--paper-detector-cost-ms must be non-negative.")

    payload = prepare_jetson_empirical_outcomes(
        args.source_outcomes,
        registry_path=args.registry,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_path,
        paper_detector=args.paper_detector,
        paper_detector_cost_ms=args.paper_detector_cost_ms,
        overwrite=args.overwrite,
    )
    candidates = payload["candidates"].set_index("id")
    print(
        json.dumps(
            {
                "status": "complete",
                "source": str(args.source_outcomes),
                "output": str(args.output_path),
                "samples": int(len(payload["labels"])),
                "model_inference_performed": False,
                "candidate_cost_ms": {
                    candidate_id: float(candidates.loc[candidate_id, "cost"])
                    for candidate_id in candidates.index.astype(str)
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
