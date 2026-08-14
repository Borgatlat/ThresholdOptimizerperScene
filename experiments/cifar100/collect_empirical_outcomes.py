"""Collect rich CIFAR-100 empirical outcomes, including an optional endpoint.

Every independently saved non-deterministic candidate is evaluated on the
same cascade-validation rows.  The compact pickle columns are directly
compatible with the generic threshold replay format, while per-candidate NPZ
files preserve full local logits/probabilities for later analysis.

This module never loads the official CIFAR-100 test split and never creates a
perfect-detector placeholder.  When the measured ConvNeXt endpoint is present
in the training manifest, its real predictions and cost are merged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from empirical_outcomes import save_empirical_outcomes
from experiments.cifar100.data import (
    build_convnext_evaluation_transform,
    build_dataset_view,
    load_split_bundle,
    load_training_dataset,
)
from experiments.cifar100.labels import (
    CIFAR100_PROFILE,
    COARSE_LABEL_NAMES,
    FINE_LABEL_NAMES,
)
from experiments.cifar100.models import (
    CandidateSpec,
    candidate_specs,
    file_sha256,
    load_checkpoint,
)


MANIFEST_SCHEMA_VERSION = "cifar100-empirical-manifest/v1"
TRAINING_MANIFEST_SCHEMA_VERSION = "cifar100-training-manifest/v1"
LATENCY_SCHEMA_VERSION = "cifar100-latency/v1"
DEFAULT_DATA_ROOT = Path("datasets/cifar100")
DEFAULT_SPLIT_NPZ = Path("checkpoints/cifar100/splits/cifar100_split_indices.npz")
DEFAULT_TRAINING_MANIFEST = Path(
    "checkpoints/cifar100/training/training_manifest.json"
)
DEFAULT_LATENCY_PATH = Path("checkpoints/cifar100/latency.json")
DEFAULT_OUTPUT_DIR = Path("checkpoints/cifar100/empirical")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_id(record: Mapping[str, object]) -> str:
    nested = record.get("candidate")
    value = record.get("candidate_id", record.get("id"))
    if value is None and isinstance(nested, Mapping):
        value = nested.get("candidate_id", nested.get("id"))
    if value is None:
        raise ValueError("Training manifest candidate has no id.")
    return str(value)


def _manifest_records(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    values = manifest.get("candidates")
    if not isinstance(values, list) or not all(
        isinstance(item, Mapping) for item in values
    ):
        raise ValueError("Training manifest requires a candidates list.")
    records = [dict(item) for item in values]
    if len({_candidate_id(item) for item in records}) != len(records):
        raise ValueError("Training manifest candidate ids must be unique.")
    return records


def _manifest_detector(manifest: Mapping[str, object]) -> dict[str, object] | None:
    value = manifest.get("detector")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Training manifest detector must be an object or null.")
    return dict(value)


def _latency_by_id(payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    values = payload.get("candidates")
    if not isinstance(values, list) or not all(
        isinstance(item, Mapping) for item in values
    ):
        raise ValueError("Latency JSON requires a candidates list.")
    result: dict[str, Mapping[str, object]] = {}
    for item in values:
        candidate_id = _candidate_id(item)
        if candidate_id in result:
            raise ValueError(f"Duplicate latency candidate id {candidate_id!r}.")
        result[candidate_id] = item
    return result


def _require_hash(
    payload: Mapping[str, object], key: str, path: Path, *, source: str
) -> None:
    declared = payload.get(key)
    measured = file_sha256(path)
    if declared != measured:
        raise ValueError(
            f"{source} {key} does not match {path}; expected {measured!r}, "
            f"received {declared!r}."
        )


def _validate_training_manifest(
    manifest: Mapping[str, object],
    *,
    split_npz: Path,
    split_manifest: Path,
) -> None:
    expected = {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "dataset_id": CIFAR100_PROFILE.dataset_id,
        "profile_fingerprint": CIFAR100_PROFILE.fingerprint,
        "official_test_used": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Training manifest has invalid {key!r}.")
    _require_hash(
        manifest,
        "split_indices_sha256",
        split_npz,
        source="Training manifest",
    )
    _require_hash(
        manifest,
        "split_manifest_sha256",
        split_manifest,
        source="Training manifest",
    )


def _validate_latency_manifest(
    payload: Mapping[str, object],
    *,
    training_manifest_path: Path,
    split_npz: Path,
    split_manifest: Path,
) -> None:
    if payload.get("schema_version") != LATENCY_SCHEMA_VERSION:
        raise ValueError("Unsupported CIFAR-100 latency schema.")
    source = payload.get("source_manifest")
    if not isinstance(source, Mapping):
        raise ValueError("Latency JSON has no source training manifest.")
    if source.get("sha256") != file_sha256(training_manifest_path):
        raise ValueError("Latency results were measured from another training manifest.")
    inputs = payload.get("input")
    if not isinstance(inputs, Mapping):
        raise ValueError("Latency JSON has no input provenance.")
    if inputs.get("source") != "official_training_cascade_validation":
        raise ValueError("Empirical collection requires real cascade-validation latency.")
    _require_hash(
        inputs, "split_indices_sha256", split_npz, source="Latency input"
    )
    _require_hash(
        inputs,
        "split_manifest_sha256",
        split_manifest,
        source="Latency input",
    )


def _resolve_record_path(
    record: Mapping[str, object], key: str, *, manifest_dir: Path
) -> Path:
    value = record.get(key)
    if value is None:
        nested = record.get(key.removesuffix("_path"))
        if isinstance(nested, Mapping):
            value = nested.get("path")
    if value is None:
        raise ValueError(f"Candidate {_candidate_id(record)!r} has no {key}.")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _softmax_statistics(logits: np.ndarray) -> tuple[np.ndarray, ...]:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    local_prediction = probabilities.argmax(axis=1).astype(np.int64)
    confidence = probabilities[np.arange(len(probabilities)), local_prediction]
    if probabilities.shape[1] > 1:
        top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
        margin = top_two.max(axis=1) - top_two.min(axis=1)
    else:
        margin = confidence.copy()
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    )
    return (
        probabilities.astype(np.float32),
        local_prediction,
        confidence.astype(np.float32),
        entropy.astype(np.float32),
        margin.astype(np.float32),
    )


@torch.inference_mode()
def infer_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    """Infer one independently loaded checkpoint over a stable row order."""

    model = model.to(device)
    model.eval()
    chunks: list[np.ndarray] = []
    for inputs, _ in loader:
        logits = model(inputs.to(device, non_blocking=True))
        if logits.ndim != 2:
            raise ValueError("Every classifier must return [batch, classes] logits.")
        chunks.append(logits.detach().cpu().to(torch.float32).numpy())
    if not chunks:
        raise ValueError("Cascade-validation inference received no samples.")
    return np.concatenate(chunks, axis=0)


def build_candidate_outcomes(
    *,
    spec: CandidateSpec,
    logits: np.ndarray,
    true_fine: np.ndarray,
    true_coarse: np.ndarray,
    sample_ids: np.ndarray,
    checkpoint_sha256: str,
    checkpoint_config_hash: str,
    artifact_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Map local outputs into optimizer spaces and save the lossless sidecar."""

    if logits.shape != (len(sample_ids), spec.num_classes):
        raise ValueError(
            f"{spec.candidate_id} logits shape {logits.shape} does not match "
            f"({len(sample_ids)}, {spec.num_classes})."
        )
    probabilities, local_prediction, confidence, entropy, margin = (
        _softmax_statistics(logits)
    )
    output_labels = tuple(spec.output_labels)
    if spec.kind == "identifier":
        local_to_shared = np.asarray(
            [CIFAR100_PROFILE.router_index[label] for label in output_labels],
            dtype=np.int64,
        )
        shared_prediction = local_to_shared[local_prediction]
        global_prediction = np.full(len(sample_ids), -1, dtype=np.int64)
        role_correct = shared_prediction == true_coarse
        in_group = np.ones(len(sample_ids), dtype=bool)
    else:
        local_to_global = np.asarray(
            [CIFAR100_PROFILE.global_index[label] for label in output_labels],
            dtype=np.int64,
        )
        global_prediction = local_to_global[local_prediction]
        shared_prediction = global_prediction
        if spec.kind == "specialized":
            group_index = CIFAR100_PROFILE.router_index[str(spec.group)]
            in_group = true_coarse == group_index
            role_correct = in_group & (global_prediction == true_fine)
        else:
            in_group = np.ones(len(sample_ids), dtype=bool)
            role_correct = global_prediction == true_fine

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        logits=logits.astype(np.float32),
        probabilities=probabilities,
        sample_id=sample_ids.astype(np.int64),
        output_labels=np.asarray(output_labels),
    )
    artifact = {
        "path": str(artifact_path.resolve()),
        "sha256": file_sha256(artifact_path),
        "dtype": "float32",
        "logits_shape": list(logits.shape),
        "probabilities_shape": list(probabilities.shape),
        "output_labels": list(output_labels),
    }
    frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "candidate_id": spec.candidate_id,
            # A neutral accept-all collection threshold preserves raw behavior;
            # occurrence-specific IDK thresholds are chosen later.
            "accepted": np.ones(len(sample_ids), dtype=bool),
            "prediction": shared_prediction,
            "confidence": confidence,
            "predicted_local_label": local_prediction,
            "predicted_global_label": global_prediction,
            "entropy": entropy,
            "top2_margin": margin,
            "role_correct": role_correct,
            "in_specialist_group": in_group,
            "checkpoint_id": checkpoint_sha256,
            "config_hash": checkpoint_config_hash,
        }
    )
    return frame, artifact


def _latency_values(
    candidate_id: str,
    latency: Mapping[str, Mapping[str, object]],
) -> tuple[float, float, Mapping[str, object]]:
    if candidate_id not in latency:
        raise ValueError(f"Latency results are missing {candidate_id!r}.")
    entry = latency[candidate_id]
    values = entry.get("latency_ms")
    if not isinstance(values, Mapping):
        raise ValueError(f"Latency result {candidate_id!r} has no latency_ms.")
    expected = float(values["mean"])
    wcet = float(values.get("max", values.get("p99", expected)))
    if not np.isfinite((expected, wcet)).all() or min(expected, wcet) < 0.0:
        raise ValueError(f"Latency result {candidate_id!r} is invalid.")
    return expected, wcet, entry


def collect_empirical_outcomes(
    *,
    data_root: Path,
    split_npz: Path,
    training_manifest_path: Path,
    latency_path: Path,
    output_dir: Path,
    batch_size: int = 128,
    detector_batch_size: int = 16,
    num_workers: int = 0,
    device: torch.device = torch.device("cpu"),
    candidate_ids: Sequence[str] | None = None,
    max_samples: int | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if batch_size < 1 or detector_batch_size < 1:
        raise ValueError("Inference batch sizes must be positive.")
    split_npz = split_npz.resolve()
    dataset = load_training_dataset(data_root, download=False)
    split_manifest_path = split_npz.with_suffix(".json")
    splits = load_split_bundle(
        split_npz,
        manifest_path=split_manifest_path,
        fine_targets=dataset.fine_targets,
        coarse_targets=dataset.coarse_targets,
    )
    source_indices = splits.cascade_validation
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("max_samples must be positive.")
        source_indices = source_indices[:max_samples]
    sample_ids = np.arange(len(source_indices), dtype=np.int64)
    true_fine = dataset.fine_targets[source_indices]
    true_coarse = dataset.coarse_targets[source_indices]
    view = build_dataset_view(
        dataset, source_indices, target_mode="fine", train=False
    )
    loader = DataLoader(
        view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    detector_view = build_dataset_view(
        dataset,
        source_indices,
        target_mode="fine",
        transform=build_convnext_evaluation_transform(),
    )
    detector_loader = DataLoader(
        detector_view,
        batch_size=detector_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    training_manifest_path = training_manifest_path.resolve()
    training_manifest = json.loads(
        training_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(training_manifest, Mapping):
        raise ValueError("Training manifest must contain a JSON object.")
    _validate_training_manifest(
        training_manifest,
        split_npz=split_npz,
        split_manifest=split_manifest_path,
    )
    nondeterministic_records = _manifest_records(training_manifest)
    detector_record = _manifest_detector(training_manifest)
    records = [*nondeterministic_records]
    if detector_record is not None:
        records.append(detector_record)
    if candidate_ids is not None:
        selected = set(candidate_ids)
        records = [item for item in records if _candidate_id(item) in selected]
        missing = selected - {_candidate_id(item) for item in records}
        if missing:
            raise ValueError(f"Unknown requested candidates: {sorted(missing)}")
    if not records:
        raise ValueError("No candidate checkpoints were selected.")
    latency_path = latency_path.resolve()
    latency_payload = json.loads(latency_path.read_text(encoding="utf-8"))
    if not isinstance(latency_payload, Mapping):
        raise ValueError("Latency results must contain a JSON object.")
    _validate_latency_manifest(
        latency_payload,
        training_manifest_path=training_manifest_path,
        split_npz=split_npz,
        split_manifest=split_manifest_path,
    )
    latency = _latency_by_id(latency_payload)
    registry = candidate_specs(
        CIFAR100_PROFILE.groups, CIFAR100_PROFILE.global_classes
    )
    expected_candidate_ids = {item.candidate_id for item in registry.candidates}
    selected_candidate_ids = {
        _candidate_id(item) for item in nondeterministic_records
    }
    if candidate_ids is None and selected_candidate_ids != expected_candidate_ids:
        missing = sorted(expected_candidate_ids - selected_candidate_ids)
        extra = sorted(selected_candidate_ids - expected_candidate_ids)
        raise ValueError(
            "A full empirical run requires the exact 24 non-deterministic "
            f"candidates (missing={missing}, extra={extra}); use --candidate only "
            "for an explicit smoke run."
        )
    if max_samples is None and len(source_indices) != 5_000:
        raise ValueError(
            "A full empirical run requires all 5,000 cascade-validation rows."
        )
    artifact_dir = output_dir / "raw_outputs"
    outcome_frames: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, object]] = []
    manifest_candidates: list[dict[str, object]] = []

    for record in records:
        candidate_id = _candidate_id(record)
        try:
            spec = registry.by_id[candidate_id]
        except KeyError as exc:
            raise ValueError(f"Unknown CIFAR candidate {candidate_id!r}.") from exc
        if not spec.is_candidate:
            raise ValueError(f"{candidate_id!r} is an initializer, not a candidate.")
        checkpoint_path = _resolve_record_path(
            record, "checkpoint_path", manifest_dir=training_manifest_path.parent
        )
        declared_sha = record.get("checkpoint_sha256")
        measured_sha = file_sha256(checkpoint_path)
        if declared_sha is not None and str(declared_sha) != measured_sha:
            raise ValueError(f"Checkpoint checksum mismatch for {candidate_id!r}.")
        model, checkpoint_metadata = load_checkpoint(
            checkpoint_path, expected_spec=spec, map_location="cpu"
        )
        checkpoint_training = checkpoint_metadata.get("training_config")
        if (
            not isinstance(checkpoint_training, Mapping)
            or checkpoint_training.get("split_manifest_sha256")
            != file_sha256(split_manifest_path)
            or checkpoint_training.get("official_test_used") is not False
            or checkpoint_training.get("cascade_validation_used_for_selection")
            is not False
        ):
            raise ValueError(
                f"Checkpoint {candidate_id!r} has incompatible split/isolation "
                "provenance."
            )
        checkpoint_config_hash = str(checkpoint_metadata["config_hash"])
        if record.get("config_hash") != checkpoint_config_hash:
            raise ValueError(
                f"Training manifest/config hash mismatch for {candidate_id!r}."
            )
        expected_cost, wcet, latency_entry = _latency_values(candidate_id, latency)
        if latency_entry.get("checkpoint_sha256") != measured_sha:
            raise ValueError(
                f"Latency/checkpoint hash mismatch for {candidate_id!r}."
            )
        if latency_entry.get("config_hash") != checkpoint_config_hash:
            raise ValueError(f"Latency/config hash mismatch for {candidate_id!r}.")
        metrics_path = _resolve_record_path(
            record, "metrics_path", manifest_dir=training_manifest_path.parent
        )
        training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(training_metrics, Mapping):
            raise ValueError(f"Training metrics are invalid for {candidate_id!r}.")
        if training_metrics.get("candidate") != spec.as_dict():
            raise ValueError(f"Training metrics candidate mismatch for {candidate_id!r}.")
        if training_metrics.get("checkpoint_sha256") != measured_sha:
            raise ValueError(f"Training metrics hash mismatch for {candidate_id!r}.")
        if training_metrics.get("config_hash") != checkpoint_config_hash:
            raise ValueError(f"Training metrics config mismatch for {candidate_id!r}.")

        candidate_loader = (
            detector_loader if spec.kind == "detector" else loader
        )
        logits = infer_logits(model, candidate_loader, device=device)
        artifact_path = artifact_dir / f"{candidate_id}.npz"
        outcome_frame, artifact = build_candidate_outcomes(
            spec=spec,
            logits=logits,
            true_fine=true_fine,
            true_coarse=true_coarse,
            sample_ids=sample_ids,
            checkpoint_sha256=measured_sha,
            checkpoint_config_hash=checkpoint_config_hash,
            artifact_path=artifact_path,
        )
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        input_resolution = (
            [3, 224, 224]
            if spec.architecture == "convnextv2_large"
            else [3, 32, 32]
        )
        candidate_rows.append(
            {
                "id": candidate_id,
                "kind": spec.kind,
                "role": spec.role,
                "group": spec.group,
                "name": candidate_id,
                "threshold": 0.0,
                "cost": expected_cost,
                "wcet": wcet,
                "output_labels": list(spec.output_labels),
                "checkpoint_id": measured_sha,
                "checkpoint_path": str(checkpoint_path),
                "config_hash": checkpoint_config_hash,
                "parameter_count": parameter_count,
                "input_resolution": input_resolution,
                "probability_artifact": str(artifact_path.resolve()),
            }
        )
        manifest_candidates.append(
            {
                **spec.as_dict(),
                "checkpoint": {
                    "path": str(checkpoint_path),
                    "sha256": measured_sha,
                    "config_hash": checkpoint_config_hash,
                },
                "training_metrics": {
                    "path": str(metrics_path),
                    "sha256": file_sha256(metrics_path),
                },
                "parameter_count": parameter_count,
                "input_resolution": input_resolution,
                "probability_artifact": artifact,
                "latency": dict(latency_entry),
            }
        )
        outcome_frames.append(outcome_frame)
        del model

    labels = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "source_sample_id": source_indices.astype(np.int64),
            "partition": "validation",
            "true_global_label": [FINE_LABEL_NAMES[index] for index in true_fine],
            "true_fine_index": true_fine,
            "true_fine_label": [FINE_LABEL_NAMES[index] for index in true_fine],
            "true_fine_name": [FINE_LABEL_NAMES[index] for index in true_fine],
            "true_coarse_index": true_coarse,
            "true_coarse_name": [
                COARSE_LABEL_NAMES[index] for index in true_coarse
            ],
        }
    )
    collection_identity = {
        "profile_fingerprint": CIFAR100_PROFILE.fingerprint,
        "split_manifest_sha256": file_sha256(split_manifest_path),
        "source_sample_ids": source_indices.tolist(),
        "checkpoints": {
            item["candidate_id"]: item["checkpoint"]["sha256"]
            for item in manifest_candidates
        },
    }
    detector_manifest_entry = next(
        (
            item
            for item in manifest_candidates
            if item["kind"] == "detector"
        ),
        None,
    )
    detector_status = (
        "available" if detector_manifest_entry is not None else "external_pending"
    )
    detector_metadata = None
    if detector_manifest_entry is not None:
        detector_row = next(
            item for item in candidate_rows if item["kind"] == "detector"
        )
        detector_metadata = {
            "id": detector_row["id"],
            "kind": "detector",
            "name": detector_row["name"],
            "cost": detector_row["cost"],
            "wcet": detector_row["wcet"],
            "checkpoint_id": detector_row["checkpoint_id"],
        }

    payload: dict[str, object] = {
        "schema_version": "empirical-outcomes/v2",
        "profile": CIFAR100_PROFILE.as_dict(),
        "collection": {
            "fingerprint": _canonical_hash(collection_identity),
            "model_fingerprints": {
                item["candidate_id"]: item["checkpoint"]["sha256"]
                for item in manifest_candidates
            },
        },
        "detector_status": detector_status,
        "detector": detector_metadata,
        "labels": labels,
        "candidates": pd.DataFrame(candidate_rows),
        "outcomes": pd.concat(outcome_frames, ignore_index=True),
        "artifacts": {
            item["candidate_id"]: item["probability_artifact"]
            for item in manifest_candidates
        },
        "dataset": {
            "source_partition": "official_train",
            "empirical_partition": "cascade_validation",
            "official_test_used": False,
            "split_indices": str(split_npz.resolve()),
            "split_manifest": str(split_manifest_path.resolve()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes_path = output_dir / "empirical_outcomes.pkl"
    manifest_path = output_dir / "manifest.json"
    save_empirical_outcomes(payload, outcomes_path)
    manifest: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_id": CIFAR100_PROFILE.dataset_id,
        "profile": CIFAR100_PROFILE.as_dict(),
        "profile_fingerprint": CIFAR100_PROFILE.fingerprint,
        "detector_status": detector_status,
        "detector": detector_metadata,
        "official_test_used": False,
        "collection_threshold": 0.0,
        "sample_count": len(labels),
        "candidate_count": len(candidate_rows),
        "nondeterministic_candidate_count": len(
            [item for item in candidate_rows if item["kind"] != "detector"]
        ),
        "outcomes_path": str(outcomes_path.resolve()),
        "outcomes_sha256": file_sha256(outcomes_path),
        "split_indices": str(split_npz.resolve()),
        "split_manifest": str(split_manifest_path.resolve()),
        "training_manifest": {
            "path": str(training_manifest_path),
            "sha256": file_sha256(training_manifest_path),
        },
        "latency_results": {
            "path": str(latency_path),
            "sha256": file_sha256(latency_path),
        },
        "label_mappings": {
            "fine": list(FINE_LABEL_NAMES),
            "coarse": list(COARSE_LABEL_NAMES),
            "groups": {
                group: list(labels)
                for group, labels in CIFAR100_PROFILE.groups.items()
            },
        },
        "candidates": manifest_candidates,
    }
    manifest["manifest_content_hash"] = _canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload, manifest


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-npz", type=Path, default=DEFAULT_SPLIT_NPZ)
    parser.add_argument(
        "--training-manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST
    )
    parser.add_argument("--latency", type=Path, default=DEFAULT_LATENCY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--detector-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--candidate", nargs="+")
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Smoke-only cap; omit for the required 5,000 rows.",
    )
    args = parser.parse_args()
    payload, manifest = collect_empirical_outcomes(
        data_root=args.data_root,
        split_npz=args.split_npz,
        training_manifest_path=args.training_manifest,
        latency_path=args.latency,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        detector_batch_size=args.detector_batch_size,
        num_workers=args.num_workers,
        device=_resolve_device(args.device),
        candidate_ids=args.candidate,
        max_samples=args.max_samples,
    )
    print(
        json.dumps(
            {
                "samples": len(payload["labels"]),
                "candidates": len(payload["candidates"]),
                "manifest": str(args.output_dir / "manifest.json"),
                "detector_status": manifest["detector_status"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
