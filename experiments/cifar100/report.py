"""Create reproducible CIFAR-100 candidate reports from empirical artifacts.

The report is deliberately post-hoc: it reads the saved empirical manifest,
training metrics, and latency JSON and never executes a model.  JSON is the
complete machine-readable output; CSV and Markdown provide compact tables for
inspection and comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.cifar100.labels import (
    CIFAR100_PROFILE,
    COARSE_LABEL_NAMES,
    COARSE_NAME_TO_INDEX,
    COARSE_TO_FINE_INDICES,
    FINE_LABEL_NAMES,
    FINE_TO_COARSE_INDEX,
)


REPORT_SCHEMA_VERSION = "cifar100-candidate-report/v1"
LATENCY_SCHEMA_VERSION = "cifar100-latency/v1"
DEFAULT_THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested_path(entry: Mapping[str, object], *keys: str) -> str | None:
    for key in keys:
        value = entry.get(key)
        if value is not None and not isinstance(value, Mapping):
            return str(value)
        nested_name = key.removesuffix("_path")
        nested = entry.get(nested_name)
        if isinstance(nested, Mapping) and nested.get("path") is not None:
            return str(nested["path"])
    return None


def _manifest_candidates(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    raw = manifest.get("candidates")
    if isinstance(raw, Mapping):
        result = []
        for candidate_id, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError("Manifest candidate entries must be objects.")
            entry = dict(value)
            entry.setdefault("candidate_id", str(candidate_id))
            result.append(entry)
        return result
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise ValueError("The empirical manifest requires candidate objects.")
    return [dict(item) for item in raw]


def _candidate_id(entry: Mapping[str, object]) -> str:
    value = entry.get("candidate_id", entry.get("id"))
    nested = entry.get("candidate")
    if value is None and isinstance(nested, Mapping):
        value = nested.get("candidate_id", nested.get("id"))
    if value is None or not str(value):
        raise ValueError("Candidate metadata is missing candidate_id.")
    return str(value)


def _candidate_value(entry: Mapping[str, object], key: str, default: object = None) -> object:
    if key in entry:
        return entry[key]
    nested = entry.get("candidate")
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    return default


def _canonical_role(entry: Mapping[str, object]) -> str:
    role = str(
        _candidate_value(entry, "role", _candidate_value(entry, "kind", ""))
    ).lower()
    if role in {"identifier", "intermediate", "router"}:
        return "intermediate"
    if role in {"specialized", "specialist"}:
        return "specialized"
    if role == "global":
        return "global"
    if role in {"detector", "deterministic"}:
        return "detector"
    raise ValueError(f"Unknown role {role!r} for {_candidate_id(entry)!r}.")


def _load_bundle(manifest: Mapping[str, object], manifest_dir: Path) -> dict[str, object]:
    path_value = _nested_path(
        manifest,
        "outcomes_path",
        "empirical_outcomes_path",
        "bundle_path",
    )
    if path_value is None:
        files = manifest.get("files")
        if isinstance(files, Mapping):
            path_value = _nested_path(
                files,
                "outcomes_path",
                "empirical_outcomes_path",
                "bundle_path",
            )
    if path_value is None:
        raise ValueError("The empirical manifest has no outcomes path.")
    path = _resolve(manifest_dir, path_value)
    declared_sha = manifest.get("outcomes_sha256")
    if declared_sha is not None and str(declared_sha) != _file_sha256(path):
        raise ValueError("Empirical outcome bundle checksum differs from its manifest.")
    if path.suffix.lower() in {".pkl", ".pickle"}:
        value = pd.read_pickle(path)
    else:
        raise ValueError("The empirical bundle must be a pickle mapping.")
    if not isinstance(value, Mapping):
        raise ValueError("The empirical bundle must contain a mapping.")
    bundle = dict(value)
    for name in ("labels", "candidates", "outcomes"):
        if not isinstance(bundle.get(name), pd.DataFrame):
            raise ValueError(f"The empirical bundle has no {name} DataFrame.")
    return bundle


def _merge_candidate_metadata(
    manifest_entries: Sequence[Mapping[str, object]], bundle: Mapping[str, object]
) -> list[dict[str, object]]:
    frame = bundle["candidates"]
    assert isinstance(frame, pd.DataFrame)
    frame_ids = "candidate_id" if "candidate_id" in frame.columns else "id"
    table = {
        str(row[frame_ids]): {
            str(key): value
            for key, value in row.items()
            if not (isinstance(value, float) and math.isnan(value))
        }
        for _, row in frame.iterrows()
    }
    result: list[dict[str, object]] = []
    for raw in manifest_entries:
        candidate_id = _candidate_id(raw)
        merged = dict(table.get(candidate_id, {}))
        merged.update(raw)
        merged["candidate_id"] = candidate_id
        result.append(merged)
    ids = [_candidate_id(item) for item in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Empirical manifest candidate ids must be unique.")
    outcome_ids = set(bundle["outcomes"]["candidate_id"].astype(str))
    missing = set(ids) - outcome_ids
    if missing:
        raise ValueError(f"Candidates are missing empirical rows: {sorted(missing)}")
    return result


def _label_indices(values: Sequence[object], names: Sequence[str], field: str) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(names)}
    result: list[int] = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            index = int(value)
        elif str(value) in lookup:
            index = lookup[str(value)]
        elif str(value).isdigit():
            index = int(str(value))
        else:
            raise ValueError(f"Unknown {field} label {value!r}.")
        if not 0 <= index < len(names):
            raise ValueError(f"{field} index {index} is out of range.")
        result.append(index)
    return np.asarray(result, dtype=np.int64)


def _true_labels(labels: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fine_column = next(
        (
            name
            for name in (
                "true_fine_label",
                "true_global_label",
                "true_fine_label_name",
                "true_fine_index",
                "fine_label",
            )
            if name in labels.columns
        ),
        None,
    )
    if fine_column is None:
        raise ValueError("Labels have no true fine/global label column.")
    fine = _label_indices(labels[fine_column].tolist(), FINE_LABEL_NAMES, "fine")
    coarse_column = next(
        (
            name
            for name in (
                "true_coarse_label",
                "true_coarse_label_name",
                "true_coarse_index",
                "true_coarse",
                "coarse_label",
            )
            if name in labels.columns
        ),
        None,
    )
    if coarse_column is None:
        coarse = np.asarray([FINE_TO_COARSE_INDEX[index] for index in fine], dtype=np.int64)
    else:
        coarse = _label_indices(
            labels[coarse_column].tolist(), COARSE_LABEL_NAMES, "coarse"
        )
    return fine, coarse


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=1, keepdims=True)


def _artifact_description(entry: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("probability_artifact", "probabilities", "prediction_artifact"):
        value = entry.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _load_probability_artifact(
    entry: Mapping[str, object], manifest_dir: Path, expected_samples: int
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[str, ...] | None]:
    description = _artifact_description(entry)
    path_value = _nested_path(
        entry, "probability_path", "probabilities_path", "artifact_path"
    )
    if path_value is None and description.get("path") is not None:
        path_value = str(description["path"])
    if path_value is None:
        return None, None, None
    path = _resolve(manifest_dir, path_value)
    declared_sha = description.get("sha256")
    if declared_sha is not None and str(declared_sha) != _file_sha256(path):
        raise ValueError(f"Probability artifact checksum differs for {path}.")
    with np.load(path, allow_pickle=False) as artifact:
        keys = set(artifact.files)
        probabilities: np.ndarray | None = None
        if "probabilities" in keys:
            probabilities = np.asarray(artifact["probabilities"], dtype=np.float64)
        elif "probs" in keys:
            probabilities = np.asarray(artifact["probs"], dtype=np.float64)
        elif "logits" in keys:
            probabilities = _softmax(np.asarray(artifact["logits"], dtype=np.float64))
        sample_ids = (
            np.asarray(
                artifact[
                    "sample_ids" if "sample_ids" in keys else "sample_id"
                ]
            )
            if "sample_ids" in keys or "sample_id" in keys
            else None
        )
        embedded_labels = (
            tuple(str(value) for value in artifact["output_labels"].tolist())
            if "output_labels" in keys
            else None
        )
    if probabilities is None:
        raise ValueError(f"Probability artifact {path} has no probabilities or logits.")
    if probabilities.ndim != 2 or probabilities.shape[0] != expected_samples:
        raise ValueError(f"Probability artifact {path} has the wrong shape.")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Probability artifact {path} contains non-finite values.")
    row_sums = probabilities.sum(axis=1)
    if (probabilities < -1e-7).any() or not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(f"Probability artifact {path} is not normalized.")
    configured_labels = description.get("output_labels", entry.get("output_labels"))
    labels = (
        tuple(str(value) for value in configured_labels)
        if isinstance(configured_labels, (list, tuple))
        else embedded_labels
    )
    return probabilities, sample_ids, labels


def _align_outcomes(
    labels: pd.DataFrame, outcomes: pd.DataFrame, candidate_id: str
) -> pd.DataFrame:
    rows = outcomes[outcomes["candidate_id"].astype(str) == candidate_id].copy()
    if len(rows) != len(labels) or rows["sample_id"].duplicated().any():
        raise ValueError(f"Candidate {candidate_id!r} does not have one row per sample.")
    order = pd.DataFrame({"sample_id": labels["sample_id"].to_numpy()})
    aligned = order.merge(rows, on="sample_id", how="left", validate="one_to_one")
    if aligned["candidate_id"].isna().any():
        raise ValueError(f"Candidate {candidate_id!r} sample ids do not align.")
    return aligned


def _predictions(
    entry: Mapping[str, object],
    role: str,
    aligned: pd.DataFrame,
    probabilities: np.ndarray | None,
    output_labels: tuple[str, ...] | None,
) -> np.ndarray:
    if role == "intermediate":
        columns = ("predicted_coarse_label", "prediction", "predicted_local_label")
        names = COARSE_LABEL_NAMES
    elif role == "specialized":
        columns = ("predicted_global_label", "prediction", "predicted_local_label")
        names = FINE_LABEL_NAMES
    else:
        columns = ("predicted_global_label", "prediction")
        names = FINE_LABEL_NAMES
    column = next((name for name in columns if name in aligned.columns), None)
    if column is not None:
        predictions = _label_indices(aligned[column].tolist(), names, "prediction")
    elif probabilities is not None:
        local = np.argmax(probabilities, axis=1)
        if output_labels is None:
            predictions = local.astype(np.int64)
        else:
            predictions = _label_indices(
                [output_labels[index] for index in local], names, "prediction"
            )
    else:
        raise ValueError(f"Candidate {_candidate_id(entry)!r} has no predictions.")

    local_specialist_prediction = role == "specialized" and (
        column == "predicted_local_label"
        or (
            column is None
            and probabilities is not None
            and probabilities.shape[1] == 5
            and output_labels is None
        )
    )
    if local_specialist_prediction:
        group = str(_candidate_value(entry, "group"))
        if group not in COARSE_NAME_TO_INDEX:
            raise ValueError(f"Specialist {_candidate_id(entry)!r} has unknown group.")
        group_indices = COARSE_TO_FINE_INDICES[COARSE_NAME_TO_INDEX[group]]
        if (predictions < 0).any() or (predictions >= len(group_indices)).any():
            raise ValueError("Specialist local predictions are outside [0, 4].")
        predictions = np.asarray([group_indices[index] for index in predictions])
    return predictions


def _confidence(
    aligned: pd.DataFrame, probabilities: np.ndarray | None
) -> np.ndarray:
    column = next(
        (
            name
            for name in ("max_softmax_probability", "confidence")
            if name in aligned.columns
        ),
        None,
    )
    confidence = (
        aligned[column].to_numpy(dtype=np.float64)
        if column is not None
        else None
    )
    if confidence is None and probabilities is not None:
        confidence = probabilities.max(axis=1)
    if confidence is None:
        raise ValueError("Empirical rows have no confidence values.")
    if not np.isfinite(confidence).all() or ((confidence < 0) | (confidence > 1)).any():
        raise ValueError("Confidence values must be finite and inside [0, 1].")
    return confidence


def _selective_metrics(
    confidence: np.ndarray,
    correct: np.ndarray,
    thresholds: Sequence[float],
    population: np.ndarray | None = None,
) -> list[dict[str, object]]:
    mask = np.ones(len(confidence), dtype=bool) if population is None else population
    denominator = int(mask.sum())
    result: list[dict[str, object]] = []
    for threshold in thresholds:
        accepted = mask & (confidence >= threshold)
        count = int(accepted.sum())
        result.append(
            {
                "threshold": float(threshold),
                "population": denominator,
                "accepted": count,
                "coverage": float(count / denominator) if denominator else None,
                "selective_accuracy": (
                    float(correct[accepted].mean()) if count else None
                ),
                "correct_coverage": (
                    float((accepted & correct).sum() / denominator)
                    if denominator
                    else None
                ),
            }
        )
    return result


def _deep_get(data: Mapping[str, object], paths: Sequence[Sequence[str]]) -> object | None:
    for path in paths:
        current: object = data
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                break
            current = current[key]
        else:
            return current
    return None


def _load_training_metrics(
    entry: Mapping[str, object], manifest_dir: Path
) -> Mapping[str, object]:
    embedded = entry.get("training_metrics")
    if isinstance(embedded, Mapping):
        if embedded.get("path") is not None:
            if embedded.get("sha256") is None:
                raise ValueError("Training-metrics artifacts require a SHA-256.")
            artifact_path = _resolve(manifest_dir, str(embedded["path"]))
            if str(embedded["sha256"]) != _file_sha256(artifact_path):
                raise ValueError("Training-metrics artifact checksum differs.")
            return _load_json(artifact_path)
        return embedded
    path_value = _nested_path(entry, "metrics_path", "training_metrics_path")
    if path_value is None:
        checkpoint_value = _nested_path(entry, "checkpoint_path")
        if checkpoint_value is None:
            return {}
        adjacent = _resolve(manifest_dir, checkpoint_value).with_name("metrics.json")
        return _load_json(adjacent) if adjacent.is_file() else {}
    return _load_json(_resolve(manifest_dir, path_value))


def _accuracy_fraction(value: object | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    if 1.0 < result <= 100.0:
        result /= 100.0
    return result if 0.0 <= result <= 1.0 else None


def _training_summary(
    entry: Mapping[str, object], metrics: Mapping[str, object]
) -> dict[str, object]:
    accuracy = _deep_get(
        metrics,
        (
            ("best_model_selection_accuracy",),
            ("model_selection_accuracy",),
            ("best_validation_accuracy",),
            ("best_val_accuracy",),
            ("best", "accuracy"),
            ("best", "model_selection_accuracy"),
            ("summary", "best_accuracy"),
        ),
    )
    parameters = _deep_get(
        metrics,
        (("parameter_count",), ("model", "parameter_count"), ("summary", "parameters")),
    )
    if parameters is None:
        parameters = _candidate_value(entry, "parameter_count")
    input_resolution = _deep_get(
        metrics,
        (("input_resolution",), ("config", "input_resolution")),
    )
    if input_resolution is None:
        input_resolution = _candidate_value(entry, "input_resolution", [32, 32])
    if isinstance(input_resolution, (list, tuple)) and len(input_resolution) == 3:
        input_resolution = list(input_resolution[-2:])
    return {
        "model_selection_accuracy": _accuracy_fraction(accuracy),
        "parameter_count": None if parameters is None else int(parameters),
        "input_resolution": input_resolution,
    }


def _latency_lookup(payload: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = payload.get("candidates")
    if isinstance(raw, Mapping):
        result: dict[str, Mapping[str, object]] = {}
        for key, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError("Latency candidate results must be objects.")
            candidate_id = str(value.get("candidate_id", value.get("id", key)))
            if candidate_id in result:
                raise ValueError(f"Duplicate latency candidate id: {candidate_id!r}")
            result[candidate_id] = value
        return result
    if not isinstance(raw, list):
        raise ValueError("Latency JSON requires candidate results.")
    result: dict[str, Mapping[str, object]] = {}
    for value in raw:
        if not isinstance(value, Mapping):
            raise ValueError("Latency candidate results must be objects.")
        candidate_id = _candidate_id(value)
        if candidate_id in result:
            raise ValueError(f"Duplicate latency candidate id: {candidate_id!r}")
        result[candidate_id] = value
    return result


def _verify_latency_identity(
    entry: Mapping[str, object], latency: Mapping[str, object] | None
) -> None:
    candidate_id = _candidate_id(entry)
    if latency is None:
        raise ValueError(f"Latency results are missing {candidate_id!r}.")
    if _candidate_id(latency) != candidate_id:
        raise ValueError(f"Latency candidate id differs for {candidate_id!r}.")
    checkpoint = entry.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Candidate {candidate_id!r} has no checkpoint identity.")
    for key in ("sha256", "config_hash"):
        expected = checkpoint.get(key)
        latency_key = "checkpoint_sha256" if key == "sha256" else key
        measured = latency.get(latency_key)
        if expected is None or measured is None:
            raise ValueError(
                f"Candidate {candidate_id!r} is missing {latency_key} identity data."
            )
        if str(expected) != str(measured):
            raise ValueError(
                f"Latency {latency_key} differs for candidate {candidate_id!r}."
            )


def _latency_summary(value: Mapping[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    stats = value.get("latency_ms", value)
    if not isinstance(stats, Mapping):
        return None
    keys = ("mean", "median", "std", "p95", "p99")
    if not all(key in stats for key in keys):
        return None
    return {key: float(stats[key]) for key in keys}


def _candidate_report(
    entry: Mapping[str, object],
    labels: pd.DataFrame,
    outcomes: pd.DataFrame,
    true_fine: np.ndarray,
    true_coarse: np.ndarray,
    manifest_dir: Path,
    latency: Mapping[str, object] | None,
    thresholds: Sequence[float],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    candidate_id = _candidate_id(entry)
    _verify_latency_identity(entry, latency)
    role = _canonical_role(entry)
    aligned = _align_outcomes(labels, outcomes, candidate_id)
    probabilities, artifact_ids, output_labels = _load_probability_artifact(
        entry, manifest_dir, len(labels)
    )
    if artifact_ids is not None:
        expected_ids = labels["sample_id"].to_numpy()
        if not np.array_equal(artifact_ids.astype(str), expected_ids.astype(str)):
            raise ValueError(f"Candidate {candidate_id!r} artifact sample order differs.")
    predictions = _predictions(entry, role, aligned, probabilities, output_labels)
    confidence = _confidence(aligned, probabilities)
    if role == "intermediate":
        correct = predictions == true_coarse
        role_population = np.ones(len(labels), dtype=bool)
    elif role == "specialized":
        group_name = str(_candidate_value(entry, "group"))
        if group_name not in COARSE_NAME_TO_INDEX:
            raise ValueError(f"Specialist {candidate_id!r} has unknown group {group_name!r}.")
        role_population = true_coarse == COARSE_NAME_TO_INDEX[group_name]
        correct = predictions == true_fine
    else:
        correct = predictions == true_fine
        role_population = np.ones(len(labels), dtype=bool)

    population_count = int(role_population.sum())
    role_accuracy = (
        float(correct[role_population].mean()) if population_count else None
    )
    training = _training_summary(entry, _load_training_metrics(entry, manifest_dir))
    report: dict[str, object] = {
        "candidate_id": candidate_id,
        "role": role,
        "group": _candidate_value(entry, "group"),
        "architecture": _candidate_value(entry, "architecture"),
        **training,
        "cascade_validation_samples": population_count,
        "cascade_validation_accuracy": role_accuracy,
        "all_sample_accuracy": float(correct.mean()),
        "confidence": {
            "mean": float(confidence.mean()),
            "median": float(np.median(confidence)),
        },
        "selective": _selective_metrics(
            confidence, correct, thresholds, role_population
        ),
        "latency_ms": _latency_summary(latency),
    }
    if role == "intermediate":
        report["routing_precision_vs_coverage"] = report["selective"]
    elif role == "specialized":
        out_group = ~role_population
        all_sample_selective = _selective_metrics(
            confidence,
            correct,
            thresholds,
        )
        report["specialist_behavior"] = {
            "in_group_samples": int(role_population.sum()),
            "out_group_samples": int(out_group.sum()),
            "in_group_accuracy": role_accuracy,
            "in_group_mean_confidence": (
                float(confidence[role_population].mean())
                if role_population.any()
                else None
            ),
            "out_group_mean_confidence": (
                float(confidence[out_group].mean()) if out_group.any() else None
            ),
            "in_group_selective": report["selective"],
            "all_sample_selective": all_sample_selective,
            "out_group_acceptance": _selective_metrics(
                confidence,
                correct,
                thresholds,
                out_group,
            ),
        }
    state = {
        "correct": correct.astype(bool),
        "confidence": confidence,
        "prediction": predictions,
    }
    return report, state


def _global_complementarity(
    reports: Sequence[Mapping[str, object]],
    states: Mapping[str, Mapping[str, np.ndarray]],
    threshold: float,
) -> list[dict[str, object]]:
    globals_ = [item for item in reports if item["role"] == "global"]
    result: list[dict[str, object]] = []
    for left_index, left in enumerate(globals_):
        for right in globals_[left_index + 1 :]:
            left_id = str(left["candidate_id"])
            right_id = str(right["candidate_id"])
            left_state = states[left_id]
            right_state = states[right_id]
            left_correct = left_state["correct"]
            right_correct = right_state["correct"]
            left_accepted = left_state["confidence"] >= threshold
            right_accepted = right_state["confidence"] >= threshold
            left_latency = left.get("latency_ms")
            right_latency = right.get("latency_ms")
            left_mean = (
                float(left_latency["mean"])
                if isinstance(left_latency, Mapping)
                else None
            )
            right_mean = (
                float(right_latency["mean"])
                if isinstance(right_latency, Mapping)
                else None
            )
            left_accuracy = float(left["cascade_validation_accuracy"])
            right_accuracy = float(right["cascade_validation_accuracy"])
            left_dominates = (
                left_mean is not None
                and right_mean is not None
                and left_accuracy >= right_accuracy
                and left_mean <= right_mean
                and (left_accuracy > right_accuracy or left_mean < right_mean)
            )
            right_dominates = (
                left_mean is not None
                and right_mean is not None
                and right_accuracy >= left_accuracy
                and right_mean <= left_mean
                and (right_accuracy > left_accuracy or right_mean < left_mean)
            )
            result.append(
                {
                    "candidate_a": left_id,
                    "candidate_b": right_id,
                    "both_correct": float((left_correct & right_correct).mean()),
                    "only_a_correct": float((left_correct & ~right_correct).mean()),
                    "only_b_correct": float((right_correct & ~left_correct).mean()),
                    "neither_correct": float((~left_correct & ~right_correct).mean()),
                    "confidence_threshold": float(threshold),
                    "a_accepted_coverage": float(left_accepted.mean()),
                    "b_accepted_coverage": float(right_accepted.mean()),
                    "a_accepted_correct_coverage": float(
                        (left_accepted & left_correct).mean()
                    ),
                    "b_accepted_correct_coverage": float(
                        (right_accepted & right_correct).mean()
                    ),
                    "a_accepted_complementary_correct": float(
                        (left_accepted & left_correct & ~right_correct).mean()
                    ),
                    "b_accepted_complementary_correct": float(
                        (right_accepted & right_correct & ~left_correct).mean()
                    ),
                    "a_accepted_error": float((left_accepted & ~left_correct).mean()),
                    "b_accepted_error": float((right_accepted & ~right_correct).mean()),
                    "a_dominates_b_accuracy_latency": left_dominates,
                    "b_dominates_a_accuracy_latency": right_dominates,
                }
            )
    return result


def _dominated_candidates(
    reports: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Identify same-role candidates weakly dominated on accuracy and mean cost."""

    result: list[dict[str, object]] = []
    for candidate in reports:
        candidate_latency = candidate.get("latency_ms")
        if not isinstance(candidate_latency, Mapping):
            continue
        candidate_accuracy_value = candidate.get("cascade_validation_accuracy")
        if candidate_accuracy_value is None:
            continue
        candidate_cost = float(candidate_latency["mean"])
        candidate_accuracy = float(candidate_accuracy_value)
        for alternative in reports:
            if candidate is alternative or candidate["role"] != alternative["role"]:
                continue
            if candidate["role"] == "specialized" and candidate.get(
                "group"
            ) != alternative.get("group"):
                continue
            alternative_latency = alternative.get("latency_ms")
            if not isinstance(alternative_latency, Mapping):
                continue
            alternative_accuracy_value = alternative.get(
                "cascade_validation_accuracy"
            )
            if alternative_accuracy_value is None:
                continue
            alternative_cost = float(alternative_latency["mean"])
            alternative_accuracy = float(alternative_accuracy_value)
            if (
                alternative_accuracy >= candidate_accuracy
                and alternative_cost <= candidate_cost
                and (
                    alternative_accuracy > candidate_accuracy
                    or alternative_cost < candidate_cost
                )
            ):
                result.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "dominated_by": alternative["candidate_id"],
                        "accuracy_delta": alternative_accuracy - candidate_accuracy,
                        "mean_latency_delta_ms": alternative_cost - candidate_cost,
                    }
                )
                break
    return result


def _safe_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _format_float(value: object, digits: int = 4) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def _summary_rows(reports: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result = []
    for item in reports:
        latency = item.get("latency_ms")
        row = {
            "candidate_id": item["candidate_id"],
            "role": item["role"],
            "group": item.get("group"),
            "architecture": item.get("architecture"),
            "parameter_count": item.get("parameter_count"),
            "input_resolution": json.dumps(item.get("input_resolution")),
            "model_selection_accuracy": item.get("model_selection_accuracy"),
            "cascade_validation_accuracy": item.get("cascade_validation_accuracy"),
            "all_sample_accuracy": item.get("all_sample_accuracy"),
            "latency_mean_ms": (
                latency.get("mean") if isinstance(latency, Mapping) else None
            ),
            "latency_median_ms": (
                latency.get("median") if isinstance(latency, Mapping) else None
            ),
            "latency_std_ms": (
                latency.get("std") if isinstance(latency, Mapping) else None
            ),
            "latency_p95_ms": (
                latency.get("p95") if isinstance(latency, Mapping) else None
            ),
            "latency_p99_ms": (
                latency.get("p99") if isinstance(latency, Mapping) else None
            ),
        }
        for metric in item["selective"]:
            suffix = f"{float(metric['threshold']):.3f}".rstrip("0").rstrip(".")
            row[f"coverage_at_{suffix}"] = metric["coverage"]
            row[f"selective_accuracy_at_{suffix}"] = metric["selective_accuracy"]
        behavior = item.get("specialist_behavior")
        if isinstance(behavior, Mapping):
            row["in_group_accuracy"] = behavior["in_group_accuracy"]
            row["in_group_mean_confidence"] = behavior["in_group_mean_confidence"]
            row["out_group_mean_confidence"] = behavior["out_group_mean_confidence"]
            for metric in behavior["all_sample_selective"]:
                suffix = f"{float(metric['threshold']):.3f}".rstrip("0").rstrip(".")
                row[f"all_sample_coverage_at_{suffix}"] = metric["coverage"]
                row[f"all_sample_selective_accuracy_at_{suffix}"] = metric[
                    "selective_accuracy"
                ]
            for metric in behavior["out_group_acceptance"]:
                suffix = f"{float(metric['threshold']):.3f}".rstrip("0").rstrip(".")
                row[f"out_group_coverage_at_{suffix}"] = metric["coverage"]
        result.append(row)
    return result


def _selective_rows(reports: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for item in reports:
        for metric in item["selective"]:
            result.append(
                {
                    "candidate_id": item["candidate_id"],
                    "role": item["role"],
                    "group": item.get("group"),
                    "population_kind": "role",
                    **metric,
                }
            )
        behavior = item.get("specialist_behavior")
        if isinstance(behavior, Mapping):
            for metric in behavior["all_sample_selective"]:
                result.append(
                    {
                        "candidate_id": item["candidate_id"],
                        "role": item["role"],
                        "group": item.get("group"),
                        "population_kind": "all_samples",
                        **metric,
                    }
                )
            for metric in behavior["out_group_acceptance"]:
                result.append(
                    {
                        "candidate_id": item["candidate_id"],
                        "role": item["role"],
                        "group": item.get("group"),
                        "population_kind": "out_group",
                        **metric,
                    }
                )
    return result


def _markdown(
    reports: Sequence[Mapping[str, object]],
    pairs: Sequence[Mapping[str, object]],
    dominated: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# CIFAR-100 candidate report",
        "",
        "## Candidate summary",
        "",
        "| Candidate | Role | Group | Parameters | Selection accuracy | Cascade accuracy | Mean ms | p95 ms |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in reports:
        latency = item.get("latency_ms")
        lines.append(
            "| {candidate} | {role} | {group} | {parameters} | {selection} | "
            "{cascade} | {mean} | {p95} |".format(
                candidate=item["candidate_id"],
                role=item["role"],
                group=item.get("group") or "",
                parameters=item.get("parameter_count") or "",
                selection=_format_float(item.get("model_selection_accuracy")),
                cascade=_format_float(item.get("cascade_validation_accuracy")),
                mean=_format_float(
                    latency.get("mean") if isinstance(latency, Mapping) else None,
                    3,
                ),
                p95=_format_float(
                    latency.get("p95") if isinstance(latency, Mapping) else None,
                    3,
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Coverage and selective accuracy",
            "",
            "| Candidate | Population | Threshold | Coverage | Selective accuracy | Correct coverage |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in reports:
        population_label = "in group" if item["role"] == "specialized" else "role"
        for metric in item["selective"]:
            lines.append(
                f"| {item['candidate_id']} | {population_label} | "
                f"{_format_float(metric['threshold'], 2)} | "
                f"{_format_float(metric['coverage'])} | "
                f"{_format_float(metric['selective_accuracy'])} | "
                f"{_format_float(metric['correct_coverage'])} |"
            )
        behavior = item.get("specialist_behavior")
        if isinstance(behavior, Mapping):
            for metric in behavior["all_sample_selective"]:
                lines.append(
                    f"| {item['candidate_id']} | all samples | "
                    f"{_format_float(metric['threshold'], 2)} | "
                    f"{_format_float(metric['coverage'])} | "
                    f"{_format_float(metric['selective_accuracy'])} | "
                    f"{_format_float(metric['correct_coverage'])} |"
                )
    specialists = [
        item for item in reports if isinstance(item.get("specialist_behavior"), Mapping)
    ]
    if specialists:
        lines.extend(
            [
                "",
                "## Specialist in-group and out-group behavior",
                "",
                "| Candidate | Group | In-group accuracy | In-group mean confidence | Out-group mean confidence |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for item in specialists:
            behavior = item["specialist_behavior"]
            lines.append(
                f"| {item['candidate_id']} | {item.get('group') or ''} | "
                f"{_format_float(behavior['in_group_accuracy'])} | "
                f"{_format_float(behavior['in_group_mean_confidence'])} | "
                f"{_format_float(behavior['out_group_mean_confidence'])} |"
            )
    if pairs:
        lines.extend(
            [
                "",
                "## Global-model complementarity",
                "",
                "| Candidate A | Candidate B | Only A correct | Only B correct | A accepted complementary | B accepted complementary |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for pair in pairs:
            lines.append(
                f"| {pair['candidate_a']} | {pair['candidate_b']} | "
                f"{_format_float(pair['only_a_correct'])} | "
                f"{_format_float(pair['only_b_correct'])} | "
                f"{_format_float(pair['a_accepted_complementary_correct'])} | "
                f"{_format_float(pair['b_accepted_complementary_correct'])} |"
            )
    if dominated:
        lines.extend(
            [
                "",
                "## Accuracy-latency dominance flags",
                "",
                "| Candidate | Dominated by | Accuracy delta | Mean latency delta (ms) |",
                "|---|---|---:|---:|",
            ]
        )
        for item in dominated:
            lines.append(
                f"| {item['candidate_id']} | {item['dominated_by']} | "
                f"{_format_float(item['accuracy_delta'])} | "
                f"{_format_float(item['mean_latency_delta_ms'], 3)} |"
            )
    return "\n".join(lines) + "\n"


def generate_report(
    empirical_manifest_path: str | Path,
    latency_path: str | Path,
    output_dir: str | Path,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    complementarity_threshold: float = 0.9,
) -> dict[str, object]:
    """Generate JSON, CSV, and Markdown tables from saved empirical data."""

    threshold_values = tuple(float(value) for value in thresholds)
    if not threshold_values or any(not 0.0 <= value <= 1.0 for value in threshold_values):
        raise ValueError("Thresholds must be a non-empty sequence inside [0, 1].")
    if not 0.0 <= float(complementarity_threshold) <= 1.0:
        raise ValueError("complementarity_threshold must be inside [0, 1].")
    manifest_path = Path(empirical_manifest_path).resolve()
    manifest = _load_json(manifest_path)
    schema = manifest.get("schema_version")
    if schema not in {"cifar100-empirical-manifest/v1", "empirical-manifest/v1"}:
        raise ValueError(f"Unsupported empirical manifest schema: {schema!r}")
    dataset_id = manifest.get("dataset_id")
    if dataset_id is not None and str(dataset_id) != CIFAR100_PROFILE.dataset_id:
        raise ValueError(f"Unexpected empirical dataset id: {dataset_id!r}")
    profile_fingerprint = manifest.get("profile_fingerprint")
    if (
        profile_fingerprint is not None
        and str(profile_fingerprint) != CIFAR100_PROFILE.fingerprint
    ):
        raise ValueError("Empirical manifest hierarchy fingerprint does not match.")
    if manifest.get("official_test_used") is not False:
        raise ValueError("Empirical manifest does not affirm held-back data isolation.")
    if manifest.get("detector_status") not in {"external_pending", "available"}:
        raise ValueError("Empirical manifest has an invalid detector status.")
    bundle = _load_bundle(manifest, manifest_path.parent)
    entries = _merge_candidate_metadata(_manifest_candidates(manifest), bundle)
    labels = bundle["labels"]
    outcomes = bundle["outcomes"]
    assert isinstance(labels, pd.DataFrame)
    assert isinstance(outcomes, pd.DataFrame)
    if "sample_id" not in labels.columns or labels["sample_id"].duplicated().any():
        raise ValueError("Empirical labels require unique sample ids.")
    true_fine, true_coarse = _true_labels(labels)
    resolved_latency_path = Path(latency_path).resolve()
    declared_latency = manifest.get("latency_results")
    if schema == "cifar100-empirical-manifest/v1" and not isinstance(
        declared_latency, Mapping
    ):
        raise ValueError("Empirical manifest has no checksummed latency artifact.")
    if isinstance(declared_latency, Mapping):
        declared_latency_sha = declared_latency.get("sha256")
        if declared_latency_sha != _file_sha256(resolved_latency_path):
            raise ValueError("Latency artifact checksum differs from the manifest.")
    latency_payload = _load_json(resolved_latency_path)
    if latency_payload.get("schema_version") != LATENCY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported latency schema: {latency_payload.get('schema_version')!r}"
        )
    latency_by_id = _latency_lookup(latency_payload)

    candidate_reports: list[dict[str, object]] = []
    states: dict[str, Mapping[str, np.ndarray]] = {}
    for entry in entries:
        candidate_id = _candidate_id(entry)
        report, state = _candidate_report(
            entry,
            labels,
            outcomes,
            true_fine,
            true_coarse,
            manifest_path.parent,
            latency_by_id.get(candidate_id),
            threshold_values,
        )
        candidate_reports.append(report)
        states[candidate_id] = state
    pairs = _global_complementarity(
        candidate_reports, states, float(complementarity_threshold)
    )
    dominated = _dominated_candidates(candidate_reports)
    payload: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": CIFAR100_PROFILE.dataset_id,
            "profile_fingerprint": CIFAR100_PROFILE.fingerprint,
            "empirical_samples": len(labels),
        },
        "sources": {
            "empirical_manifest": str(manifest_path),
            "latency": str(resolved_latency_path),
        },
        "thresholds": list(threshold_values),
        "candidates": candidate_reports,
        "global_complementarity": pairs,
        "dominated_candidates": dominated,
    }
    payload = _safe_json(payload)  # type: ignore[assignment]
    assert isinstance(payload, dict)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "candidate_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(_summary_rows(candidate_reports)).to_csv(
        destination / "candidate_summary.csv", index=False
    )
    pd.DataFrame(_selective_rows(candidate_reports)).to_csv(
        destination / "selective_metrics.csv", index=False
    )
    pd.DataFrame(pairs).to_csv(
        destination / "global_complementarity.csv", index=False
    )
    pd.DataFrame(dominated).to_csv(
        destination / "dominated_candidates.csv", index=False
    )
    (destination / "candidate_report.md").write_text(
        _markdown(candidate_reports, pairs, dominated), encoding="utf-8"
    )
    return payload


def _parse_thresholds(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--empirical-manifest", type=Path, required=True)
    parser.add_argument("--latency", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=DEFAULT_THRESHOLDS,
        help="Comma-separated confidence thresholds.",
    )
    parser.add_argument("--complementarity-threshold", type=float, default=0.9)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = generate_report(
        args.empirical_manifest,
        args.latency,
        args.output_dir,
        thresholds=args.thresholds,
        complementarity_threshold=args.complementarity_threshold,
    )
    print(f"Wrote report for {len(payload['candidates'])} candidates to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_THRESHOLDS",
    "REPORT_SCHEMA_VERSION",
    "generate_report",
]
