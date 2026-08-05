"""Generic empirical-outcome collection and validated persistence.

Dataset adapters supply validation/test examples and pretrained predictors;
this module maps every prediction into the shared hierarchy label spaces and
caches the resulting joint per-sample table.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from cascade_profile import HierarchyProfile, profile_from_payload


DEFAULT_OUTPUT_PATH = Path("checkpoints/empirical_outcomes.pkl")


@dataclass(frozen=True)
class EvaluationSplit:
    """One validation or test partition passed to pretrained predictors."""

    name: str
    inputs: Any
    true_labels: Sequence[str]
    sample_ids: Sequence[str | int] | None = None
    metadata: Mapping[str, Sequence[object]] | None = None


@dataclass(frozen=True)
class PredictionBatch:
    """Raw predictions returned by a dataset-specific model adapter."""

    predictions: Sequence[str | int]
    confidence: Sequence[float]


class PredictionFunction(Protocol):
    def __call__(self, split: EvaluationSplit) -> PredictionBatch: ...


@dataclass(frozen=True)
class PretrainedClassifier:
    """Dataset-neutral description of one pretrained cascade candidate."""

    id: str
    kind: str
    predict: PredictionFunction
    output_labels: tuple[str, ...]
    expected_cost_ms: float
    threshold: float | None = None
    group: str | None = None
    wcet_ms: float | None = None
    name: str | None = None
    model_fingerprint: str | None = None


def collection_fingerprint(
    profile: HierarchyProfile,
    splits: Sequence[EvaluationSplit],
    classifiers: Sequence[PretrainedClassifier],
) -> str:
    """Fingerprint inputs that determine a reusable empirical cache."""

    manifest = {
        "profile": profile.fingerprint,
        "splits": [
            {
                "name": split.name,
                "sample_ids": [
                    str(item)
                    for item in (
                        range(len(split.true_labels))
                        if split.sample_ids is None
                        else split.sample_ids
                    )
                ],
                "true_labels": [str(item) for item in split.true_labels],
            }
            for split in splits
        ],
        "classifiers": [
            {
                "id": item.id,
                "kind": item.kind,
                "group": item.group,
                "output_labels": item.output_labels,
                "cost": item.expected_cost_ms,
                "threshold": item.threshold,
                "model_fingerprint": item.model_fingerprint,
            }
            for item in classifiers
        ],
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _prediction_indices(
    values: Sequence[str | int], output_labels: Sequence[str]
) -> np.ndarray:
    labels = tuple(str(item) for item in output_labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    result: list[int] = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            index = int(value)
            if index < 0 or index >= len(labels):
                raise ValueError(f"Prediction index {index} is outside output_labels.")
            result.append(index)
        else:
            try:
                result.append(label_to_index[str(value)])
            except KeyError as exc:
                raise ValueError(f"Unknown prediction label {value!r}.") from exc
    return np.asarray(result, dtype=int)


def _validate_classifier(
    classifier: PretrainedClassifier, profile: HierarchyProfile
) -> None:
    if not classifier.id:
        raise ValueError("Classifier ids must not be empty.")
    if classifier.kind not in {"identifier", "global", "specialized", "detector"}:
        raise ValueError(f"Unknown classifier kind: {classifier.kind!r}")
    if not classifier.output_labels or len(set(classifier.output_labels)) != len(
        classifier.output_labels
    ):
        raise ValueError(f"{classifier.id} output labels must be non-empty and unique.")
    expected_outputs = (
        profile.router_outputs
        if classifier.kind == "identifier"
        else profile.global_classes
    )
    if not set(classifier.output_labels).issubset(expected_outputs):
        raise ValueError(
            f"{classifier.id} outputs are not in the shared label space."
        )
    if classifier.kind == "specialized" and classifier.group not in profile.groups:
        raise ValueError(f"{classifier.id} has unknown group {classifier.group!r}.")
    if classifier.kind == "specialized" and not set(
        classifier.output_labels
    ).issubset(profile.groups[classifier.group]):
        raise ValueError(
            f"{classifier.id} outputs classes outside group {classifier.group!r}."
        )
    if classifier.kind != "specialized" and classifier.group is not None:
        raise ValueError(f"Only specialized classifiers may set group.")
    if classifier.kind != "detector" and classifier.threshold is None:
        raise ValueError(f"{classifier.id} requires a collection threshold.")
    if classifier.threshold is not None and not 0.0 <= float(
        classifier.threshold
    ) <= 1.0:
        raise ValueError(f"{classifier.id} threshold must be between 0 and 1.")
    if not np.isfinite(float(classifier.expected_cost_ms)) or float(
        classifier.expected_cost_ms
    ) < 0.0:
        raise ValueError(
            f"{classifier.id} requires a finite, nonnegative expected cost."
        )


def build_empirical_outcomes(
    profile: HierarchyProfile,
    splits: Sequence[EvaluationSplit],
    classifiers: Sequence[PretrainedClassifier],
) -> dict[str, object]:
    """Evaluate all classifiers on the same rows in every supplied split."""

    if not splits:
        raise ValueError("At least one evaluation split is required.")
    if not classifiers:
        raise ValueError("At least one pretrained classifier is required.")
    if len({split.name for split in splits}) != len(splits):
        raise ValueError("Evaluation split names must be unique.")
    ids = [classifier.id for classifier in classifiers]
    if len(ids) != len(set(ids)):
        raise ValueError("Classifier ids must be unique.")
    detector_candidates = [item for item in classifiers if item.kind == "detector"]
    if len(detector_candidates) != 1:
        raise ValueError("Exactly one detector classifier is required.")
    for classifier in classifiers:
        _validate_classifier(classifier, profile)

    labels_frames: list[pd.DataFrame] = []
    outcome_frames: list[pd.DataFrame] = []
    next_sample_id = 0
    for split in splits:
        if split.name not in {"validation", "test", "holdout"}:
            raise ValueError(
                "Split names must be 'validation', 'test', or 'holdout'."
            )
        true_labels = [str(item) for item in split.true_labels]
        unknown = set(true_labels) - set(profile.global_classes)
        if unknown:
            raise ValueError(f"Unknown true labels in {split.name}: {sorted(unknown)}")
        count = len(true_labels)
        source_ids = (
            list(range(count)) if split.sample_ids is None else list(split.sample_ids)
        )
        if len(source_ids) != count:
            raise ValueError(f"{split.name} sample_ids and labels differ in length.")
        sample_ids = np.arange(next_sample_id, next_sample_id + count, dtype=int)
        next_sample_id += count
        labels_data: dict[str, object] = {
            "sample_id": sample_ids,
            "source_sample_id": [str(item) for item in source_ids],
            "partition": split.name,
            "true_global_label": true_labels,
        }
        for key, values in (split.metadata or {}).items():
            if len(values) != count:
                raise ValueError(f"Metadata column {key!r} has the wrong length.")
            labels_data[str(key)] = list(values)
        labels_frames.append(pd.DataFrame(labels_data))

        for classifier in classifiers:
            prediction = classifier.predict(split)
            confidence = np.asarray(prediction.confidence, dtype=float)
            if len(prediction.predictions) != count or len(confidence) != count:
                raise ValueError(
                    f"{classifier.id} returned the wrong number of {split.name} rows."
                )
            if not np.isfinite(confidence).all():
                raise ValueError(f"{classifier.id} returned non-finite confidence.")
            if ((confidence < 0.0) | (confidence > 1.0)).any():
                raise ValueError(
                    f"{classifier.id} returned confidence outside [0, 1]."
                )
            local_indices = _prediction_indices(
                prediction.predictions, classifier.output_labels
            )
            shared_labels = (
                profile.router_outputs
                if classifier.kind == "identifier"
                else profile.global_classes
            )
            local_to_shared = np.asarray(
                [shared_labels.index(label) for label in classifier.output_labels],
                dtype=int,
            )
            shared_prediction = local_to_shared[local_indices]
            accepted = (
                np.ones(count, dtype=bool)
                if classifier.kind == "detector"
                else confidence >= float(classifier.threshold)
            )
            outcome_frames.append(
                pd.DataFrame(
                    {
                        "sample_id": sample_ids,
                        "candidate_id": classifier.id,
                        "accepted": accepted,
                        "prediction": shared_prediction,
                        "confidence": confidence,
                    }
                )
            )

    candidate_rows = [
        {
            "id": item.id,
            "kind": item.kind,
            "group": item.group,
            "name": item.name or item.id,
            "threshold": item.threshold,
            "cost": float(item.expected_cost_ms),
            "wcet": item.wcet_ms,
            "output_labels": list(item.output_labels),
        }
        for item in classifiers
    ]
    detector = detector_candidates[0]
    payload: dict[str, object] = {
        "schema_version": "empirical-outcomes/v2",
        "profile": profile.as_dict(),
        "collection": {
            "fingerprint": collection_fingerprint(profile, splits, classifiers),
            "model_fingerprints": {
                item.id: item.model_fingerprint for item in classifiers
            },
        },
        "labels": pd.concat(labels_frames, ignore_index=True),
        "candidates": pd.DataFrame(candidate_rows),
        "detector": {
            "id": detector.id,
            "kind": "detector",
            "name": detector.name or detector.id,
            "cost": float(detector.expected_cost_ms),
            "wcet": detector.wcet_ms,
        },
        "outcomes": pd.concat(outcome_frames, ignore_index=True),
    }
    validate_empirical_outcomes(payload)
    return payload


def validate_empirical_outcomes(payload: Mapping[str, object]) -> None:
    profile = profile_from_payload(payload)
    labels = payload.get("labels")
    candidates = payload.get("candidates")
    outcomes = payload.get("outcomes")
    if not isinstance(labels, pd.DataFrame) or not isinstance(
        candidates, pd.DataFrame
    ) or not isinstance(outcomes, pd.DataFrame):
        raise ValueError("labels, candidates, and outcomes must be DataFrames.")
    required_labels = {"sample_id", "true_global_label"}
    required_candidates = {"id", "kind", "group", "threshold", "cost"}
    required_outcomes = {
        "sample_id",
        "candidate_id",
        "accepted",
        "prediction",
        "confidence",
    }
    if missing := required_labels - set(labels.columns):
        raise ValueError(f"labels are missing columns: {sorted(missing)}")
    if missing := required_candidates - set(candidates.columns):
        raise ValueError(f"candidates are missing columns: {sorted(missing)}")
    if missing := required_outcomes - set(outcomes.columns):
        raise ValueError(f"outcomes are missing columns: {sorted(missing)}")
    unknown_labels = set(labels["true_global_label"].astype(str)) - set(
        profile.global_classes
    )
    if unknown_labels:
        raise ValueError(f"Unknown true labels: {sorted(unknown_labels)}")
    if labels["sample_id"].duplicated().any():
        raise ValueError("labels.sample_id values must be unique.")
    if candidates["id"].astype(str).duplicated().any():
        raise ValueError("Candidate ids must be unique.")
    expected_ids = set(candidates["id"].astype(str))
    if set(outcomes["candidate_id"].astype(str)) != expected_ids:
        raise ValueError("Outcome candidate ids do not match candidate metadata.")
    expected_rows = len(labels)
    expected_sample_ids = set(labels["sample_id"])
    if set(outcomes["sample_id"]) != expected_sample_ids:
        raise ValueError("Outcome sample ids do not match label sample ids.")
    if outcomes.duplicated(["candidate_id", "sample_id"]).any():
        raise ValueError("Each classifier must have exactly one outcome per sample.")
    counts = outcomes.groupby("candidate_id")["sample_id"].nunique()
    if len(outcomes) != expected_rows * len(expected_ids) or not (
        counts == expected_rows
    ).all():
        raise ValueError("Every classifier must have one outcome per sample.")


def save_empirical_outcomes(payload: Mapping[str, object], path: str | Path) -> Path:
    validate_empirical_outcomes(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(dict(payload), destination)
    profile_path = destination.with_suffix(".profile.json")
    profile_path.write_text(
        json.dumps(payload["profile"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_empirical_outcomes(path: str | Path = DEFAULT_OUTPUT_PATH) -> dict:
    source = Path(path)
    payload = dict(pd.read_pickle(source))
    if "profile" not in payload:
        profile_path = source.with_suffix(".profile.json")
        if not profile_path.is_file():
            raise ValueError(
                f"Legacy outcomes {source} need profile sidecar {profile_path}."
            )
        payload["profile"] = json.loads(profile_path.read_text(encoding="utf-8"))
        payload.setdefault("schema_version", "empirical-outcomes/v2")
    validate_empirical_outcomes(payload)
    return payload


def ensure_empirical_outcomes(
    path: str | Path,
    profile: HierarchyProfile,
    splits: Sequence[EvaluationSplit],
    classifiers: Sequence[PretrainedClassifier],
    *,
    force: bool = False,
) -> dict:
    """Load a compatible cache or create it from supplied model adapters."""

    destination = Path(path)
    if destination.is_file() and not force:
        payload = load_empirical_outcomes(destination)
        existing = profile_from_payload(payload)
        if existing.fingerprint != profile.fingerprint:
            raise ValueError(
                "Existing empirical outcomes use a different hierarchy profile; "
                "pass force=True or choose another output path."
            )
        collection = payload.get("collection")
        expected_fingerprint = collection_fingerprint(
            profile, splits, classifiers
        )
        if not isinstance(collection, Mapping) or collection.get(
            "fingerprint"
        ) != expected_fingerprint:
            raise ValueError(
                "Existing empirical outcomes do not match the supplied splits "
                "and pretrained-model fingerprints; pass force=True or choose "
                "another output path."
            )
        return payload
    payload = build_empirical_outcomes(profile, splits, classifiers)
    save_empirical_outcomes(payload, destination)
    return payload
