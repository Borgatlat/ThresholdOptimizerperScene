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
    if len(detector_candidates) > 1:
        raise ValueError("At most one detector classifier is allowed.")
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
    detector = detector_candidates[0] if detector_candidates else None
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
        "detector_status": "available" if detector is not None else "external_pending",
        "detector": (
            {
                "id": detector.id,
                "kind": "detector",
                "name": detector.name or detector.id,
                "cost": float(detector.expected_cost_ms),
                "wcet": detector.wcet_ms,
            }
            if detector is not None
            else None
        ),
        "outcomes": pd.concat(outcome_frames, ignore_index=True),
    }
    validate_empirical_outcomes(payload)
    return payload


def _is_missing_scalar(value: object) -> bool:
    """Return whether one metadata cell is null without treating arrays as cells."""

    missing = pd.isna(value)
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _validated_candidate_metadata(
    candidates: pd.DataFrame, profile: HierarchyProfile
) -> dict[str, dict[str, object]]:
    """Validate candidate roles and resolve their legal shared predictions."""

    allowed_kinds = {"identifier", "global", "specialized", "detector"}
    has_output_labels = "output_labels" in candidates.columns
    result: dict[str, dict[str, object]] = {}
    for row in candidates.to_dict(orient="records"):
        raw_id = row["id"]
        if _is_missing_scalar(raw_id) or not str(raw_id):
            raise ValueError("Candidate ids must not be empty.")
        candidate_id = str(raw_id)
        kind = str(row["kind"])
        if kind not in allowed_kinds:
            raise ValueError(
                f"Candidate {candidate_id!r} has unknown kind {kind!r}."
            )
        if "role" in candidates.columns:
            raw_role = row["role"]
            role = None if _is_missing_scalar(raw_role) else str(raw_role)
            expected_role = {
                "identifier": "intermediate",
                "global": "global",
                "specialized": "specialized",
                "detector": "detector",
            }[kind]
            if role != expected_role:
                raise ValueError(
                    f"Candidate {candidate_id!r} role {role!r} does not match "
                    f"kind {kind!r}; expected {expected_role!r}."
                )

        raw_group = row["group"]
        group = None if _is_missing_scalar(raw_group) else str(raw_group)
        if kind == "specialized":
            if group not in profile.groups:
                raise ValueError(
                    f"Specialized candidate {candidate_id!r} has unknown group "
                    f"{group!r}."
                )
        elif group is not None:
            raise ValueError(
                f"Only specialized candidates may set group; {candidate_id!r} "
                f"sets {group!r}."
            )

        raw_threshold = row["threshold"]
        threshold = None if _is_missing_scalar(raw_threshold) else float(raw_threshold)
        if kind != "detector" and threshold is None:
            raise ValueError(
                f"Candidate {candidate_id!r} requires a confidence threshold."
            )
        if threshold is not None and (
            not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError(
                f"Candidate {candidate_id!r} threshold must be within [0, 1]."
            )

        try:
            cost = float(row["cost"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Candidate {candidate_id!r} has a nonnumeric cost."
            ) from exc
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError(
                f"Candidate {candidate_id!r} requires a finite, nonnegative cost."
            )

        shared_labels = (
            profile.router_outputs if kind == "identifier" else profile.global_classes
        )
        permitted_labels = (
            profile.groups[group] if kind == "specialized" else shared_labels
        )
        if has_output_labels:
            raw_outputs = row["output_labels"]
            if isinstance(raw_outputs, (str, bytes)) or not isinstance(
                raw_outputs, (list, tuple, np.ndarray, pd.Index)
            ):
                raise ValueError(
                    f"Candidate {candidate_id!r} output_labels must be a sequence."
                )
            if any(_is_missing_scalar(item) for item in raw_outputs):
                raise ValueError(
                    f"Candidate {candidate_id!r} output_labels contain null values."
                )
            output_labels = tuple(str(item) for item in raw_outputs)
            if not output_labels or len(set(output_labels)) != len(output_labels):
                raise ValueError(
                    f"Candidate {candidate_id!r} output_labels must be non-empty "
                    "and unique."
                )
            unknown_outputs = set(output_labels) - set(permitted_labels)
            if unknown_outputs:
                raise ValueError(
                    f"Candidate {candidate_id!r} outputs labels outside its "
                    f"{kind} label space: {sorted(unknown_outputs)}"
                )
        else:
            # Original M3N-VC packets predate per-candidate output mappings.
            # Their predictions are already indices in the shared role space.
            output_labels = tuple(permitted_labels)

        shared_index = {label: index for index, label in enumerate(shared_labels)}
        result[candidate_id] = {
            "kind": kind,
            "threshold": threshold,
            "allowed_predictions": frozenset(
                shared_index[label] for label in output_labels
            ),
        }
    return result


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
    if candidates.empty:
        raise ValueError("At least one candidate is required.")
    if candidates["id"].astype(str).duplicated().any():
        raise ValueError("Candidate ids must be unique.")
    candidate_metadata = _validated_candidate_metadata(candidates, profile)
    detector_rows = candidates[candidates["kind"].astype(str) == "detector"]
    if len(detector_rows) > 1:
        raise ValueError("At most one detector candidate is allowed.")
    detector = payload.get("detector")
    detector_status = str(
        payload.get(
            "detector_status",
            "available" if isinstance(detector, Mapping) else "external_pending",
        )
    )
    if detector_status not in {"available", "external_pending"}:
        raise ValueError(f"Unknown detector_status: {detector_status!r}")
    if detector_status == "external_pending":
        if detector is not None or len(detector_rows):
            raise ValueError(
                "external_pending outcomes must not contain detector metadata or rows."
            )
    else:
        if not isinstance(detector, Mapping) or len(detector_rows) != 1:
            raise ValueError(
                "available detector outcomes require exactly one detector candidate."
            )
        if str(detector.get("id")) != str(detector_rows.iloc[0]["id"]):
            raise ValueError("Detector metadata id does not match its candidate row.")
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

    try:
        confidence = pd.to_numeric(outcomes["confidence"], errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Outcome confidence values must be numeric.") from exc
    if not np.isfinite(confidence).all() or (
        (confidence < 0.0) | (confidence > 1.0)
    ).any():
        raise ValueError("Outcome confidence values must be finite and within [0, 1].")

    accepted_series = outcomes["accepted"]
    if accepted_series.isna().any() or not all(
        isinstance(value, (bool, np.bool_)) for value in accepted_series
    ):
        raise ValueError("Outcome accepted values must be Boolean and non-null.")
    accepted = accepted_series.to_numpy(dtype=bool)

    try:
        numeric_prediction = pd.to_numeric(
            outcomes["prediction"], errors="raise"
        ).to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Outcome predictions must be integer indices.") from exc
    if not np.isfinite(numeric_prediction).all() or not np.equal(
        numeric_prediction, np.floor(numeric_prediction)
    ).all():
        raise ValueError("Outcome predictions must be finite integer indices.")
    prediction = numeric_prediction.astype(np.int64)

    outcome_candidate_ids = outcomes["candidate_id"].astype(str).to_numpy()
    legacy_without_output_labels = "output_labels" not in candidates.columns
    for candidate_id, metadata in candidate_metadata.items():
        mask = outcome_candidate_ids == candidate_id
        candidate_accepted = accepted[mask]
        candidate_prediction = prediction[mask]
        candidate_confidence = confidence[mask]
        kind = str(metadata["kind"])

        legacy_idk = candidate_prediction == -1
        if legacy_idk.any() and (
            not legacy_without_output_labels or candidate_accepted[legacy_idk].any()
        ):
            raise ValueError(
                f"Candidate {candidate_id!r} uses -1 outside a rejected legacy row."
            )
        allowed_predictions = metadata["allowed_predictions"]
        invalid_prediction = ~np.isin(
            candidate_prediction[~legacy_idk], list(allowed_predictions)
        )
        if invalid_prediction.any():
            invalid_values = sorted(
                set(candidate_prediction[~legacy_idk][invalid_prediction].tolist())
            )
            raise ValueError(
                f"Candidate {candidate_id!r} has predictions outside its shared "
                f"{kind} mapping: {invalid_values}"
            )

        threshold = metadata["threshold"]
        expected_accepted = (
            np.ones(mask.sum(), dtype=bool)
            if kind == "detector"
            else candidate_confidence >= float(threshold)
        )
        if not np.array_equal(candidate_accepted, expected_accepted):
            raise ValueError(
                f"Candidate {candidate_id!r} accepted values do not match its "
                "confidence threshold."
            )


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
