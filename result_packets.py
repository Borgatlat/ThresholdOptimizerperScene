"""Standard JSON interchange packets for optimization results and figures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from cascade_profile import HierarchyProfile


RESULT_SCHEMA_VERSION = "cascade-result/v1"
REQUIRED_PARTITIONS = ("validation", "test")


def normalize_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    """Normalize optimizer metrics into the public packet vocabulary."""

    routes_value = metrics.get("routes", metrics.get("route_counts"))
    if not isinstance(routes_value, Mapping):
        raise ValueError("Metrics require routes or route_counts.")
    routes = {str(key): int(value) for key, value in routes_value.items()}
    total = int(metrics.get("samples", metrics.get("total", sum(routes.values()))))
    if total <= 0 or sum(routes.values()) != total:
        raise ValueError("Route counts must sum to the sample count.")
    expected_cost = (
        metrics["expected_cost_ms"]
        if "expected_cost_ms" in metrics
        else metrics["expected_cost"]
    )
    result: dict[str, object] = {
        "accuracy": float(metrics["accuracy"]),
        "expected_cost_ms": float(expected_cost),
        "samples": total,
        "routes": routes,
        "thresholds": {
            str(key): float(value)
            for key, value in dict(metrics.get("thresholds", {})).items()
        },
    }
    for key in (
        "feasible",
        "macro_accuracy",
        "worst_class_accuracy",
        "per_class_accuracy",
        "occurrence_routes",
    ):
        if key in metrics:
            result[key] = metrics[key]
    return result


def create_result_packet(
    *,
    profile: HierarchyProfile,
    method_id: str,
    method_label: str,
    target_accuracy: float,
    layout: Mapping[str, object],
    validation: Mapping[str, object],
    test: Mapping[str, object],
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    detector_id = str(layout.get("detector", "detector"))
    branch_value = dict(layout.get("branches", layout.get("specialized", {})))
    packet: dict[str, object] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "dataset": {
            "id": profile.dataset_id,
            "profile_fingerprint": profile.fingerprint,
        },
        "method": {
            "id": str(method_id),
            "label": str(method_label),
            "target_accuracy": float(target_accuracy),
        },
        "layout": {
            "initial": [
                str(item)
                for item in layout.get("initial", [])
                if str(item) != detector_id
            ],
            "branches": {
                str(key): [str(item) for item in value if str(item) != detector_id]
                for key, value in branch_value.items()
            },
            "detector": detector_id,
        },
        "partitions": {
            "validation": normalize_metrics(validation),
            "test": normalize_metrics(test),
        },
        "provenance": dict(provenance or {}),
    }
    validate_result_packet(packet)
    return packet


def validate_result_packet(packet: Mapping[str, object]) -> None:
    if packet.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(
            f"Expected {RESULT_SCHEMA_VERSION!r}, got {packet.get('schema_version')!r}."
        )
    for field in ("dataset", "method", "layout", "partitions"):
        if not isinstance(packet.get(field), Mapping):
            raise ValueError(f"Result packet field {field!r} must be an object.")
    dataset = packet["dataset"]
    method = packet["method"]
    layout = packet["layout"]
    assert isinstance(dataset, Mapping)
    assert isinstance(method, Mapping)
    assert isinstance(layout, Mapping)
    if not dataset.get("id") or not dataset.get("profile_fingerprint"):
        raise ValueError("Result packet dataset requires id and profile_fingerprint.")
    if not method.get("id") or not method.get("label"):
        raise ValueError("Result packet method requires id and label.")
    target_accuracy = float(method.get("target_accuracy", -1.0))
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("method.target_accuracy must be between 0 and 1.")
    if not isinstance(layout.get("initial"), list) or not isinstance(
        layout.get("branches"), Mapping
    ) or not layout.get("detector"):
        raise ValueError("Result packet layout requires initial, branches, and detector.")
    partitions = packet["partitions"]
    assert isinstance(partitions, Mapping)
    for partition in REQUIRED_PARTITIONS:
        metrics = partitions.get(partition)
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Result packet has no {partition!r} metrics.")
        normalized = normalize_metrics(metrics)
        accuracy = float(normalized["accuracy"])
        cost = float(normalized["expected_cost_ms"])
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError(f"{partition} accuracy must be between 0 and 1.")
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"{partition} expected cost must be finite and nonnegative.")
        if any(int(value) < 0 for value in normalized["routes"].values()):
            raise ValueError(f"{partition} route counts must be nonnegative.")
        if not all(
            math.isfinite(float(value))
            for value in normalized["thresholds"].values()
        ):
            raise ValueError(f"{partition} thresholds must be finite.")


def write_result_packet(packet: Mapping[str, object], path: str | Path) -> Path:
    validate_result_packet(packet)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(packet), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_result_packet(path: str | Path) -> dict[str, object]:
    source = Path(path)
    packet = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise ValueError(f"{source} must contain a JSON object.")
    validate_result_packet(packet)
    return packet
