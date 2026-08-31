"""Profile K0-K6 and write the paper's device-specific classifier table.

The detector used by the paper-mode cascade is not executed.  Its registry
cost is set to the configured synthetic value (10,000 ms by default).
Quality statistics use the classifier-testing split, which is intentionally
separate from the run1/3/5/7/9 pool used for cascade optimization.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.m3n_vc.checkpoint_paths import (
    file_fingerprint,
    resolve_registry_checkpoint,
)
from experiments.m3n_vc.live_cascade_benchmark import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DETECTOR_COST_MS,
    DEFAULT_PROCESSED_DIR,
    _device_description,
    _latency_summary,
    _load_scene_arrays,
    _synchronize,
    load_live_models,
    resolve_device,
)
from experiments.m3n_vc.training.trainer import get_ki_labels
from experiments.m3n_vc.utils.labels import KI_REGISTRY, threshold_hi_for_ki
from experiments.m3n_vc.utils.splits import (
    apply_background_val_holdout,
    run_level_masks,
)


DEFAULT_SOURCE_REGISTRY = Path("checkpoints/classifier_registry.json")
DEFAULT_OUTPUT_REGISTRY = Path("checkpoints/classifier_registry_jetson_nano.json")
DEFAULT_OUTPUT_REPORT = Path("checkpoints/jetson_nano_model_profile.json")
PROFILE_MODEL_IDS = tuple(name for name in KI_REGISTRY if name != "Kdet")


def _environment(device: torch.device, requested: str) -> dict:
    result = {
        "device": _device_description(device),
        "device_request": requested,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result["cuda_version"] = torch.version.cuda
        result["cudnn_version"] = torch.backends.cudnn.version()
        result["cuda_device"] = {
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "major": int(props.major),
            "minor": int(props.minor),
        }
    return result


def _classifier_test_indices(
    metadata: pd.DataFrame,
    model_id: str,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the classifier-testing partition used during training."""
    spec = KI_REGISTRY[model_id]
    labels = get_ki_labels(metadata, spec)
    train_mask, test_mask, _ = run_level_masks(metadata, spec=spec)
    test_mask = test_mask.copy()
    if "background" in spec.class_names:
        _, test_mask = apply_background_val_holdout(
            metadata,
            train_mask,
            test_mask,
            background_run="run8",
            holdout_frac=0.20,
            seed=split_seed,
        )
    test_mask &= labels >= 0
    indices = np.flatnonzero(test_mask)
    if len(indices) == 0:
        raise ValueError(f"Classifier-testing split for {model_id} is empty.")
    return indices, labels[indices]


def _select_timing_inputs(
    mic: np.ndarray,
    geo: np.ndarray,
    eligible: np.ndarray,
    timed_samples: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if len(eligible) == 0:
        raise ValueError("No classifier-testing inputs are available for timing.")

    sample_count = len(eligible) if timed_samples == 0 else min(timed_samples, len(eligible))
    selected = np.random.default_rng(seed).choice(
        eligible, size=sample_count, replace=False
    )
    selected.sort()
    # Fancy indexing copies just the requested rows out of the memory maps.
    mic_values = np.ascontiguousarray(mic[selected, None, :, :])
    geo_values = np.ascontiguousarray(geo[selected, None, :, :])
    return (
        torch.from_numpy(mic_values).to(device),
        torch.from_numpy(geo_values).to(device),
        len(eligible),
    )


@torch.inference_mode()
def profile_model(
    model_id: str,
    model: torch.nn.Module,
    mic: torch.Tensor,
    geo: torch.Tensor,
    device: torch.device,
    warmup_iterations: int,
) -> dict:
    """Measure one forward + softmax/argmax at a time, with CUDA sync."""
    modality = KI_REGISTRY[model_id].modality

    def invoke(index: int) -> None:
        sample_mic = mic[index : index + 1]
        if modality == "mic":
            logits = model(sample_mic)
        else:
            logits = model(sample_mic, geo[index : index + 1])
        # Include the confidence operation used by the real cascade.
        _ = torch.softmax(logits, dim=1).max(dim=1)

    for index in range(warmup_iterations):
        invoke(index % len(mic))
    _synchronize(device)

    latencies_ms: list[float] = []
    for index in range(len(mic)):
        _synchronize(device)
        started = time.perf_counter()
        invoke(index)
        _synchronize(device)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)

    return {
        "model_id": model_id,
        "modality": modality,
        "batch_size": 1,
        "warmup_iterations": warmup_iterations,
        **_latency_summary(latencies_ms),
    }


@torch.inference_mode()
def evaluate_model_quality(
    model_id: str,
    model: torch.nn.Module,
    mic: np.ndarray,
    geo: np.ndarray,
    indices: np.ndarray,
    true_labels: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict:
    """Compute the paper's precision and success rate at the canonical threshold."""
    threshold = threshold_hi_for_ki(model_id)
    if threshold is None:
        raise ValueError(f"{model_id} unexpectedly has no confidence threshold.")
    spec = KI_REGISTRY[model_id]
    predicted_chunks: list[np.ndarray] = []
    confidence_chunks: list[np.ndarray] = []

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_mic = torch.from_numpy(
            np.ascontiguousarray(mic[batch_indices, None, :, :])
        ).to(device)
        if spec.modality == "mic":
            logits = model(batch_mic)
        else:
            batch_geo = torch.from_numpy(
                np.ascontiguousarray(geo[batch_indices, None, :, :])
            ).to(device)
            logits = model(batch_mic, batch_geo)
        confidence, prediction = torch.softmax(logits, dim=1).max(dim=1)
        predicted_chunks.append(prediction.cpu().numpy())
        confidence_chunks.append(confidence.cpu().numpy())

    predictions = np.concatenate(predicted_chunks).astype(np.int64, copy=False)
    confidences = np.concatenate(confidence_chunks).astype(np.float64, copy=False)
    accepted = confidences >= float(threshold)
    correct = predictions == true_labels
    accepted_count = int(accepted.sum())
    if accepted_count == 0:
        raise ValueError(f"{model_id} accepts no classifier-testing samples.")

    per_class: dict[str, dict[str, float | int | None]] = {}
    for class_index, class_name in enumerate(spec.class_names):
        class_mask = true_labels == class_index
        class_accepted = class_mask & accepted
        class_total = int(class_mask.sum())
        class_accepted_count = int(class_accepted.sum())
        class_correct_accepted = int((class_accepted & correct).sum())
        per_class[class_name] = {
            "scope_samples": class_total,
            "accepted_samples": class_accepted_count,
            "correct_accepted_samples": class_correct_accepted,
            "precision": (
                class_correct_accepted / class_accepted_count
                if class_accepted_count
                else None
            ),
            "success_rate": (
                class_accepted_count / class_total if class_total else None
            ),
        }

    return {
        "confidence_threshold": float(threshold),
        "precision": float((accepted & correct).sum() / accepted_count),
        "success_rate": float(accepted_count / len(indices)),
        "unconditional_accuracy": float(correct.mean()),
        "scope_samples": int(len(indices)),
        "accepted_samples": accepted_count,
        "correct_accepted_samples": int((accepted & correct).sum()),
        "per_class": per_class,
    }


def _write_profiled_registry(
    source: Path,
    output: Path,
    profiles: dict[str, dict],
    detector_cost_ms: float,
    metadata: dict,
) -> None:
    if source.resolve() == output.resolve():
        raise ValueError("Output registry must differ from the source registry.")
    payload = json.loads(source.read_text())
    rows = payload.get("classifiers")
    if not isinstance(rows, list):
        raise ValueError(f"{source} does not contain a classifiers list.")

    for row in rows:
        model_id = str(row.get("name"))
        if model_id in profiles:
            row["runtime_ms"] = profiles[model_id]["mean_ms"]
            row["wcet_ms"] = profiles[model_id]["max_ms"]
            row["threshold_hi"] = profiles[model_id]["confidence_threshold"]
        elif model_id == "Kdet":
            row["runtime_ms"] = detector_cost_ms
            row["wcet_ms"] = detector_cost_ms
    payload["runtime_profile"] = metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile K0-K6 for Jetson deployment and write a separate registry."
    )
    parser.add_argument("--scene", default="h24")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_SOURCE_REGISTRY)
    parser.add_argument("--output-registry", type=Path, default=DEFAULT_OUTPUT_REGISTRY)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--warmup-iterations", type=int, default=25)
    parser.add_argument("--quality-batch-size", type=int, default=64)
    parser.add_argument(
        "--classifier-split-seed",
        type=int,
        default=42,
        help="Seed selecting the held-out 20%% of background run8.",
    )
    parser.add_argument(
        "--timed-samples",
        type=int,
        default=500,
        help="Random batch-size-1 inputs per model; 0 uses the full eligible pool.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--detector-cost-ms", type=float, default=DEFAULT_DETECTOR_COST_MS
    )
    args = parser.parse_args()
    if (
        args.warmup_iterations < 0
        or args.timed_samples < 0
        or args.quality_batch_size < 1
    ):
        parser.error("Warmup/timed counts must be non-negative and batch size positive.")
    if args.detector_cost_ms < 0:
        parser.error("--detector-cost-ms must be non-negative.")

    device = resolve_device(args.device)
    models, registry, device = load_live_models(
        PROFILE_MODEL_IDS,
        args.checkpoint_dir,
        args.registry,
        device,
    )
    mic, geo, metadata_path = _load_scene_arrays(args.processed_dir, args.scene)
    metadata = pd.read_parquet(metadata_path)

    profiles: dict[str, dict] = {}
    for model_id in PROFILE_MODEL_IDS:
        indices, true_labels = _classifier_test_indices(
            metadata, model_id, args.classifier_split_seed
        )
        timing_mic, timing_geo, eligible_samples = _select_timing_inputs(
            mic,
            geo,
            indices,
            args.timed_samples,
            args.seed,
            device,
        )
        print(
            f"Profiling {model_id} ({len(timing_mic)} timed forwards; "
            f"{len(indices)} classifier-testing samples)..."
        )
        timing = profile_model(
            model_id,
            models[model_id],
            timing_mic,
            timing_geo,
            device,
            args.warmup_iterations,
        )
        quality = evaluate_model_quality(
            model_id,
            models[model_id],
            mic,
            geo,
            indices,
            true_labels,
            device,
            args.quality_batch_size,
        )
        record = registry.get(model_id)
        if record is None:
            raise ValueError(f"No registry record for {model_id}.")
        actual_parameters = sum(parameter.numel() for parameter in models[model_id].parameters())
        if record.num_params is not None:
            if int(record.num_params) != actual_parameters:
                raise ValueError(
                    f"{model_id} parameter count differs from registry: "
                    f"model={actual_parameters}, registry={record.num_params}."
                )
        checkpoint_path = resolve_registry_checkpoint(
            record.checkpoint,
            model_id,
            args.checkpoint_dir,
            registry_path=args.registry,
        )
        profiles[model_id] = {
            **timing,
            **quality,
            "parameters": int(actual_parameters),
            "paper_modality": "Acoustic" if KI_REGISTRY[model_id].modality == "mic" else "Both",
            "eligible_timing_samples": int(eligible_samples),
            "checkpoint": file_fingerprint(checkpoint_path),
        }

    environment = _environment(device, args.device)
    generated_at = datetime.now(timezone.utc).isoformat()
    classifier_table: dict[str, dict] = {
        model_id: {
            "modality": profiles[model_id]["paper_modality"],
            "parameters": profiles[model_id]["parameters"],
            "confidence_threshold": profiles[model_id]["confidence_threshold"],
            "precision": profiles[model_id]["precision"],
            "success_rate": profiles[model_id]["success_rate"],
            "execution_time_ms": profiles[model_id]["mean_ms"],
            "scope_samples": profiles[model_id]["scope_samples"],
            "accepted_samples": profiles[model_id]["accepted_samples"],
            "correct_accepted_samples": profiles[model_id][
                "correct_accepted_samples"
            ],
        }
        for model_id in PROFILE_MODEL_IDS
    }
    classifier_table["Kdet"] = {
        "modality": "—",
        "parameters": None,
        "confidence_threshold": None,
        "precision": 1.0,
        "success_rate": 1.0,
        "execution_time_ms": args.detector_cost_ms,
        "scope_samples": None,
        "accepted_samples": None,
        "correct_accepted_samples": None,
    }
    registry_metadata = {
        "schema_version": "classifier-runtime-profile/v2",
        "generated_at_utc": generated_at,
        "environment": environment,
        "scene": args.scene,
        "batch_size": 1,
        "timed_samples_per_model": {
            model_id: int(profiles[model_id]["samples"])
            for model_id in PROFILE_MODEL_IDS
        },
        "warmup_iterations_per_model": args.warmup_iterations,
        "classifier_testing_split": {
            "global_and_intermediate": (
                "run1,run3,run5,run7 plus deterministic 20% holdout of run8"
            ),
            "specialized_suv": "run1,run3 in-domain samples",
            "specialized_coupe": "run5,run7 in-domain samples",
            "background_holdout_fraction": 0.20,
            "background_split_seed": args.classifier_split_seed,
        },
        "classifier_training_split": {
            "global_and_intermediate": (
                "run0,run2,run4,run6 plus the complementary 80% of run8"
            ),
            "specialized_suv": "run0,run2 in-domain samples",
            "specialized_coupe": "run4,run6 in-domain samples",
            "background_training_fraction": 0.80,
            "background_split_seed": args.classifier_split_seed,
        },
        "confidence_thresholds": {
            model_id: profiles[model_id]["confidence_threshold"]
            for model_id in PROFILE_MODEL_IDS
        },
        "timing_scope": (
            "forward plus softmax/argmax; synchronized before and after each CUDA "
            "call; excludes model loading, data loading, and host-to-device transfer"
        ),
        "synthetic_detector_cost_ms": args.detector_cost_ms,
        "source_registry": str(args.registry.resolve()),
        "model_checkpoints": {
            model_id: profiles[model_id]["checkpoint"]
            for model_id in PROFILE_MODEL_IDS
        },
    }
    _write_profiled_registry(
        args.registry,
        args.output_registry,
        profiles,
        args.detector_cost_ms,
        registry_metadata,
    )

    report = {
        "schema_version": "model-classifier-profile/v2",
        "generated_at_utc": generated_at,
        "environment": environment,
        "scene": args.scene,
        "batch_size": 1,
        "classifier_testing_split": registry_metadata["classifier_testing_split"],
        "classifier_training_split": registry_metadata[
            "classifier_training_split"
        ],
        "definitions": {
            "precision": "P(correct | accepted)",
            "success_rate": "P(accepted)",
            "accepted": "maximum softmax confidence >= the model confidence threshold",
        },
        "timed_samples_per_model": registry_metadata["timed_samples_per_model"],
        "warmup_iterations_per_model": args.warmup_iterations,
        "timing_scope": registry_metadata["timing_scope"],
        "detector": {
            "mode": "paper_oracle_non_sleeping",
            "cost_ms": args.detector_cost_ms,
            "profiled": False,
        },
        "models": profiles,
        "classifier_table": classifier_table,
        "source_registry": str(args.registry.resolve()),
        "output_registry": str(args.output_registry.resolve()),
        "model_checkpoints": registry_metadata["model_checkpoints"],
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "device": environment["device"],
                "models_profiled": len(profiles),
                "timed_samples_per_model": registry_metadata[
                    "timed_samples_per_model"
                ],
                "output_registry": str(args.output_registry),
                "output_report": str(args.output_report),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
