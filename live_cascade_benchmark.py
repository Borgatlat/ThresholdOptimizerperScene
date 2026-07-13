"""Live, same-device latency benchmark for a saved cascade threshold policy.

The threshold optimizer estimates expected runtime from cached outcomes and
per-Ki timings that may have been measured on another device.  This script
loads the saved baseline and optimized threshold policies, runs their frozen
hierarchy against real spectrogram inputs, and measures their end-to-end
latency on the current machine.

This script requires a policy optimized with the logged, trained ``Kdet``.
Whenever a frozen layout reaches the detector sentinel, it runs that same
real model and compares its final global-label prediction with ground truth.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from checkpoint_paths import resolve_registry_checkpoint
from models.dual_modal_cnn import build_ki_model
from utils.classifier_registry import ClassifierRegistry
from utils.labels import GLOBAL_CLASS_NAMES, KI_REGISTRY


DEFAULT_METRICS_PATH = Path("checkpoints/threshold_optimizer_trained_metrics.json")
DEFAULT_OUTPUT_PATH = Path("checkpoints/empirical_outcomes.pkl")
DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_REGISTRY_PATH = Path("checkpoints/classifier_registry.json")
DETECTOR_SENTINEL = "detector"


@dataclass(frozen=True)
class FrozenLayout:
    initial: tuple[str, ...]
    specialized: dict[tuple[str, str], tuple[str, ...]]


@dataclass(frozen=True)
class LiveInputs:
    mic: torch.Tensor
    geo: torch.Tensor
    true_labels: np.ndarray
    scene: str
    available_samples: int


def _load_json(path: str | Path) -> dict:
    with Path(path).open() as file:
        return json.load(file)


def _load_empirical_outcomes(path: str | Path) -> dict:
    return pd.read_pickle(Path(path))


def _normalize_spectrograms(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=(1, 2), keepdims=True)
    std = values.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((values - mean) / std).astype(np.float32)


def _scene_processed_dir(processed_dir: str | Path, scene: str) -> Path:
    processed_dir = Path(processed_dir)
    if (processed_dir / f"{scene}_metadata.parquet").is_file():
        return processed_dir
    nested = processed_dir / scene
    if (nested / f"{scene}_metadata.parquet").is_file():
        return nested
    return processed_dir


def _load_scene_arrays(processed_dir: str | Path, scene: str) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load the collector's normalized input representation without training imports."""
    scene_dir = _scene_processed_dir(processed_dir, scene)
    metadata_path = scene_dir / f"{scene}_metadata.parquet"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path} not found; run process_data.py for scene {scene!r}."
        )

    normalized_mic = scene_dir / f"{scene}_paired_mic_norm.npy"
    normalized_geo = scene_dir / f"{scene}_paired_geo_norm.npy"
    if normalized_mic.is_file() and normalized_geo.is_file():
        mic = np.load(normalized_mic)
        geo = np.load(normalized_geo)
    else:
        mic = _normalize_spectrograms(np.load(scene_dir / f"{scene}_paired_mic.npy"))
        geo = _normalize_spectrograms(np.load(scene_dir / f"{scene}_paired_geo.npy"))
    return mic, geo, metadata_path


def _resolve_partition(metrics: Mapping[str, object], partition: str) -> str:
    if partition != "auto":
        return partition
    return "holdout" if "split" in metrics else "all"


def _require_trained_detector_metrics(metrics: Mapping[str, object]) -> None:
    detector_mode = metrics.get("detector_mode")
    if detector_mode != "trained":
        raise ValueError(
            "Live comparison requires metrics optimized with the real logged Kdet "
            "(--detector-mode trained). The supplied metrics use "
            f"{detector_mode!r}; regenerate them before benchmarking."
        )


def _policy_section(metrics: Mapping[str, object], policy: str, partition: str) -> Mapping[str, object]:
    key = "annealing" if policy == "optimized" else "baseline"
    if key not in metrics:
        raise KeyError(f"Metrics file has no {key!r} policy.")

    section = metrics[key]
    if not isinstance(section, Mapping):
        raise ValueError(f"Metrics entry {key!r} is not an object.")
    if partition != "all" and "thresholds" not in section:
        saved_partition = partition
        if partition == "validation" and "validation" not in section:
            saved_partition = "optimization"  # Legacy report compatibility.
        if saved_partition not in section or not isinstance(section[saved_partition], Mapping):
            raise KeyError(
                f"Metrics policy {key!r} has no {partition!r} thresholds. "
                "Choose a partition saved by the threshold optimizer."
            )
        section = section[saved_partition]
    elif partition == "all" and "thresholds" not in section:
        # A holdout experiment keeps the learned thresholds inside its saved
        # partition. Use holdout first because it is the deployment-facing
        # policy, then fall back to the validation partition.
        for saved_partition in ("holdout", "validation", "optimization"):
            candidate = section.get(saved_partition)
            if isinstance(candidate, Mapping) and "thresholds" in candidate:
                section = candidate
                break

    if "thresholds" not in section:
        raise KeyError(f"Metrics policy {key!r} does not contain thresholds.")
    return section


def load_policy_thresholds(
    metrics: Mapping[str, object],
    policy: str,
    partition: str,
) -> dict[str, float]:
    section = _policy_section(metrics, policy, partition)
    thresholds = section["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise ValueError(f"{policy} thresholds are not a JSON object.")
    return {str(candidate_id): float(value) for candidate_id, value in thresholds.items()}


def load_frozen_layout(metrics: Mapping[str, object]) -> FrozenLayout:
    split = metrics.get("split")
    if not isinstance(split, Mapping):
        raise ValueError(
            "The metrics file does not contain a frozen split/layout. "
            "Regenerate it with --holdout-fraction before live benchmarking."
        )
    initial = split.get("initial_layout")
    specialized = split.get("specialized_layout")
    if not isinstance(initial, list) or not isinstance(specialized, Mapping):
        raise ValueError("Metrics file is missing initial_layout or specialized_layout.")

    parsed_specialized: dict[tuple[str, str], tuple[str, ...]] = {}
    for key, chain in specialized.items():
        if not isinstance(key, str) or ":" not in key or not isinstance(chain, list):
            raise ValueError(f"Malformed specialized layout entry: {key!r}")
        router_id, group = key.split(":", 1)
        parsed_specialized[(router_id, group)] = tuple(str(candidate_id) for candidate_id in chain)

    return FrozenLayout(
        initial=tuple(str(candidate_id) for candidate_id in initial),
        specialized=parsed_specialized,
    )


def active_model_ids(layout: FrozenLayout) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for chain in (layout.initial, *layout.specialized.values()):
        for candidate_id in chain:
            if candidate_id == DETECTOR_SENTINEL or candidate_id in seen:
                continue
            seen.add(candidate_id)
            ordered.append(candidate_id)
    if "Kdet" not in seen:
        ordered.append("Kdet")
    return tuple(ordered)


def load_live_models(
    model_ids: Sequence[str],
    checkpoint_dir: str | Path,
    registry_path: str | Path,
) -> tuple[dict[str, torch.nn.Module], ClassifierRegistry, torch.device]:
    """Load only models reached by the saved layout, plus Kdet."""
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    registry_path = Path(registry_path).expanduser().resolve()
    registry = ClassifierRegistry.load(registry_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models: dict[str, torch.nn.Module] = {}

    for model_id in model_ids:
        record = registry.get(model_id)
        if record is None:
            raise ValueError(f"No registry record for {model_id}")
        checkpoint_path = resolve_registry_checkpoint(
            record.checkpoint,
            model_id,
            checkpoint_dir,
            registry_path=registry_path,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain a state dictionary.")
        model = build_ki_model(model_id, len(record.class_names)).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models[model_id] = model
        print(f"Loaded {model_id} from {checkpoint_path}")

    return models, registry, device


class LiveCascade:
    """Execute one frozen hierarchy using live softmax confidences."""

    def __init__(
        self,
        layout: FrozenLayout,
        thresholds: Mapping[str, float],
        models: Mapping[str, torch.nn.Module],
        registry: ClassifierRegistry,
    ) -> None:
        self.layout = layout
        self.thresholds = {candidate_id: float(value) for candidate_id, value in thresholds.items()}
        self.models = models
        self.class_names = {
            model_id: tuple(registry.get(model_id).class_names)  # type: ignore[union-attr]
            for model_id in models
        }
        active = set(active_model_ids(layout)) - {"Kdet"}
        missing_thresholds = active - set(self.thresholds)
        if missing_thresholds:
            raise ValueError(
                f"Saved policy is missing thresholds for {sorted(missing_thresholds)}"
            )

    @torch.inference_mode()
    def run(self, mic: torch.Tensor, geo: torch.Tensor) -> str:
        """Run the live cascade and return its final global-label prediction."""
        for candidate_id in self.layout.initial:
            if candidate_id == DETECTOR_SENTINEL:
                return self._infer("Kdet", mic, geo)[0]

            label, confidence = self._infer(candidate_id, mic, geo)
            if confidence < self.thresholds[candidate_id]:
                continue

            if KI_REGISTRY[candidate_id].level == "intermediate":
                if label in {"suv", "coupe"}:
                    chain = self.layout.specialized.get(
                        (candidate_id, label), (DETECTOR_SENTINEL,)
                    )
                    return self._run_specialized(chain, mic, geo)
                # "background" is a valid global leaf from an identifier.
                if label in GLOBAL_CLASS_NAMES:
                    return label
                return self._infer("Kdet", mic, geo)[0]

            return label

        return self._infer("Kdet", mic, geo)[0]

    def warmup_models(self, mic: torch.Tensor, geo: torch.Tensor) -> None:
        """Warm every reachable model once, including branch-only models."""
        for candidate_id in self.models:
            self._infer(candidate_id, mic, geo)

    def _run_specialized(
        self,
        chain: Sequence[str],
        mic: torch.Tensor,
        geo: torch.Tensor,
    ) -> str:
        for candidate_id in chain:
            if candidate_id == DETECTOR_SENTINEL:
                return self._infer("Kdet", mic, geo)[0]
            label, confidence = self._infer(candidate_id, mic, geo)
            if confidence >= self.thresholds[candidate_id]:
                return label
        return self._infer("Kdet", mic, geo)[0]

    def _infer(self, candidate_id: str, mic: torch.Tensor, geo: torch.Tensor) -> tuple[str, float]:
        model = self.models[candidate_id]
        if KI_REGISTRY[candidate_id].modality == "mic":
            logits = model(mic)
        else:
            logits = model(mic, geo)
        confidence, class_index = torch.softmax(logits, dim=1).max(dim=1)
        return self.class_names[candidate_id][int(class_index.item())], float(confidence.item())


def _select_partition_sample_ids(
    labels,
    metrics: Mapping[str, object],
    partition: str,
) -> np.ndarray:
    labels = labels.sort_values("sample_id")
    sample_ids = labels["sample_id"].to_numpy(dtype=int)
    if partition == "all":
        return sample_ids

    split = metrics.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("The metrics file has no split to reproduce.")
    strategy = str(split.get("strategy", "blocked_per_run"))
    fraction = float(split.get("holdout_fraction", 0.20))
    seed = int(split.get("random_seed", 0))
    if strategy not in {"blocked_per_run", "random_per_run"}:
        raise ValueError(f"Unsupported saved split strategy: {strategy!r}")

    rng = np.random.default_rng(seed)
    holdout = np.zeros(len(labels), dtype=bool)
    for _, run_labels in labels.groupby("run_id", sort=False):
        run_ids = run_labels["sample_id"].to_numpy(dtype=int)
        count = int(round(len(run_ids) * fraction))
        count = min(max(count, 1), len(run_ids) - 1)
        selected = run_ids[-count:] if strategy == "blocked_per_run" else rng.choice(
            run_ids, size=count, replace=False
        )
        holdout[selected] = True

    if partition == "holdout":
        return sample_ids[holdout]
    if partition == "validation":
        return sample_ids[~holdout]
    raise ValueError("partition must be all, validation, holdout, or auto.")


def load_live_inputs(
    outcomes_path: str | Path,
    metrics: Mapping[str, object],
    scene: str | None,
    partition: str,
    processed_dir: str | Path,
    device: torch.device,
    max_samples: int,
    random_seed: int,
) -> LiveInputs:
    def tensors_for_indices(
        mic: np.ndarray,
        geo: np.ndarray,
        raw_indices: np.ndarray,
        true_labels: np.ndarray,
    ) -> LiveInputs:
        available = len(raw_indices)
        if len(true_labels) != available:
            raise ValueError("Live input count differs from the number of ground-truth labels.")
        if max_samples > 0 and available > max_samples:
            selected_positions = np.random.default_rng(random_seed).choice(
                available, size=max_samples, replace=False
            )
            raw_indices = raw_indices[selected_positions]
            true_labels = true_labels[selected_positions]
        mic_tensor = torch.from_numpy(mic[raw_indices, None, :, :]).to(device)
        geo_tensor = torch.from_numpy(geo[raw_indices, None, :, :]).to(device)
        return LiveInputs(
            mic=mic_tensor,
            geo=geo_tensor,
            true_labels=np.asarray(true_labels, dtype=str),
            scene="",
            available_samples=available,
        )

    if partition == "all" and scene is not None:
        mic, geo, metadata_path = _load_scene_arrays(processed_dir, scene)
        metadata = pd.read_parquet(metadata_path)
        if "global_label" not in metadata:
            raise ValueError(f"{metadata_path} has no global_label column for live accuracy.")
        if len(metadata) != len(mic) or len(mic) != len(geo):
            raise ValueError("Processed mic, geo, and metadata rows do not align.")
        loaded = tensors_for_indices(
            mic,
            geo,
            np.arange(len(mic), dtype=int),
            metadata["global_label"].astype(str).to_numpy(),
        )
        return LiveInputs(
            mic=loaded.mic,
            geo=loaded.geo,
            true_labels=loaded.true_labels,
            scene=scene,
            available_samples=loaded.available_samples,
        )

    payload = _load_empirical_outcomes(outcomes_path)
    labels = payload["labels"].sort_values("sample_id")
    scenes = labels["scene"].astype(str).unique().tolist()
    if scene is None:
        if len(scenes) != 1:
            raise ValueError(f"Outcomes contains multiple scenes: {scenes}; pass --scene.")
        scene = scenes[0]
    elif scene not in scenes:
        raise ValueError(f"Scene {scene!r} does not match outcomes scenes {scenes}.")

    mic, geo, metadata_path = _load_scene_arrays(processed_dir, scene)
    metadata = pd.read_parquet(metadata_path)
    # The collector's h24 table contains only its held-out run ids; every
    # non-h24 table contains all rows. Matching the run ids reproduces both
    # choices without importing collector/training dependencies.
    logged_runs = set(labels["run_id"].astype(str))
    eval_mask = metadata["run_id"].astype(str).isin(logged_runs).to_numpy()
    raw_indices = np.flatnonzero(eval_mask)
    eval_metadata = metadata.loc[eval_mask].reset_index(drop=True)
    if len(raw_indices) != len(labels):
        raise ValueError(
            "Processed inputs do not line up with empirical outcomes. Regenerate "
            "the outcomes for this scene or use their matching --processed-dir."
        )
    if not np.array_equal(
        eval_metadata["run_id"].astype(str).to_numpy(),
        labels["run_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Processed input order differs from the saved empirical outcomes.")
    if not np.array_equal(
        eval_metadata["global_label"].astype(str).to_numpy(),
        labels["true_global_label"].astype(str).to_numpy(),
    ):
        raise ValueError("Processed input labels differ from the saved empirical outcomes.")

    sample_ids = _select_partition_sample_ids(labels, metrics, partition)
    selected_raw_indices = raw_indices[sample_ids]
    selected_labels = labels.set_index("sample_id").loc[
        sample_ids, "true_global_label"
    ].astype(str).to_numpy()
    loaded = tensors_for_indices(
        mic, geo, selected_raw_indices, selected_labels
    )
    return LiveInputs(
        mic=loaded.mic,
        geo=loaded.geo,
        true_labels=loaded.true_labels,
        scene=scene,
        available_samples=loaded.available_samples,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_one(
    cascade: LiveCascade,
    mic: torch.Tensor,
    geo: torch.Tensor,
    device: torch.device,
) -> tuple[float, str]:
    _synchronize(device)
    started = time.perf_counter()
    prediction = cascade.run(mic, geo)
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0, prediction


def _latency_summary(latencies_ms: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(latencies_ms, dtype=float)
    if len(values) == 0:
        raise ValueError("No timed samples were collected.")
    return {
        "samples": int(len(values)),
        "avg_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "wcet_ms": float(values.max()),
        "min_ms": float(values.min()),
        "std_ms": float(values.std()),
        "total_ms": float(values.sum()),
    }


def _accuracy_summary(
    predictions: Sequence[str],
    true_labels: Sequence[str],
) -> dict[str, object]:
    predicted = np.asarray(predictions, dtype=str)
    truth = np.asarray(true_labels, dtype=str)
    if len(predicted) != len(truth):
        raise ValueError("Prediction count differs from the number of ground-truth labels.")
    if len(truth) == 0:
        raise ValueError("No samples were available for live accuracy measurement.")

    correct_mask = predicted == truth
    per_class: dict[str, dict[str, float | int | None]] = {}
    represented_accuracies: list[float] = []
    for class_name in GLOBAL_CLASS_NAMES:
        class_mask = truth == class_name
        total = int(class_mask.sum())
        correct = int((correct_mask & class_mask).sum())
        accuracy = correct / total if total else None
        per_class[class_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        if accuracy is not None:
            represented_accuracies.append(accuracy)

    return {
        "accuracy": float(correct_mask.mean()),
        "correct": int(correct_mask.sum()),
        "accuracy_samples": int(len(truth)),
        "per_class_accuracy": per_class,
        "macro_accuracy": float(np.mean(represented_accuracies)),
        "worst_class_accuracy": float(np.min(represented_accuracies)),
    }


def benchmark_live_policies(
    baseline: LiveCascade,
    optimized: LiveCascade,
    mic: torch.Tensor,
    geo: torch.Tensor,
    true_labels: Sequence[str],
    device: torch.device,
    warmup_samples: int,
    timed_samples: int,
    random_seed: int,
) -> tuple[dict, dict]:
    if len(mic) != len(geo):
        raise ValueError("Mic and geo input counts differ.")
    if len(mic) != len(true_labels):
        raise ValueError("Live input count differs from the number of ground-truth labels.")
    if len(mic) < 2:
        raise ValueError("Need at least two live samples to benchmark a cascade.")

    rng = np.random.default_rng(random_seed)
    order = rng.permutation(len(mic))
    warmup_count = min(max(warmup_samples, 0), len(order) - 1)
    remaining = order[warmup_count:]
    if timed_samples > 0:
        remaining = remaining[: min(timed_samples, len(remaining))]
    if len(remaining) == 0:
        raise ValueError("No samples remain after warmup; lower --warmup-samples.")

    # First-use kernel/runtime work should not pollute either policy's timing.
    baseline.warmup_models(mic[:1], geo[:1])
    optimized.warmup_models(mic[:1], geo[:1])
    _synchronize(device)
    baseline_latencies: list[float] = []
    optimized_latencies: list[float] = []
    baseline_predictions: list[str] = []
    optimized_predictions: list[str] = []
    accuracy_indices: list[int] = []
    for index in order[:warmup_count]:
        index = int(index)
        sample_mic = mic[index : index + 1]
        sample_geo = geo[index : index + 1]
        baseline_predictions.append(baseline.run(sample_mic, sample_geo))
        optimized_predictions.append(optimized.run(sample_mic, sample_geo))
        accuracy_indices.append(index)
    _synchronize(device)

    for position, index in enumerate(remaining):
        index = int(index)
        sample_mic = mic[index : index + 1]
        sample_geo = geo[index : index + 1]
        # Alternate first position so neither policy always benefits from cache state.
        if position % 2 == 0:
            latency, prediction = _measure_one(optimized, sample_mic, sample_geo, device)
            optimized_latencies.append(latency)
            optimized_predictions.append(prediction)
            latency, prediction = _measure_one(baseline, sample_mic, sample_geo, device)
            baseline_latencies.append(latency)
            baseline_predictions.append(prediction)
        else:
            latency, prediction = _measure_one(baseline, sample_mic, sample_geo, device)
            baseline_latencies.append(latency)
            baseline_predictions.append(prediction)
            latency, prediction = _measure_one(optimized, sample_mic, sample_geo, device)
            optimized_latencies.append(latency)
            optimized_predictions.append(prediction)
        accuracy_indices.append(index)

    accuracy_truth = np.asarray(true_labels, dtype=str)[accuracy_indices]
    return (
        {
            **_latency_summary(baseline_latencies),
            **_accuracy_summary(baseline_predictions, accuracy_truth),
        },
        {
            **_latency_summary(optimized_latencies),
            **_accuracy_summary(optimized_predictions, accuracy_truth),
        },
    )


def _device_description(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(device)}"
    return str(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark saved baseline/optimized thresholds with live model inference."
    )
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--scene", default=None, help="Infer from outcomes when omitted.")
    parser.add_argument(
        "--partition",
        choices=("auto", "all", "validation", "holdout"),
        default="auto",
        help=(
            "Use the same saved partition as the threshold policy. With --scene, "
            "all samples from every processed input without needing outcomes metadata."
        ),
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=25,
        help="Untimed dynamic-cascade warmup samples per policy.",
    )
    parser.add_argument(
        "--timed-samples",
        type=int,
        default=250,
        help="Timed samples; use 0 to time every sample in the selected partition.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON benchmark report.",
    )
    args = parser.parse_args()
    if args.warmup_samples < 0 or args.timed_samples < 0:
        parser.error("--warmup-samples and --timed-samples must be non-negative.")

    metrics = _load_json(args.metrics)
    _require_trained_detector_metrics(metrics)
    partition = _resolve_partition(metrics, args.partition)
    layout = load_frozen_layout(metrics)
    baseline_thresholds = load_policy_thresholds(metrics, "baseline", partition)
    optimized_thresholds = load_policy_thresholds(metrics, "optimized", partition)
    model_ids = active_model_ids(layout)

    models, registry, device = load_live_models(
        model_ids, args.checkpoint_dir, args.registry
    )
    input_sample_limit = (
        args.warmup_samples + args.timed_samples if args.timed_samples > 0 else 0
    )
    live_inputs = load_live_inputs(
        args.outcomes,
        metrics,
        args.scene,
        partition,
        args.processed_dir,
        device,
        input_sample_limit,
        args.seed,
    )
    baseline = LiveCascade(layout, baseline_thresholds, models, registry)
    optimized = LiveCascade(layout, optimized_thresholds, models, registry)
    baseline_stats, optimized_stats = benchmark_live_policies(
        baseline,
        optimized,
        live_inputs.mic,
        live_inputs.geo,
        live_inputs.true_labels,
        device,
        args.warmup_samples,
        args.timed_samples,
        args.seed,
    )

    report = {
        "scene": live_inputs.scene,
        "partition": partition,
        "device": _device_description(device),
        "torch_version": torch.__version__,
        "timing_scope": (
            "live model forwards, softmax, and cascade routing; excludes data "
            "loading and host-to-device transfer"
        ),
        "detector_mode": "trained",
        "fallback": "live trained Kdet model",
        "accuracy_scope": (
            "live predictions on every loaded sample, including untimed warmup "
            "samples; use --timed-samples 0 to load the full selected partition"
        ),
        "layout": {
            "initial": list(layout.initial),
            "specialized": {
                f"{router_id}:{group}": list(chain)
                for (router_id, group), chain in layout.specialized.items()
            },
        },
        "warmup_samples_per_policy": args.warmup_samples,
        "available_samples": live_inputs.available_samples,
        "loaded_samples": int(len(live_inputs.mic)),
        "baseline": {"thresholds": baseline_thresholds, **baseline_stats},
        "optimized": {"thresholds": optimized_thresholds, **optimized_stats},
        "optimized_vs_baseline": {
            "avg_speedup": float(baseline_stats["avg_ms"] / optimized_stats["avg_ms"]),
            "avg_reduction_percent": float(
                100.0 * (1.0 - optimized_stats["avg_ms"] / baseline_stats["avg_ms"])
            ),
            "p95_speedup": float(baseline_stats["p95_ms"] / optimized_stats["p95_ms"]),
            "wcet_speedup": float(baseline_stats["wcet_ms"] / optimized_stats["wcet_ms"]),
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
