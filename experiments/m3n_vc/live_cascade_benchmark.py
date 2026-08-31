"""Live, same-device latency benchmark for a saved cascade threshold policy.

The threshold optimizer estimates expected runtime from cached outcomes and
per-Ki timings that may have been measured on another device.  This script
loads the saved baseline and optimized threshold policies, runs their frozen
hierarchy against real spectrogram inputs, and measures their end-to-end
latency on the current machine.

For the paper-mode experiments, ``detector`` is an oracle with a synthetic
10,000 ms cost.  It therefore returns the ground-truth label immediately and
adds the configured cost without sleeping.  K0-K6 are always real model
forwards.  Run ``profile_models_jetson.py`` first so the report can also
recalculate expected cost from timings measured on the deployment device.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from experiments.m3n_vc.checkpoint_paths import resolve_registry_checkpoint
from experiments.m3n_vc.models.dual_modal_cnn import build_ki_model
from threshold_optimizer import enumerate_threshold_slots, normalise_threshold_slots
from experiments.m3n_vc.utils.classifier_registry import ClassifierRegistry
from experiments.m3n_vc.utils.labels import GLOBAL_CLASS_NAMES, KI_REGISTRY


DEFAULT_METRICS_PATH = Path(
    "checkpoints/k1_including_h24_with_run9_target_095_paper_sa/ga/summary.json"
)
DEFAULT_OUTPUT_PATH = Path("checkpoints/empirical_outcomes_h24_with_run9.pkl")
DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_REGISTRY_PATH = Path("checkpoints/classifier_registry_jetson_nano.json")
DEFAULT_REPORT_PATH = Path("checkpoints/jetson_nano_live_cascade_h24.json")
DEFAULT_DETECTOR_COST_MS = 10_000.0
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


@dataclass(frozen=True)
class CascadeRun:
    prediction: str
    terminal_route: str
    route_path: tuple[str, ...]
    invocations: tuple[tuple[str, str, float], ...]
    measured_wall_ms: float
    synthetic_detector_ms: float

    @property
    def measured_model_ms(self) -> float:
        return float(sum(latency for _, _, latency in self.invocations))

    @property
    def detector_used(self) -> bool:
        return self.synthetic_detector_ms > 0.0


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
        # Memory mapping matters on a 4 GB Jetson Nano; selected rows are
        # copied only after the saved validation/testing split is resolved.
        mic = np.load(normalized_mic, mmap_mode="r")
        geo = np.load(normalized_geo, mmap_mode="r")
    else:
        mic = _normalize_spectrograms(np.load(scene_dir / f"{scene}_paired_mic.npy"))
        geo = _normalize_spectrograms(np.load(scene_dir / f"{scene}_paired_geo.npy"))
    return mic, geo, metadata_path


def _resolve_partition(metrics: Mapping[str, object], partition: str) -> str:
    if partition != "auto":
        return partition
    return "holdout" if "split" in metrics else "all"


def resolve_device(requested: str = "auto") -> torch.device:
    """Prefer CUDA, but make Jetson installations without CUDA usable on CPU."""
    if requested not in {"auto", "cuda", "cpu"}:
        raise ValueError("device must be auto, cuda, or cpu")
    if requested != "cpu" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if requested == "cuda":
        print("CUDA was requested but is unavailable; falling back to CPU.")
    elif requested == "auto":
        print("CUDA is unavailable; using CPU.")
    return torch.device("cpu")


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


def _parse_saved_layout(raw_layout: object) -> "FrozenLayout":
    if not isinstance(raw_layout, Mapping):
        raise ValueError("Saved policy has no layout object.")
    initial = raw_layout.get("initial")
    specialized = raw_layout.get("specialized")
    if not isinstance(initial, list) or not isinstance(specialized, Mapping):
        raise ValueError("Saved layout is missing initial or specialized chains.")

    parsed_specialized: dict[tuple[str, str], tuple[str, ...]] = {}
    for key, chain in specialized.items():
        if not isinstance(key, str) or ":" not in key or not isinstance(chain, list):
            raise ValueError(f"Malformed saved specialized layout: {key!r}")
        router_id, group = key.split(":", 1)
        parsed_specialized[(router_id, group)] = tuple(str(value) for value in chain)

    return FrozenLayout(
        initial=tuple(str(value) for value in initial),
        specialized=parsed_specialized,
    )


def load_saved_policy(
    summary: Mapping[str, object],
    saved_policy: str,
    partition: str,
) -> tuple[
    "FrozenLayout",
    dict[str, float],
    tuple[str, ...] | None,
    Mapping[str, object],
]:
    """Load a GA winner or one method from a DP/threshold benchmark summary."""
    if saved_policy == "winner":
        container = summary.get("winner")
        if not isinstance(container, Mapping):
            raise ValueError("Summary has no winner object.")
        raw_layout = container.get("layout")
    else:
        methods = summary.get("methods")
        if not isinstance(methods, Mapping):
            raise ValueError("Summary has no methods object.")
        container = methods.get(saved_policy)
        if not isinstance(container, Mapping):
            raise ValueError(f"Summary has no {saved_policy!r} method.")
        raw_layout = container.get("layout", summary.get("layout"))

    layout = _parse_saved_layout(raw_layout)

    saved_partition = "holdout" if partition in {"all", "holdout"} else partition
    policy = container.get(saved_partition)
    if not isinstance(policy, Mapping):
        raise ValueError(
            f"Saved policy {saved_policy!r} has no {saved_partition!r} packet."
        )
    # Holdout is evaluation-only. Thresholds and validation-time pruning must
    # not change just because a slot happened to be unreachable on the test
    # samples, so deploy the validation-selected configuration verbatim.
    configuration = (
        container.get("validation")
        if saved_partition == "holdout"
        else policy
    )
    if not isinstance(configuration, Mapping):
        raise ValueError(
            f"Saved policy {saved_policy!r} has no validation configuration."
        )
    raw_thresholds = configuration.get("thresholds")
    if not isinstance(raw_thresholds, Mapping):
        raise ValueError(
            f"Saved policy {saved_policy!r} {saved_partition!r} packet has no thresholds."
        )
    raw_active = configuration.get("active_slots")
    if raw_active is None:
        active_slots: tuple[str, ...] | None = None
    elif isinstance(raw_active, list):
        active_slots = tuple(str(value) for value in raw_active)
    else:
        raise ValueError("Saved active_slots must be a list when present.")
    return (
        layout,
        {str(key): float(value) for key, value in raw_thresholds.items()},
        active_slots,
        policy,
    )


def load_winner_policy(
    summary: Mapping[str, object],
    partition: str,
) -> tuple[
    "FrozenLayout",
    dict[str, float],
    tuple[str, ...] | None,
    Mapping[str, object],
]:
    """Compatibility wrapper for loading a joint-optimizer winner."""
    return load_saved_policy(summary, "winner", partition)


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
    return tuple(ordered)


def load_live_models(
    model_ids: Sequence[str],
    checkpoint_dir: str | Path,
    registry_path: str | Path,
    device: torch.device | None = None,
) -> tuple[dict[str, torch.nn.Module], ClassifierRegistry, torch.device]:
    """Load only real models reached by the saved layout."""
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    registry_path = Path(registry_path).expanduser().resolve()
    registry = ClassifierRegistry.load(registry_path)
    device = device or resolve_device("auto")
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
    """Execute one frozen hierarchy with real Ki models and an oracle detector."""

    def __init__(
        self,
        layout: FrozenLayout,
        thresholds: Mapping[str, float],
        models: Mapping[str, torch.nn.Module],
        registry: ClassifierRegistry,
        device: torch.device,
        *,
        active_slots: Sequence[str] | None = None,
        detector_cost_ms: float = DEFAULT_DETECTOR_COST_MS,
    ) -> None:
        self.layout = layout
        self.threshold_slots = enumerate_threshold_slots(
            layout.initial,
            layout.specialized,
            DETECTOR_SENTINEL,
        )
        self.thresholds = normalise_threshold_slots(
            self.threshold_slots,
            thresholds,
        )
        self._slot_by_location = {
            slot.location: slot.key for slot in self.threshold_slots
        }
        known_slots = {slot.key for slot in self.threshold_slots}
        unknown_active = set(active_slots or ()) - known_slots
        if unknown_active:
            raise ValueError(f"active_slots contains unknown slots: {sorted(unknown_active)}")
        self.active_slots = None if active_slots is None else frozenset(active_slots)
        self.models = models
        self.device = device
        self.detector_cost_ms = float(detector_cost_ms)
        self.class_names = {
            model_id: tuple(registry.get(model_id).class_names)  # type: ignore[union-attr]
            for model_id in models
        }

    @torch.inference_mode()
    def run(self, mic: torch.Tensor, geo: torch.Tensor, true_label: str) -> CascadeRun:
        """Run one sample; the detector returns ``true_label`` without waiting."""
        _synchronize(self.device)
        wall_started = time.perf_counter()
        route_path: list[str] = []
        invocations: list[tuple[str, str, float]] = []

        for index, candidate_id in enumerate(self.layout.initial):
            if candidate_id == DETECTOR_SENTINEL:
                return self._finish(
                    true_label, DETECTOR_SENTINEL, route_path, invocations, wall_started, True
                )

            slot_id = self._slot_by_location[f"initial[{index}]"]
            if not self._is_active(slot_id):
                continue
            label, confidence, latency_ms = self._infer(candidate_id, mic, geo)
            route_path.append(slot_id)
            invocations.append((slot_id, candidate_id, latency_ms))
            if confidence <= self.thresholds[slot_id]:
                continue

            if KI_REGISTRY[candidate_id].level == "intermediate":
                if label in {"suv", "coupe"}:
                    chain = self.layout.specialized.get(
                        (candidate_id, label), (DETECTOR_SENTINEL,)
                    )
                    return self._run_specialized(
                        chain,
                        candidate_id,
                        label,
                        mic,
                        geo,
                        true_label,
                        route_path,
                        invocations,
                        wall_started,
                    )
                # "background" is a valid global leaf from an identifier.
                if label in GLOBAL_CLASS_NAMES:
                    return self._finish(
                        label, slot_id, route_path, invocations, wall_started, False
                    )
                return self._finish(
                    true_label, DETECTOR_SENTINEL, route_path, invocations, wall_started, True
                )

            return self._finish(label, slot_id, route_path, invocations, wall_started, False)

        return self._finish(
            true_label, DETECTOR_SENTINEL, route_path, invocations, wall_started, True
        )

    def warmup_models(
        self, mic: torch.Tensor, geo: torch.Tensor, iterations: int = 25
    ) -> None:
        """Warm every reachable real model; detector is synthetic and omitted."""
        for candidate_id in self.models:
            for _ in range(max(0, iterations)):
                self._infer(candidate_id, mic, geo)

    def _run_specialized(
        self,
        chain: Sequence[str],
        router_id: str,
        group: str,
        mic: torch.Tensor,
        geo: torch.Tensor,
        true_label: str,
        route_path: list[str],
        invocations: list[tuple[str, str, float]],
        wall_started: float,
    ) -> CascadeRun:
        for index, candidate_id in enumerate(chain):
            if candidate_id == DETECTOR_SENTINEL:
                return self._finish(
                    true_label, DETECTOR_SENTINEL, route_path, invocations, wall_started, True
                )
            location = f"specialized[{router_id}:{group}][{index}]"
            slot_id = self._slot_by_location[location]
            if not self._is_active(slot_id):
                continue
            label, confidence, latency_ms = self._infer(candidate_id, mic, geo)
            route_path.append(slot_id)
            invocations.append((slot_id, candidate_id, latency_ms))
            if confidence > self.thresholds[slot_id]:
                return self._finish(
                    label, slot_id, route_path, invocations, wall_started, False
                )
        return self._finish(
            true_label, DETECTOR_SENTINEL, route_path, invocations, wall_started, True
        )

    def _is_active(self, slot_id: str) -> bool:
        return self.active_slots is None or slot_id in self.active_slots

    def _finish(
        self,
        prediction: str,
        terminal_route: str,
        route_path: Sequence[str],
        invocations: Sequence[tuple[str, str, float]],
        wall_started: float,
        detector_used: bool,
    ) -> CascadeRun:
        _synchronize(self.device)
        return CascadeRun(
            prediction=prediction,
            terminal_route=terminal_route,
            route_path=tuple(route_path) + (("detector",) if detector_used else ()),
            invocations=tuple(invocations),
            measured_wall_ms=(time.perf_counter() - wall_started) * 1000.0,
            synthetic_detector_ms=self.detector_cost_ms if detector_used else 0.0,
        )

    def _infer(
        self, candidate_id: str, mic: torch.Tensor, geo: torch.Tensor
    ) -> tuple[str, float, float]:
        model = self.models[candidate_id]
        _synchronize(self.device)
        started = time.perf_counter()
        if KI_REGISTRY[candidate_id].modality == "mic":
            logits = model(mic)
        else:
            logits = model(mic, geo)
        confidence, class_index = torch.softmax(logits, dim=1).max(dim=1)
        _synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return (
            self.class_names[candidate_id][int(class_index.item())],
            float(confidence.item()),
            elapsed_ms,
        )


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


def _latency_summary(latencies_ms: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(latencies_ms, dtype=float)
    if len(values) == 0:
        raise ValueError("No timed samples were collected.")
    mean = float(values.mean())
    maximum = float(values.max())
    stdev = float(values.std())
    return {
        "samples": int(len(values)),
        "mean_ms": mean,
        "avg_ms": mean,
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "p99_ms": float(np.percentile(values, 99)),
        "max_ms": maximum,
        "wcet_ms": maximum,
        "min_ms": float(values.min()),
        "stdev_ms": stdev,
        "std_ms": stdev,
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


def benchmark_live_cascade(
    cascade: LiveCascade,
    mic: torch.Tensor,
    geo: torch.Tensor,
    true_labels: Sequence[str],
    registry: ClassifierRegistry,
    warmup_iterations: int,
) -> dict:
    """Run the selected policy once over every loaded testing sample."""
    if len(mic) != len(geo):
        raise ValueError("Mic and geo input counts differ.")
    if len(mic) != len(true_labels):
        raise ValueError("Live input count differs from the number of ground-truth labels.")
    if len(mic) == 0:
        raise ValueError("No live samples were loaded.")

    cascade.warmup_models(mic[:1], geo[:1], warmup_iterations)
    _synchronize(cascade.device)

    runs: list[CascadeRun] = []
    for index, truth in enumerate(np.asarray(true_labels, dtype=str)):
        runs.append(
            cascade.run(
                mic[index : index + 1],
                geo[index : index + 1],
                str(truth),
            )
        )

    invocation_counts_by_slot: Counter[str] = Counter()
    invocation_counts_by_model: Counter[str] = Counter()
    invocation_timings_by_slot: dict[str, list[float]] = defaultdict(list)
    invocation_timings_by_model: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for slot_id, model_id, latency_ms in run.invocations:
            invocation_counts_by_slot[slot_id] += 1
            invocation_counts_by_model[model_id] += 1
            invocation_timings_by_slot[slot_id].append(latency_ms)
            invocation_timings_by_model[model_id].append(latency_ms)

    detector_routes = sum(run.detector_used for run in runs)
    profile_total_ms = float(detector_routes * cascade.detector_cost_ms)
    for model_id, count in invocation_counts_by_model.items():
        record = registry.get(model_id)
        if record is None or record.runtime_ms is None:
            raise ValueError(f"Registry has no profiled runtime_ms for {model_id}.")
        profile_total_ms += count * float(record.runtime_ms)

    truth = np.asarray(true_labels, dtype=str)
    predictions = [run.prediction for run in runs]
    measured_model = [run.measured_model_ms for run in runs]
    measured_wall = [run.measured_wall_ms for run in runs]
    adjusted = [
        run.measured_model_ms + run.synthetic_detector_ms for run in runs
    ]
    end_to_end_adjusted = [
        run.measured_wall_ms + run.synthetic_detector_ms for run in runs
    ]
    terminal_counts = Counter(run.terminal_route for run in runs)
    route_path_counts = Counter(" -> ".join(run.route_path) for run in runs)

    return {
        **_accuracy_summary(predictions, truth),
        "measured_wall_latency": _latency_summary(measured_wall),
        "measured_model_latency": _latency_summary(measured_model),
        "costs": {
            "detector_cost_ms": cascade.detector_cost_ms,
            # Retain the historical key while making its model-only scope
            # explicit with an alias. The end-to-end value additionally
            # includes measured Python routing/dispatch overhead.
            "measured_detector_adjusted_expected_cost_ms": float(np.mean(adjusted)),
            "measured_model_detector_adjusted_expected_cost_ms": float(
                np.mean(adjusted)
            ),
            "measured_end_to_end_detector_adjusted_expected_cost_ms": float(
                np.mean(end_to_end_adjusted)
            ),
            "profiled_registry_expected_cost_ms": profile_total_ms / len(runs),
            "measured_real_model_total_ms": float(np.sum(measured_model)),
            "synthetic_detector_total_ms": float(
                detector_routes * cascade.detector_cost_ms
            ),
        },
        "routing": {
            "terminal_route_counts": dict(sorted(terminal_counts.items())),
            "route_path_counts": dict(sorted(route_path_counts.items())),
            "detector_routes": int(detector_routes),
            "detector_route_fraction": float(detector_routes / len(runs)),
            "invocation_counts_by_slot": dict(sorted(invocation_counts_by_slot.items())),
            "invocation_counts_by_model": dict(sorted(invocation_counts_by_model.items())),
        },
        "invocation_timing_by_slot": {
            key: _latency_summary(values)
            for key, values in sorted(invocation_timings_by_slot.items())
        },
        "invocation_timing_by_model": {
            key: _latency_summary(values)
            for key, values in sorted(invocation_timings_by_model.items())
        },
        "samples": [
            {
                "sample_index": index,
                "true_label": str(truth[index]),
                "prediction": run.prediction,
                "correct": bool(run.prediction == truth[index]),
                "terminal_route": run.terminal_route,
                "route_path": list(run.route_path),
                "measured_wall_ms": run.measured_wall_ms,
                "measured_model_ms": run.measured_model_ms,
                "synthetic_detector_ms": run.synthetic_detector_ms,
                "detector_adjusted_cost_ms": (
                    run.measured_model_ms + run.synthetic_detector_ms
                ),
                "end_to_end_detector_adjusted_cost_ms": (
                    run.measured_wall_ms + run.synthetic_detector_ms
                ),
            }
            for index, run in enumerate(runs)
        ],
    }


def _device_description(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(device)}"
    return str(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a saved GA or DP cascade policy with live K0-K6 inference "
            "and a non-sleeping synthetic detector."
        )
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument(
        "--saved-policy",
        choices=("winner", "dp_fixed_thresholds", "sa_on_dp_layout"),
        default="winner",
        help=(
            "Policy packet to execute: a joint-GA winner or one method from "
            "benchmark_full_candidate_dp_sa.py."
        ),
    )
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--scene", default=None, help="Infer from outcomes when omitted.")
    parser.add_argument(
        "--partition",
        choices=("auto", "all", "validation", "holdout"),
        default="holdout",
        help=(
            "Dataset partition to execute. 'holdout' reproduces the saved testing split."
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=25,
        help="Untimed forwards per reachable real model before the cascade run.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap; 0 runs every sample in the selected partition.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--detector-cost-ms", type=float, default=DEFAULT_DETECTOR_COST_MS
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="JSON benchmark report path.",
    )
    args = parser.parse_args()
    if args.warmup_iterations < 0 or args.max_samples < 0:
        parser.error("--warmup-iterations and --max-samples must be non-negative.")
    if args.detector_cost_ms < 0:
        parser.error("--detector-cost-ms must be non-negative.")

    summary = _load_json(args.summary)
    partition = _resolve_partition(summary, args.partition)
    layout, thresholds, active_slots, saved_policy = load_saved_policy(
        summary, args.saved_policy, partition
    )
    model_ids = active_model_ids(layout)

    device = resolve_device(args.device)
    models, registry, device = load_live_models(
        model_ids, args.checkpoint_dir, args.registry, device
    )
    live_inputs = load_live_inputs(
        args.outcomes,
        summary,
        args.scene,
        partition,
        args.processed_dir,
        device,
        args.max_samples,
        args.seed,
    )
    cascade = LiveCascade(
        layout,
        thresholds,
        models,
        registry,
        device,
        active_slots=active_slots,
        detector_cost_ms=args.detector_cost_ms,
    )
    benchmark = benchmark_live_cascade(
        cascade,
        live_inputs.mic,
        live_inputs.geo,
        live_inputs.true_labels,
        registry,
        args.warmup_iterations,
    )

    report = {
        "schema_version": "live-cascade-benchmark/v1",
        "scene": live_inputs.scene,
        "partition": partition,
        "environment": {
            "device": _device_description(device),
            "device_request": args.device,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "timing_scope": (
            "batch-size-1 live model forwards, softmax, and routing; CUDA is "
            "synchronized around every forward; excludes model loading, data "
            "loading, and host-to-device transfer"
        ),
        "detector": {
            "mode": "paper_oracle_non_sleeping",
            "cost_ms": args.detector_cost_ms,
            "behavior": "returns ground truth immediately and adds cost without sleeping",
        },
        "accuracy_scope": (
            "live predictions over every loaded sample after separate untimed warmup"
        ),
        "sources": {
            "summary": str(args.summary.resolve()),
            "outcomes": str(args.outcomes.resolve()),
            "registry": str(args.registry.resolve()),
        },
        "saved_policy": args.saved_policy,
        "configuration_partition": (
            "validation" if partition in {"all", "holdout"} else partition
        ),
        "empirical_reference_partition": partition,
        "layout": {
            "initial": list(layout.initial),
            "specialized": {
                f"{router_id}:{group}": list(chain)
                for (router_id, group), chain in layout.specialized.items()
            },
        },
        "thresholds": thresholds,
        "active_slots": None if active_slots is None else list(active_slots),
        "warmup_iterations_per_model": args.warmup_iterations,
        "available_samples": live_inputs.available_samples,
        "loaded_samples": int(len(live_inputs.mic)),
        "saved_empirical_reference": {
            key: saved_policy.get(key)
            for key in ("accuracy", "expected_cost", "route_counts", "total")
            if key in saved_policy
        },
        "benchmark": benchmark,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n")
    print(
        json.dumps(
            {
                "device": report["environment"]["device"],
                "samples": report["loaded_samples"],
                "accuracy": benchmark["accuracy"],
                "end_to_end_detector_adjusted_expected_cost_ms": benchmark["costs"][
                    "measured_end_to_end_detector_adjusted_expected_cost_ms"
                ],
                "profiled_registry_expected_cost_ms": benchmark["costs"][
                    "profiled_registry_expected_cost_ms"
                ],
                "detector_routes": benchmark["routing"]["detector_routes"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
