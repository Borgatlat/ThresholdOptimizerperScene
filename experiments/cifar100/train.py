"""Reproducible training for the independent CIFAR-100 cascade candidates.

Only rows from the official training split are loaded.  Checkpoint selection
uses the saved 2,500-row model-selection partition; the 5,000-row
cascade-validation partition is never passed to this module.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from experiments.cifar100.data import (
    DEFAULT_SPLIT_SEED,
    CIFAR100SplitIndices,
    build_dataset_view,
    generate_stratified_splits,
    load_split_bundle,
    load_training_dataset,
    save_split_bundle,
)
from experiments.cifar100.labels import CIFAR100_PROFILE
from experiments.cifar100.models import (
    CandidateRegistry,
    CandidateSpec,
    build_model,
    candidate_specs,
    config_hash,
    file_sha256,
    load_checkpoint_metadata,
    save_checkpoint,
    transfer_wrn_base_features,
)


DEFAULT_DATA_ROOT = Path("datasets/cifar100")
DEFAULT_SPLIT_DIR = Path("checkpoints/cifar100/splits")
DEFAULT_OUTPUT_DIR = Path("checkpoints/cifar100/training")
DEFAULT_CONFIG = Path(__file__).with_name("configs") / "default_training.json"
SMOKE_CONFIG = Path(__file__).with_name("configs") / "smoke_training.json"
TRAINING_MANIFEST_SCHEMA = "cifar100-training-manifest/v1"


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "loss": float(self.loss),
            "accuracy": float(self.accuracy),
            "samples": int(self.samples),
        }


def load_training_config(path: str | Path) -> dict[str, object]:
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    if config.get("schema_version") != "cifar100-training/v1":
        raise ValueError(f"Unsupported training configuration: {source}")
    return config


def set_reproducible_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _limit_dataset(
    dataset: Dataset,
    maximum: int | None,
    *,
    seed: int,
) -> Dataset:
    if maximum is None or maximum >= len(dataset):
        return dataset
    if maximum < 1:
        raise ValueError("Sample limits must be positive.")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(dataset), size=maximum, replace=False))
    return Subset(dataset, selected.tolist())


def _target_mode(spec: CandidateSpec) -> tuple[str, str | None]:
    if spec.role == "specialized":
        return "specialist", spec.group
    if spec.role == "intermediate":
        return "coarse", None
    return "fine", None


def build_candidate_datasets(
    dataset,
    splits: CIFAR100SplitIndices,
    spec: CandidateSpec,
    config: Mapping[str, object],
) -> tuple[Dataset, Dataset]:
    """Return training/model-selection views for one role only."""

    target_mode, group = _target_mode(spec)
    train_view = build_dataset_view(
        dataset,
        splits.train,
        target_mode=target_mode,
        group=group,
        train=True,
    )
    selection_view = build_dataset_view(
        dataset,
        splits.model_selection,
        target_mode=target_mode,
        group=group,
        train=False,
    )
    seed = int(config["seed"])
    return (
        _limit_dataset(
            train_view,
            _optional_int(config.get("max_train_samples")),
            seed=seed + 101,
        ),
        _limit_dataset(
            selection_view,
            _optional_int(config.get("max_validation_samples")),
            seed=seed + 202,
        ),
    )


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=_seed_worker if workers else None,
        generator=generator,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    correct = 0
    samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch_size = int(targets.numel())
            samples += batch_size
            loss_total += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=1) == targets).sum())
    if not samples:
        raise ValueError("A training or validation dataset is empty.")
    return EpochMetrics(loss_total / samples, correct / samples, samples)


def _role_epochs(spec: CandidateSpec, config: Mapping[str, object]) -> int:
    epochs = config.get("epochs")
    if not isinstance(epochs, Mapping):
        raise ValueError("Training config requires an epochs mapping.")
    key = "wrn16_2_base" if spec.role == "base_initializer" else spec.kind
    if key not in epochs and spec.role == "specialized":
        key = "specialized"
    value = int(epochs[key])
    if value < 1:
        raise ValueError("Every candidate must train for at least one epoch.")
    return value


def _learning_rate(
    epoch: int,
    epochs: int,
    *,
    base: float,
    minimum: float,
    warmup: int,
) -> float:
    if warmup and epoch < warmup:
        return base * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, epochs - warmup - 1)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))


def _resnet_initialization_provenance(pretrained: bool) -> dict[str, object]:
    if not pretrained:
        return {"source": "random", "external_checkpoint": None}
    from urllib.parse import urlparse

    from torchvision.models import ResNet18_Weights

    weights = ResNet18_Weights.DEFAULT
    filename = Path(urlparse(weights.url).path).name
    cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cached.is_file():
        raise FileNotFoundError(
            "torchvision reported successful pretrained initialization but its "
            f"cached checkpoint is missing: {cached}"
        )
    return {
        "source": "torchvision.ResNet18_Weights.DEFAULT",
        "url": weights.url,
        "checkpoint_path": str(cached.resolve()),
        "checkpoint_sha256": file_sha256(cached),
    }


def _effective_training_config(
    config: Mapping[str, object],
    spec: CandidateSpec,
    *,
    split_manifest: Path,
    initialization: Mapping[str, object],
) -> dict[str, object]:
    return {
        "base_config": dict(config),
        "candidate": spec.as_dict(),
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": file_sha256(split_manifest),
        "initialization": dict(initialization),
        "official_test_used": False,
        "cascade_validation_used_for_selection": False,
    }


def _load_reusable_candidate(
    checkpoint_path: Path,
    metrics_path: Path,
    spec: CandidateSpec,
    config: Mapping[str, object],
    *,
    split_manifest: Path,
) -> dict[str, object]:
    """Validate and return an existing run before initializing any model."""

    if not checkpoint_path.is_file() or not metrics_path.is_file():
        missing = checkpoint_path if not checkpoint_path.is_file() else metrics_path
        raise FileNotFoundError(f"Existing candidate artifact is incomplete: {missing}")
    metadata = load_checkpoint_metadata(
        checkpoint_path, expected_spec=spec, map_location="cpu"
    )
    stored = metadata.get("training_config")
    if not isinstance(stored, Mapping):
        raise ValueError(f"Checkpoint {checkpoint_path} has no training config.")
    stored_base = stored.get("base_config")
    if not isinstance(stored_base, Mapping) or config_hash(stored_base) != config_hash(
        config
    ):
        raise ValueError(
            f"Existing checkpoint {checkpoint_path} uses another base configuration; "
            "pass --force or choose another output directory."
        )
    if stored.get("candidate") != spec.as_dict():
        raise ValueError(f"Existing checkpoint {checkpoint_path} has another candidate.")
    if stored.get("split_manifest_sha256") != file_sha256(split_manifest):
        raise ValueError(
            f"Existing checkpoint {checkpoint_path} uses another data split; "
            "pass --force or choose another output directory."
        )
    if stored.get("official_test_used") is not False or stored.get(
        "cascade_validation_used_for_selection"
    ) is not False:
        raise ValueError(f"Checkpoint {checkpoint_path} violates data isolation.")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, dict):
        raise ValueError(f"Checkpoint metrics are not an object: {metrics_path}")
    if metrics.get("schema_version") != "cifar100-training-metrics/v1":
        raise ValueError(f"Unsupported checkpoint metrics: {metrics_path}")
    if metrics.get("candidate") != spec.as_dict():
        raise ValueError(f"Metrics candidate differs from {spec.candidate_id!r}.")
    if metrics.get("config_hash") != metadata.get("config_hash"):
        raise ValueError(f"Metrics/config hash mismatch for {spec.candidate_id!r}.")
    declared_sha = metrics.get("checkpoint_sha256")
    if declared_sha is not None and declared_sha != metadata["checkpoint_sha256"]:
        raise ValueError(f"Metrics/checkpoint hash mismatch for {spec.candidate_id!r}.")
    return metrics


def _build_optimizer(
    model: nn.Module, config: Mapping[str, object]
) -> torch.optim.Optimizer:
    optimizer_config = config.get("optimizer")
    if not isinstance(optimizer_config, Mapping) or optimizer_config.get("name") != "sgd":
        raise ValueError("Only the documented SGD optimizer is supported.")
    return torch.optim.SGD(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        momentum=float(optimizer_config.get("momentum", 0.9)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        nesterov=bool(optimizer_config.get("nesterov", False)),
    )


def train_candidate(
    dataset,
    splits: CIFAR100SplitIndices,
    spec: CandidateSpec,
    config: Mapping[str, object],
    *,
    split_manifest: Path,
    output_dir: Path,
    device: torch.device,
    base_checkpoint: Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Train one independent model and save its best-selection checkpoint."""

    candidate_dir = output_dir / spec.candidate_id
    checkpoint_path = candidate_dir / "best.pt"
    metrics_path = candidate_dir / "metrics.json"
    if checkpoint_path.exists() or metrics_path.exists():
        if not force:
            return _load_reusable_candidate(
                checkpoint_path,
                metrics_path,
                spec,
                config,
                split_manifest=split_manifest,
            )

    seed = int(config["seed"])
    set_reproducible_seed(seed)
    pretrained_resnet = bool(
        config.get("resnet18_imagenet_pretrained", False)
        and spec.architecture == "resnet18"
    )
    model = build_model(spec, pretrained=pretrained_resnet)
    initialization: dict[str, object]
    if spec.architecture == "wrn16_2" and spec.role != "base_initializer":
        if base_checkpoint is None or not base_checkpoint.is_file():
            raise FileNotFoundError(
                f"{spec.candidate_id} requires the independently trained "
                "wrn16_2_base checkpoint."
            )
        transfer = transfer_wrn_base_features(base_checkpoint, model)
        initialization = {
            "source": "local_wrn16_2_base",
            "checkpoint_path": str(base_checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(base_checkpoint),
            "copied_parameter_keys": len(transfer.copied_keys),
            "skipped_parameter_keys": list(transfer.skipped_keys),
        }
    elif spec.architecture == "resnet18":
        initialization = _resnet_initialization_provenance(pretrained_resnet)
    else:
        initialization = {"source": "random", "external_checkpoint": None}

    effective_config = _effective_training_config(
        config,
        spec,
        split_manifest=split_manifest,
        initialization=initialization,
    )

    train_dataset, selection_dataset = build_candidate_datasets(
        dataset, splits, spec, config
    )
    batch_size = int(config["batch_size"])
    workers = int(config.get("num_workers", 0))
    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=True,
        seed=seed + 303,
    )
    selection_loader = _make_loader(
        selection_dataset,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed + 404,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )
    optimizer = _build_optimizer(model, config)
    epoch_count = _role_epochs(spec, config)
    scheduler_config = config.get("scheduler")
    if not isinstance(scheduler_config, Mapping):
        raise ValueError("Training config requires scheduler settings.")
    base_lr = float(config["optimizer"]["learning_rate"])
    minimum_lr = float(scheduler_config.get("minimum_learning_rate", 0.0))
    warmup = int(scheduler_config.get("warmup_epochs", 0))
    patience = int(config.get("early_stopping_patience", epoch_count))

    history: list[dict[str, object]] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    started = perf_counter()
    for epoch in range(epoch_count):
        learning_rate = _learning_rate(
            epoch,
            epoch_count,
            base=base_lr,
            minimum=minimum_lr,
            warmup=warmup,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        selection_metrics = run_epoch(
            model, selection_loader, criterion, device, optimizer=None
        )
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "train": train_metrics.as_dict(),
                "model_selection": selection_metrics.as_dict(),
            }
        )
        improved = (
            selection_metrics.accuracy > best_accuracy
            or (
                selection_metrics.accuracy == best_accuracy
                and selection_metrics.loss < best_loss
            )
        )
        if improved:
            best_accuracy = selection_metrics.accuracy
            best_loss = selection_metrics.loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint state.")
    model.load_state_dict(best_state, strict=True)
    elapsed = perf_counter() - started
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metrics: dict[str, object] = {
        "schema_version": "cifar100-training-metrics/v1",
        "candidate": spec.as_dict(),
        "candidate_id": spec.candidate_id,
        "role": spec.role,
        "kind": spec.kind,
        "group": spec.group,
        "architecture": spec.architecture,
        "is_candidate": spec.is_candidate,
        "parameter_count": int(parameter_count),
        "input_resolution": [3, 32, 32],
        "train_samples": len(train_dataset),
        "model_selection_samples": len(selection_dataset),
        "best_epoch": best_epoch,
        "best_model_selection_accuracy": best_accuracy,
        "best_model_selection_loss": best_loss,
        "epochs_completed": len(history),
        "elapsed_seconds": elapsed,
        "history": history,
        "initialization": initialization,
        "official_test_used": False,
        "cascade_validation_used_for_selection": False,
        "config_hash": config_hash(effective_config),
        "base_training_config_hash": config_hash(config),
        "split_manifest_sha256": file_sha256(split_manifest),
        "training_environment": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor() or platform.machine() or "unknown"
            ),
            "pytorch_version": str(torch.__version__),
            "cpu_threads": torch.get_num_threads(),
        },
    }
    save_checkpoint(
        checkpoint_path,
        model,
        spec,
        effective_config,
        metrics={
            "best_epoch": best_epoch,
            "model_selection_accuracy": best_accuracy,
            "model_selection_loss": best_loss,
        },
        extra={"initialization": initialization},
    )
    metrics["checkpoint_path"] = str(checkpoint_path.resolve())
    metrics["checkpoint_sha256"] = file_sha256(checkpoint_path)
    metrics["metrics_path"] = str(metrics_path.resolve())
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metrics


def prepare_splits(
    *,
    data_root: Path,
    split_dir: Path,
    download: bool,
    seed: int,
) -> tuple[object, CIFAR100SplitIndices, Path, Path]:
    dataset = load_training_dataset(data_root, download=download)
    npz_path = split_dir / "cifar100_split_indices.npz"
    manifest_path = split_dir / "cifar100_split_indices.json"
    if npz_path.is_file() and manifest_path.is_file():
        splits = load_split_bundle(
            npz_path,
            manifest_path=manifest_path,
            fine_targets=dataset.fine_targets,
            coarse_targets=dataset.coarse_targets,
        )
        if splits.seed != seed:
            raise ValueError(
                f"Existing splits use seed {splits.seed}, requested seed is {seed}."
            )
    else:
        splits = generate_stratified_splits(dataset.fine_targets, seed=seed)
        npz_path, manifest_path = save_split_bundle(
            splits,
            split_dir,
            fine_targets=dataset.fine_targets,
            coarse_targets=dataset.coarse_targets,
            source_files=dataset.raw_files,
        )
    return dataset, splits, npz_path, manifest_path


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    return device


def _record_candidate_id(record: Mapping[str, object]) -> str:
    candidate = record.get("candidate")
    if not isinstance(candidate, Mapping) or candidate.get("candidate_id") is None:
        raise ValueError("Training record has no candidate identity.")
    return str(candidate["candidate_id"])


def _previous_manifest_records(
    manifest_path: Path,
    *,
    config: Mapping[str, object],
    split_npz: Path,
    split_manifest: Path,
) -> list[dict[str, object]]:
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != TRAINING_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported existing training manifest: {manifest_path}")
    expected = {
        "dataset_id": CIFAR100_PROFILE.dataset_id,
        "profile_fingerprint": CIFAR100_PROFILE.fingerprint,
        "official_test_used": False,
        "training_config_hash": config_hash(config),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Existing training manifest {manifest_path} has another {key}."
            )
    declared_npz_sha = manifest.get("split_indices_sha256")
    if declared_npz_sha is not None and declared_npz_sha != file_sha256(split_npz):
        raise ValueError("Existing training manifest uses another split index bundle.")
    declared_manifest_sha = manifest.get("split_manifest_sha256")
    if (
        declared_manifest_sha is not None
        and declared_manifest_sha != file_sha256(split_manifest)
    ):
        raise ValueError("Existing training manifest uses another split manifest.")
    raw_records = manifest.get("models", [])
    if not isinstance(raw_records, list) or not all(
        isinstance(item, Mapping) for item in raw_records
    ):
        raise ValueError("Existing training manifest has invalid model records.")
    return [dict(item) for item in raw_records]


def write_training_manifest(
    records: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
    split_npz: Path,
    split_manifest: Path,
    output_dir: Path,
    device: str,
    merge_existing: bool = True,
) -> dict[str, object]:
    """Write one stable manifest, optionally retaining compatible prior jobs."""

    registry = candidate_specs(
        CIFAR100_PROFILE.groups, CIFAR100_PROFILE.global_classes
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "training_manifest.json"
    combined = (
        _previous_manifest_records(
            manifest_path,
            config=config,
            split_npz=split_npz,
            split_manifest=split_manifest,
        )
        if merge_existing
        else []
    )
    combined.extend(dict(record) for record in records)
    by_id: dict[str, dict[str, object]] = {}
    for record in combined:
        candidate_id = _record_candidate_id(record)
        if candidate_id not in registry.by_id:
            raise ValueError(f"Unknown training record {candidate_id!r}.")
        if record.get("candidate") != registry.by_id[candidate_id].as_dict():
            raise ValueError(f"Training record spec mismatch for {candidate_id!r}.")
        by_id[candidate_id] = record

    spec_order = (
        registry.wrn_base_initializer,
        *registry.candidates,
        registry.detector,
    )
    ordered = [by_id[spec.candidate_id] for spec in spec_order if spec.candidate_id in by_id]
    deployable_ids = {item.candidate_id for item in registry.candidates}
    candidate_records = [
        record
        for record in ordered
        if _record_candidate_id(record) in deployable_ids
    ]
    initializer_records = [
        record
        for record in ordered
        if _record_candidate_id(record)
        == registry.wrn_base_initializer.candidate_id
    ]
    detector_records = [
        record
        for record in ordered
        if _record_candidate_id(record) == registry.detector.candidate_id
    ]
    expected_candidate_ids = {item.candidate_id for item in registry.candidates}
    manifest: dict[str, object] = {
        "schema_version": TRAINING_MANIFEST_SCHEMA,
        "dataset_id": CIFAR100_PROFILE.dataset_id,
        "profile_fingerprint": CIFAR100_PROFILE.fingerprint,
        "official_test_used": False,
        "split_indices": str(split_npz.resolve()),
        "split_indices_sha256": file_sha256(split_npz),
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": file_sha256(split_manifest),
        "training_config": dict(config),
        "training_config_hash": config_hash(config),
        "device": device,
        "candidate_count": len(candidate_records),
        "complete_candidate_set": {
            _record_candidate_id(item) for item in candidate_records
        }
        == expected_candidate_ids,
        "candidates": candidate_records,
        "initializer": initializer_records[0] if initializer_records else None,
        "detector": detector_records[0] if detector_records else None,
        "models": ordered,
    }
    temporary = manifest_path.with_name(
        f"{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    return manifest


def rebuild_training_manifest(
    *,
    config: Mapping[str, object],
    split_npz: Path,
    split_manifest: Path,
    output_dir: Path,
    device: str = "unknown",
) -> dict[str, object]:
    """Rebuild a manifest from independently produced candidate artifacts."""

    registry = candidate_specs(
        CIFAR100_PROFILE.groups, CIFAR100_PROFILE.global_classes
    )
    records: list[dict[str, object]] = []
    for spec in (
        registry.wrn_base_initializer,
        *registry.candidates,
        registry.detector,
    ):
        candidate_dir = output_dir / spec.candidate_id
        checkpoint_path = candidate_dir / "best.pt"
        metrics_path = candidate_dir / "metrics.json"
        if not checkpoint_path.exists() and not metrics_path.exists():
            continue
        records.append(
            _load_reusable_candidate(
                checkpoint_path,
                metrics_path,
                spec,
                config,
                split_manifest=split_manifest,
            )
        )
    if not records:
        raise FileNotFoundError(f"No candidate artifacts found under {output_dir}.")
    return write_training_manifest(
        records,
        config=config,
        split_npz=split_npz,
        split_manifest=split_manifest,
        output_dir=output_dir,
        device=device,
        merge_existing=False,
    )


def train_requested(
    requested: Sequence[str],
    *,
    config: Mapping[str, object],
    data_root: Path,
    split_dir: Path,
    output_dir: Path,
    download: bool,
    device: torch.device,
    force: bool,
) -> dict[str, object]:
    registry: CandidateRegistry = candidate_specs(
        CIFAR100_PROFILE.groups, CIFAR100_PROFILE.global_classes
    )
    if not requested or requested == ("all",) or requested == ["all"]:
        specs = (registry.wrn_base_initializer, *registry.candidates)
    else:
        unknown = set(requested) - set(registry.by_id)
        if unknown:
            raise ValueError(f"Unknown candidate ids: {sorted(unknown)}")
        specs = tuple(registry.by_id[item] for item in requested)
        if any(spec.role == "detector" for spec in specs):
            raise ValueError(
                "Train the ConvNeXt endpoint with "
                "python -m experiments.cifar100.train_detector."
            )

    seed = int(config.get("seed", DEFAULT_SPLIT_SEED))
    dataset, splits, split_npz, split_manifest = prepare_splits(
        data_root=data_root,
        split_dir=split_dir,
        download=download,
        seed=seed,
    )
    base_checkpoint = output_dir / registry.wrn_base_initializer.candidate_id / "best.pt"
    records: list[dict[str, object]] = []
    for spec in specs:
        record = train_candidate(
            dataset,
            splits,
            spec,
            config,
            split_manifest=split_manifest,
            output_dir=output_dir,
            device=device,
            base_checkpoint=(
                None if spec.role == "base_initializer" else base_checkpoint
            ),
            force=force,
        )
        records.append(record)

    return write_training_manifest(
        records,
        config=config,
        split_npz=split_npz,
        split_manifest=split_manifest,
        output_dir=output_dir,
        device=str(device),
        merge_existing=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        nargs="+",
        default=["all"],
        help="Candidate ids, wrn16_2_base, or all.",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config_path = args.config or (SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG)
    config = load_training_config(config_path)
    if args.cpu_threads is not None:
        if args.cpu_threads < 1:
            raise ValueError("--cpu-threads must be positive.")
        torch.set_num_threads(args.cpu_threads)
    output_dir = args.output_dir
    if args.smoke and output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = Path("checkpoints/cifar100/smoke/training")
    manifest = train_requested(
        args.candidate,
        config=config,
        data_root=args.data_root,
        split_dir=args.split_dir,
        output_dir=output_dir,
        download=args.download,
        device=_resolve_device(args.device),
        force=args.force,
    )
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "training_manifest.json"),
                "candidate_models": len(manifest["candidates"]),
                "models_in_manifest": len(manifest["models"]),
                "complete_candidate_set": manifest["complete_candidate_set"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
