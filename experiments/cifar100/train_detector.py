"""Train only a CIFAR-100 head on a frozen pretrained ConvNeXt V2-L."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Mapping

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.cifar100.data import (
    build_convnext_evaluation_transform,
    build_dataset_view,
)
from experiments.cifar100.labels import CIFAR100_PROFILE
from experiments.cifar100.models import (
    CONVNEXT_V2_LARGE_ORIGIN,
    CONVNEXT_V2_LARGE_PRETRAINED_MODEL,
    CandidateSpec,
    build_model,
    candidate_specs,
    config_hash,
    file_sha256,
    save_checkpoint,
)
from experiments.cifar100.train import (
    DEFAULT_CONFIG,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_DIR,
    SMOKE_CONFIG,
    _effective_training_config,
    _limit_dataset,
    _load_reusable_candidate,
    _make_loader,
    _resolve_device,
    load_training_config,
    prepare_splits,
    set_reproducible_seed,
    write_training_manifest,
)


def _settings(config: Mapping[str, object]) -> Mapping[str, object]:
    value = config.get("detector")
    if not isinstance(value, Mapping):
        raise ValueError("Training config requires detector settings.")
    if value.get("pretrained_model") != CONVNEXT_V2_LARGE_PRETRAINED_MODEL:
        raise ValueError("Detector config names an unsupported pretrained model.")
    if int(value.get("input_resolution", 0)) != 224:
        raise ValueError("ConvNeXt V2-L requires 224px input preprocessing.")
    return value


def _classifier_prefixes(model: nn.Module) -> tuple[str, ...]:
    pretrained_cfg = getattr(model, "pretrained_cfg", {})
    raw = (
        pretrained_cfg.get("classifier", "head.fc")
        if isinstance(pretrained_cfg, Mapping)
        else "head.fc"
    )
    values = raw if isinstance(raw, (tuple, list)) else (raw,)
    return tuple(str(value) for value in values)


def backbone_state_sha256(model: nn.Module) -> str:
    """Fingerprint the exact loaded external backbone, excluding its new head."""

    prefixes = _classifier_prefixes(model)
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        if any(key == prefix or key.startswith(f"{prefix}.") for prefix in prefixes):
            continue
        normalized = tensor.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(normalized.dtype).encode("ascii"))
        digest.update(str(tuple(normalized.shape)).encode("ascii"))
        digest.update(normalized.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def pretrained_provenance(model: nn.Module) -> dict[str, object]:
    raw_cfg = getattr(model, "pretrained_cfg", {})
    cfg = dict(raw_cfg) if isinstance(raw_cfg, Mapping) else {}
    return {
        "source": "timm pretrained weights originating from Meta ConvNeXt-V2",
        "model_name": CONVNEXT_V2_LARGE_PRETRAINED_MODEL,
        "origin_url": str(cfg.get("origin_url", CONVNEXT_V2_LARGE_ORIGIN)),
        "weights_url": cfg.get("url"),
        "hf_hub_id": cfg.get("hf_hub_id"),
        "license": cfg.get("license", "cc-by-nc-4.0"),
        "input_size": list(cfg.get("input_size", (3, 224, 224))),
        "mean": list(cfg.get("mean", (0.485, 0.456, 0.406))),
        "std": list(cfg.get("std", (0.229, 0.224, 0.225))),
        "classifier": list(_classifier_prefixes(model)),
        "pretrained_backbone_state_sha256": backbone_state_sha256(model),
        "fine_tuning": "linear_head_only; backbone frozen",
    }


@torch.inference_mode()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract deterministic pre-logit features once from the frozen backbone."""

    model.to(device)
    model.eval()
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        feature_maps = model.forward_features(inputs)
        pre_logits = model.forward_head(feature_maps, pre_logits=True)
        if pre_logits.ndim != 2:
            raise ValueError("ConvNeXt pre-logits must have shape [batch, features].")
        features.append(pre_logits.detach().cpu().to(torch.float32))
        targets.append(labels.detach().cpu().to(torch.int64))
    if not features:
        raise ValueError("Detector feature extraction received no samples.")
    return torch.cat(features), torch.cat(targets)


def _head_epoch(
    head: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float | int]:
    head.train(optimizer is not None)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count = int(targets.numel())
            total_samples += count
            total_loss += float(loss.detach()) * count
            total_correct += int((logits.argmax(dim=1) == targets).sum())
    if not total_samples:
        raise ValueError("Detector head training received no feature rows.")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples": total_samples,
    }


def train_detector(
    dataset,
    splits,
    config: Mapping[str, object],
    *,
    split_manifest: Path,
    output_dir: Path,
    device: torch.device,
    force: bool = False,
) -> dict[str, object]:
    """Fit a new 100-way linear head without updating pretrained features."""

    registry = candidate_specs(
        CIFAR100_PROFILE.groups, CIFAR100_PROFILE.global_classes
    )
    spec: CandidateSpec = registry.detector
    candidate_dir = output_dir / spec.candidate_id
    checkpoint_path = candidate_dir / "best.pt"
    metrics_path = candidate_dir / "metrics.json"
    if (checkpoint_path.exists() or metrics_path.exists()) and not force:
        return _load_reusable_candidate(
            checkpoint_path,
            metrics_path,
            spec,
            config,
            split_manifest=split_manifest,
        )

    settings = _settings(config)
    seed = int(config["seed"])
    set_reproducible_seed(seed)
    model = build_model(spec, pretrained=True)
    initialization = pretrained_provenance(model)
    effective_config = _effective_training_config(
        config,
        spec,
        split_manifest=split_manifest,
        initialization=initialization,
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.get_classifier()
    if not isinstance(head, nn.Module):
        raise TypeError("ConvNeXt V2-L did not expose a classifier module.")
    for parameter in head.parameters():
        parameter.requires_grad_(True)

    transform = build_convnext_evaluation_transform()
    train_view = build_dataset_view(
        dataset, splits.train, target_mode="fine", transform=transform
    )
    selection_view = build_dataset_view(
        dataset, splits.model_selection, target_mode="fine", transform=transform
    )
    train_view = _limit_dataset(
        train_view,
        None if config.get("max_train_samples") is None else int(config["max_train_samples"]),
        seed=seed + 701,
    )
    selection_view = _limit_dataset(
        selection_view,
        (
            None
            if config.get("max_validation_samples") is None
            else int(config["max_validation_samples"])
        ),
        seed=seed + 702,
    )
    feature_batch_size = int(settings["feature_batch_size"])
    workers = int(config.get("num_workers", 0))
    train_loader = _make_loader(
        train_view,
        batch_size=feature_batch_size,
        workers=workers,
        shuffle=False,
        seed=seed + 703,
    )
    selection_loader = _make_loader(
        selection_view,
        batch_size=feature_batch_size,
        workers=workers,
        shuffle=False,
        seed=seed + 704,
    )
    started = perf_counter()
    train_features, train_targets = extract_features(
        model, train_loader, device=device
    )
    selection_features, selection_targets = extract_features(
        model, selection_loader, device=device
    )

    head_batch_size = int(settings["head_batch_size"])
    feature_train_loader = _make_loader(
        TensorDataset(train_features, train_targets),
        batch_size=head_batch_size,
        workers=0,
        shuffle=True,
        seed=seed + 705,
    )
    feature_selection_loader = _make_loader(
        TensorDataset(selection_features, selection_targets),
        batch_size=head_batch_size,
        workers=0,
        shuffle=False,
        seed=seed + 706,
    )
    optimizer_settings = settings.get("optimizer")
    if not isinstance(optimizer_settings, Mapping) or optimizer_settings.get(
        "name"
    ) != "adamw":
        raise ValueError("Detector head optimizer must be AdamW.")
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(optimizer_settings["learning_rate"]),
        weight_decay=float(optimizer_settings.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    epochs = int(settings["epochs"])
    patience = int(settings["early_stopping_patience"])
    history: list[dict[str, object]] = []
    best_accuracy = -1.0
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    head.to(device)
    for epoch in range(epochs):
        training = _head_epoch(
            head, feature_train_loader, criterion, device, optimizer
        )
        selection = _head_epoch(
            head, feature_selection_loader, criterion, device, None
        )
        history.append(
            {"epoch": epoch + 1, "train": training, "model_selection": selection}
        )
        accuracy = float(selection["accuracy"])
        loss = float(selection["loss"])
        if accuracy > best_accuracy or (accuracy == best_accuracy and loss < best_loss):
            best_accuracy = accuracy
            best_loss = loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Detector head training produced no checkpoint.")
    head.load_state_dict(best_state, strict=True)
    model.eval()
    elapsed = perf_counter() - started

    metrics: dict[str, object] = {
        "schema_version": "cifar100-training-metrics/v1",
        "candidate": spec.as_dict(),
        "candidate_id": spec.candidate_id,
        "role": spec.role,
        "kind": spec.kind,
        "group": None,
        "architecture": spec.architecture,
        "is_candidate": True,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in head.parameters())
        ),
        "input_resolution": [3, 224, 224],
        "train_samples": len(train_view),
        "model_selection_samples": len(selection_view),
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    dataset, splits, split_npz, split_manifest = prepare_splits(
        data_root=args.data_root,
        split_dir=args.split_dir,
        download=args.download,
        seed=int(config["seed"]),
    )
    record = train_detector(
        dataset,
        splits,
        config,
        split_manifest=split_manifest,
        output_dir=output_dir,
        device=_resolve_device(args.device),
        force=args.force,
    )
    manifest = write_training_manifest(
        [record],
        config=config,
        split_npz=split_npz,
        split_manifest=split_manifest,
        output_dir=output_dir,
        device=str(args.device),
        merge_existing=True,
    )
    print(
        json.dumps(
            {
                "checkpoint": record["checkpoint_path"],
                "training_manifest": str(output_dir / "training_manifest.json"),
                "model_selection_accuracy": record[
                    "best_model_selection_accuracy"
                ],
                "detector_available": manifest["detector"] is not None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
