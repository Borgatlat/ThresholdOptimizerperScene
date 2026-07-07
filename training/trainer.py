"""Training loop with GPU-ready hooks, early stopping, and macro-F1 selection."""

from __future__ import annotations

import json
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from models.dual_modal_cnn import build_ki_model
from models.spectrogram_cnn import count_parameters
from training.augment import apply_spec_augment
from training.losses import LOSS_KEYS, build_loss, loss_uses_class_weights
from training.metrics import compute_metrics
from utils.classifier_registry import ClassifierRegistry, compute_p_idk
from utils.labels import KI_REGISTRY, KiSpec, is_deterministic_ki, threshold_hi_for_ki
from utils.splits import apply_background_val_holdout, run_level_masks





def normalize_spectrograms(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def _label_column_for_ki(spec: KiSpec) -> str:
    if spec.level == "intermediate":
        return "intermediate_label"
    return "global_label"


def _name_to_index(class_names: list[str]) -> dict[str, int]:
    """String label -> integer class id (used by vectorized pandas .map)."""
    return {name: idx for idx, name in enumerate(class_names)}


def get_ki_labels(metadata: pd.DataFrame, spec: KiSpec) -> np.ndarray:
    """Per-row class indices; -1 marks rows outside this Ki's training subset."""
    name_to_idx = _name_to_index(spec.class_names)
    labels = np.full(len(metadata), -1, dtype=np.int64)
    if spec.subset == "all":
        col = _label_column_for_ki(spec)
        labels[:] = metadata[col].astype(str).map(name_to_idx).to_numpy()
        return labels

    if spec.subset == "suv":
        mask = metadata["intermediate_label"].eq("suv").to_numpy()
        labels[mask] = (
            metadata.loc[mask, "global_label"].astype(str).map(name_to_idx).to_numpy()
        )
        return labels

    if spec.subset == "coupe":
        mask = metadata["intermediate_label"].eq("coupe").to_numpy()
        labels[mask] = (
            metadata.loc[mask, "global_label"].astype(str).map(name_to_idx).to_numpy()
        )
        return labels

    raise ValueError(f"Unknown subset: {spec.subset}")


def load_spectrogram_cache(
    processed_dir: Path,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load normalized mic/geo + metadata once (reuse across K0–K6 in train-all)."""
    norm_mic = processed_dir / "h24_paired_mic_norm.npy"
    norm_geo = processed_dir / "h24_paired_geo_norm.npy"
    meta_path = processed_dir / "h24_metadata.parquet"

    if norm_mic.exists() and norm_geo.exists():
        mic = np.load(norm_mic)
        geo = np.load(norm_geo)
    else:
        mic = normalize_spectrograms(np.load(processed_dir / "h24_paired_mic.npy"))
        geo = normalize_spectrograms(np.load(processed_dir / "h24_paired_geo.npy"))
        np.save(norm_mic, mic)
        np.save(norm_geo, geo)

    metadata = pd.read_parquet(meta_path)
    return mic, geo, metadata


def prepare_ki_arrays(
    spec: KiSpec,
    processed_dir: Path,
    cache: tuple[np.ndarray, np.ndarray, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    if cache is not None:
        mic, geo, metadata = cache
    else:
        mic, geo, metadata = load_spectrogram_cache(processed_dir)

    labels = get_ki_labels(metadata, spec)
    return mic, geo, labels, metadata, spec.class_names


def _class_sample_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights so each class is seen equally often per epoch."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    per_class = 1.0 / counts
    return torch.from_numpy(per_class[labels].astype(np.float64))


class KiDataset(Dataset):
    def __init__(
        self,
        mic: np.ndarray,
        geo: np.ndarray | None,
        labels: np.ndarray,
        modality: str,
        augment: bool = False,
    ) -> None:
        self.mic = torch.from_numpy(mic[:, None, :, :])
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.modality = modality
        self.augment = augment
        self.geo = (
            torch.from_numpy(geo[:, None, :, :]) if modality != "mic" and geo is not None else None
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        mic = self.mic[idx]
        if self.augment:
            mic = apply_spec_augment(mic)
        if self.modality == "mic":
            return mic, self.labels[idx]
        assert self.geo is not None
        geo = self.geo[idx]
        if self.augment:
            geo = apply_spec_augment(geo)
        return mic, geo, self.labels[idx]


def build_loaders(
    spec: KiSpec,
    processed_dir: Path,
    batch_size: int,
    loss_key: str = "ce",
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]] | None = None,
    augment_train: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, list[str], np.ndarray, np.ndarray, np.ndarray]:
    if arrays is None:
        mic, geo, labels, metadata, class_names = prepare_ki_arrays(spec, processed_dir)
    else:
        mic, geo, labels, metadata, class_names = arrays

    valid = labels >= 0
    train_mask, val_mask, _ = run_level_masks(metadata, spec=spec)
    train_mask = train_mask & valid
    val_mask = val_mask & valid

    # Background (run8) segment holdout → val so macro-F1 is not punished by missing class.
    if "background" in class_names:
        train_mask, val_mask = apply_background_val_holdout(
            metadata, train_mask, val_mask, seed=seed
        )

    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise ValueError(f"No train/val samples for {spec.name} subset={spec.subset}")

    train_labels = labels[train_mask]
    if spec.modality == "mic":
        train_ds = KiDataset(
            mic[train_mask], None, train_labels, spec.modality, augment=augment_train
        )
        val_ds = KiDataset(mic[val_mask], None, labels[val_mask], spec.modality, augment=False)
    else:
        train_ds = KiDataset(
            mic[train_mask],
            geo[train_mask],
            train_labels,
            spec.modality,
            augment=augment_train,
        )
        val_ds = KiDataset(
            mic[val_mask], geo[val_mask], labels[val_mask], spec.modality, augment=False
        )

    pin = torch.cuda.is_available()
    num_workers = 2 if pin and sys.platform != "win32" else 0
    loader_kw: dict = dict(batch_size=batch_size, pin_memory=pin, num_workers=num_workers)
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2

    # weighted_ce / focal already up-weight rare classes — skip sampler to avoid double balance.
    if loss_uses_class_weights(loss_key):
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    else:
        sample_weights = _class_sample_weights(train_labels, len(class_names))
        train_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=False, **loader_kw)

    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    return train_loader, val_loader, class_names, train_mask, val_mask, labels


@dataclass
class TrainResult:
    ki: str
    loss_key: str
    best_val_macro_f1: float
    best_val_loss: float
    num_params: int
    checkpoint: str
    history: list[dict] = field(default_factory=list)
    best_metrics: dict = field(default_factory=dict)
    inference_avg_ms: float | None = None
    inference_wcet_ms: float | None = None
    p_idk: float | None = None
    threshold_hi: float | None = None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    modality: str,
    scaler: torch.amp.GradScaler | None,
    sync_timing: bool = False,
) -> tuple[float, dict]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    num_samples = 0
    # Collect GPU tensors per batch; one CPU transfer at epoch end (faster than .tolist()).
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    use_amp = scaler is not None and device.type == "cuda"
    # inference_mode disables autograd on val — less memory, slightly faster forwards.
    forward_ctx = nullcontext() if is_train else torch.inference_mode()

    for batch in loader:
        if modality == "mic":
            x, y = batch
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            inputs = (x,)
        else:
            mic, geo, y = batch
            mic = mic.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            inputs = (mic, geo)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        if sync_timing and device.type == "cuda":
            torch.cuda.synchronize()

        with forward_ctx:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if modality == "mic":
                    logits = model(inputs[0])
                else:
                    logits = model(inputs[0], inputs[1])
                loss = criterion(logits, y)

        if is_train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        if sync_timing and device.type == "cuda":
            torch.cuda.synchronize()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
        all_preds.append(logits.argmax(dim=1).detach())
        all_targets.append(y.detach())

    avg_loss = total_loss / max(num_samples, 1)
    y_pred = torch.cat(all_preds).cpu().numpy()
    y_true = torch.cat(all_targets).cpu().numpy()
    return avg_loss, {"y_true": y_true, "y_pred": y_pred}


def _build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup then cosine decay — stabilizes early epochs on imbalanced Ki."""
    warmup_epochs = min(max(warmup_epochs, 0), max(epochs - 1, 0))
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warmup_epochs, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def _read_train_log_text(log_path: Path) -> str:
    """Windows Tee-Object writes UTF-16; fallback to UTF-8 for other logs."""
    raw = log_path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _last_epoch_from_train_log(log_path: Path, ki_name: str) -> int:
    """Parse the highest completed epoch from a Ki training log (crash-recovery fallback)."""
    if not log_path.exists():
        return 0
    pattern = re.compile(rf"^{re.escape(ki_name)} ep (\d+)/")
    last = 0
    for line in _read_train_log_text(log_path).splitlines():
        match = pattern.match(line.strip())
        if match:
            last = max(last, int(match.group(1)))
    return last


def _write_train_metrics(
    metrics_path: Path,
    *,
    ki_name: str,
    loss_key: str,
    best_f1: float,
    best_loss: float,
    num_params: int,
    ckpt_path: Path,
    history: list[dict],
    best_metrics: dict,
    threshold_hi: float | None,
) -> None:
    """Persist metrics after each epoch so a crash does not rewind resume state."""
    partial = TrainResult(
        ki=ki_name,
        loss_key=loss_key,
        best_val_macro_f1=best_f1,
        best_val_loss=best_loss,
        num_params=num_params,
        checkpoint=str(ckpt_path),
        history=history,
        best_metrics=best_metrics,
        threshold_hi=threshold_hi,
    )
    metrics_path.write_text(json.dumps(partial.__dict__, indent=2))


def train_ki(
    ki_name: str,
    processed_dir: Path,
    checkpoint_dir: Path,
    loss_key: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int = 5,
    seed: int = 42,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, list[str]] | None = None,
    warmup_epochs: int = 3,
    augment_train: bool = True,
    profile_inference: bool = False,
    use_torch_compile: bool = True,
    threshold_hi: float | None = None,
    registry: ClassifierRegistry | None = None,
    resume_from_checkpoint: bool = False,
) -> TrainResult:


    if threshold_hi is None and not is_deterministic_ki(ki_name):
        threshold_hi = threshold_hi_for_ki(ki_name)

    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = KI_REGISTRY[ki_name]
    device = get_device()
    print(f"{ki_name}: device={device}, loss={loss_key}")
    if arrays is None:
        arrays = prepare_ki_arrays(spec, processed_dir)
    train_loader, val_loader, class_names, train_mask, _, labels = build_loaders(
        spec,
        processed_dir,
        batch_size,
        loss_key=loss_key,
        arrays=arrays,
        augment_train=augment_train,
        seed=seed,
    )

    train_label_tensor = torch.from_numpy(labels[train_mask].astype(np.int64))

    model = build_ki_model(ki_name, len(class_names)).to(device)
    if use_torch_compile and device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)
    criterion = build_loss(loss_key, len(class_names), train_label_tensor, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = _build_lr_scheduler(optimizer, epochs, warmup_epochs)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{ki_name}.pt"
    metrics_path = checkpoint_dir / f"{ki_name}_metrics.json"
    train_log_path = checkpoint_dir / f"{ki_name}_train.log"

    best_f1 = -1.0
    best_loss = float("inf")
    best_metrics: dict = {}
    history: list[dict] = []
    stale_epochs = 0
    start_epoch = 1

    if resume_from_checkpoint and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        best_f1 = float(ckpt.get("val_macro_f1", -1.0))
        if metrics_path.exists():
            prior = json.loads(metrics_path.read_text())
            history = prior.get("history", [])
            best_metrics = prior.get("best_metrics", {})
            best_loss = float(prior.get("best_val_loss", float("inf")))
            start_epoch = len(history) + 1
        log_epoch = _last_epoch_from_train_log(train_log_path, ki_name)
        if log_epoch >= start_epoch:
            start_epoch = log_epoch + 1
        print(
            f"{ki_name}: resumed from {ckpt_path} "
            f"(epoch {start_epoch - 1}, best F1 {best_f1:.3f})"
        )

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        train_loss, train_raw = run_epoch(
            model, train_loader, criterion, optimizer, device, spec.modality, scaler
        )
        val_loss, val_raw = run_epoch(
            model, val_loader, criterion, None, device, spec.modality, scaler
        )
        scheduler.step()

        train_m = compute_metrics(train_raw["y_true"], train_raw["y_pred"], class_names)
        val_m = compute_metrics(val_raw["y_true"], val_raw["y_pred"], class_names)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_m["accuracy"],
            "val_acc": val_m["accuracy"],
            "train_macro_f1": train_m["macro_f1"],
            "val_macro_f1": val_m["macro_f1"],
            "val_macro_f1_present": val_m["macro_f1_present"],
            "elapsed_s": time.perf_counter() - t0,
        }
        history.append(row)
        print(
            f"{ki_name} ep {epoch}/{epochs} loss={loss_key} | "
            f"train L {train_loss:.4f} F1 {train_m['macro_f1']:.3f} | "
            f"val L {val_loss:.4f} F1 {val_m['macro_f1_present']:.3f} "
            f"(full {val_m['macro_f1']:.3f}) | "
            f"val recall {val_m['per_class_recall']}"
        )

        # Checkpoint on macro-F1 over classes actually present in val (fair vs paper metric).
        score = val_m["macro_f1_present"]
        if score > best_f1 or (score == best_f1 and val_loss < best_loss):
            best_f1 = val_m["macro_f1_present"]
            best_loss = val_loss
            best_metrics = val_m
            stale_epochs = 0
            torch.save(
                {
                    "ki": ki_name,
                    "loss_key": loss_key,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "modality": spec.modality,
                    "val_macro_f1": best_f1,
                    "val_macro_f1_full": val_m["macro_f1"],
                },
                ckpt_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"{ki_name}: early stop at epoch {epoch}")
                break

        _write_train_metrics(
            metrics_path,
            ki_name=ki_name,
            loss_key=loss_key,
            best_f1=best_f1,
            best_loss=best_loss,
            num_params=count_parameters(model),
            ckpt_path=ckpt_path,
            history=history,
            best_metrics=best_metrics,
            threshold_hi=threshold_hi,
        )

    # Restore best checkpoint weights (early stop leaves worse last-epoch weights in memory).
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    avg_ms: float | None = None
    wcet_ms: float | None = None
    if profile_inference:
        from training.profile import profile_ki_inference

        timing = profile_ki_inference(
            model, val_loader, device, spec.modality, warmup_batches=5, timed_batches=50
        )
        avg_ms = timing["avg_ms"]
        wcet_ms = timing["wcet_ms"]
        print(f"{ki_name}: inference avg={avg_ms:.2f}ms WCET={wcet_ms:.2f}ms (val loader batch)")

    # P{IDK}: fraction of val samples with max(softmax) < H_i (precision-calibrated deferral).

    if is_deterministic_ki(ki_name):
        p_idk = 0.0
        print(f"{ki_name}: deterministic fallback (p_idk=0.0, no H_i)")
    else:
        p_idk = compute_p_idk(model, val_loader, device, spec.modality, threshold_hi)
        print(f"{ki_name}: p_idk={p_idk:.4f} at H_i={threshold_hi}")

    result = TrainResult(
        ki=ki_name,
        loss_key=loss_key,
        best_val_macro_f1=best_f1,
        best_val_loss=best_loss,
        num_params=count_parameters(model),
        checkpoint=str(ckpt_path),
        history=history,
        best_metrics=best_metrics,
        inference_avg_ms=avg_ms,
        inference_wcet_ms=wcet_ms,
        p_idk=p_idk,
        threshold_hi=threshold_hi,
    )
    metrics_path.write_text(json.dumps(result.__dict__, indent=2))

    if registry is not None:
        registry.upsert_from_train_result(result, threshold_hi=threshold_hi)

    return result


def benchmark_losses(
    ki_name: str,
    processed_dir: Path,
    checkpoint_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    spec = KI_REGISTRY[ki_name]
    # Load + normalize once; reuse across all loss variants in this benchmark.
    arrays = prepare_ki_arrays(spec, processed_dir)
    results: dict[str, dict] = {}
    for loss_key in LOSS_KEYS:
        print(f"\n=== Benchmark {ki_name} loss={loss_key} ===")
        try:
            out = train_ki(
                ki_name=ki_name,
                processed_dir=processed_dir,
                checkpoint_dir=checkpoint_dir / "benchmark" / ki_name / loss_key,
                loss_key=loss_key,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=epochs,
                arrays=arrays,
            )
            results[loss_key] = {
                "val_macro_f1": out.best_val_macro_f1,
                "val_loss": out.best_val_loss,
                "num_params": out.num_params,
            }
        except Exception as exc:
            results[loss_key] = {"error": str(exc)}

    ranked = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda item: (-item[1]["val_macro_f1"], item[1]["val_loss"]),
    )
    best_loss = ranked[0][0] if ranked else "ce"
    summary = {"ki": ki_name, "best_loss": best_loss, "results": results}
    out_path = checkpoint_dir / f"loss_benchmark_{ki_name}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{ki_name} best loss: {best_loss}")
    return summary
