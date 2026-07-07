"""GPU-synchronized WCET / average latency profiling for Ki classifiers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.dual_modal_cnn import build_ki_model
from training.trainer import KiDataset, get_device, prepare_ki_arrays
from utils.labels import KI_REGISTRY


@torch.inference_mode()
def profile_ki_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    warmup_batches: int = 10,
    timed_batches: int = 100,
) -> dict[str, float]:
    """Measure per-forward latency with CUDA sync (RTS-compliant timing)."""
    model.eval()
    latencies_ms: list[float] = []
    seen = 0

    for batch in loader:
        if modality == "mic":
            x, _ = batch
            x = x.to(device, non_blocking=True)
            inputs = (x,)
        else:
            mic, geo, _ = batch
            mic = mic.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            inputs = (mic, geo)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if modality == "mic":
            _ = model(inputs[0])
        else:
            _ = model(inputs[0], inputs[1])

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        seen += 1
        if seen > warmup_batches:
            latencies_ms.append(elapsed_ms)
        if seen >= warmup_batches + timed_batches:
            break

    if not latencies_ms:
        return {"avg_ms": 0.0, "wcet_ms": 0.0, "p95_ms": 0.0, "batches": 0}

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "wcet_ms": float(arr.max()),
        "p95_ms": float(np.percentile(arr, 95)),
        "batches": int(len(arr)),
    }


def profile_ki_wcet(
    ki_name: str,
    processed_dir: Path,
    batch_size: int = 1,
    warmup_batches: int = 10,
    timed_batches: int = 100,
) -> dict:
    """Load Ki and report C̄_i (average) and WCET (max) inference latency in ms."""
    spec = KI_REGISTRY[ki_name]
    device = get_device()
    arrays = prepare_ki_arrays(spec, processed_dir)
    mic, geo, labels, metadata, class_names = arrays
    valid = labels >= 0
    mask = valid
    if spec.modality == "mic":
        ds = KiDataset(mic[mask], None, labels[mask], spec.modality, augment=False)
    else:
        ds = KiDataset(mic[mask], geo[mask], labels[mask], spec.modality, augment=False)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = build_ki_model(ki_name, len(class_names)).to(device)

    timing = profile_ki_inference(
        model,
        loader,
        device,
        spec.modality,
        warmup_batches=warmup_batches,
        timed_batches=timed_batches,
    )
    return {
        "ki": ki_name,
        "modality": spec.modality,
        "batch_size": batch_size,
        "num_classes": len(class_names),
        **timing,
    }
