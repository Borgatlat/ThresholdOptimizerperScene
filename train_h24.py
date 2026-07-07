"""Preprocess h24 M3N-VC data and train a global Ki classifier on spectrograms."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.spectrogram_cnn import SpectrogramCNN, count_parameters
from process_data import save_h24_spectrogram_arrays

# Paper global labels for run pairs in h24 (run0/1, run2/3, ...).
GLOBAL_CLASS_NAMES = ["gle350", "cx30", "mustang", "miata", "background"]

DEFAULT_H24_DIR = Path("datasets/h24/h24")
DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


def get_device() -> torch.device:
    """Pick GPU when available; otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_spectrograms(x: np.ndarray) -> np.ndarray:
    """Scale each spectrogram to zero mean / unit variance for stable training."""
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((x - mean) / std).astype(np.float32)


def labels_to_zero_index(labels: np.ndarray) -> np.ndarray:
    """Convert 1..5 labels from process_data into 0..4 for PyTorch."""
    return (labels.astype(np.int64) - 1)


def train_val_split(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Random split by segment (not by run) for a quick baseline."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(x))
    split = int(len(indices) * (1.0 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    """One pass over the dataset; optimizer=None means evaluation mode."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        if is_train:
            optimizer.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        if is_train:
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
        _ = (time.perf_counter() - start) * 1000.0

        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def preprocess_h24(data_dir: Path, output_dir: Path) -> None:
    """Convert raw parquet into cached .npy spectrogram arrays."""
    print(f"Preprocessing h24 from {data_dir} -> {output_dir}")
    mic, geo = save_h24_spectrogram_arrays(
        output_dir=output_dir,
        data_dir=data_dir,
    )
    mic_x, mic_y = mic
    geo_x, geo_y = geo
    print(f"Mic: {mic_x.shape}, labels: {mic_y.shape}, classes: {sorted(set(mic_y))}")
    print(f"Geo: {geo_x.shape}, labels: {geo_y.shape}, classes: {sorted(set(geo_y))}")


def load_processed(modality: str, processed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load cached spectrograms for mic or geo."""
    suffix = "mic" if modality == "mic" else "geo"
    x_path = processed_dir / f"h24_{suffix}_spectrograms.npy"
    y_path = processed_dir / f"h24_{suffix}_labels.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            f"Missing {x_path} or {y_path}. Run: python train_h24.py --preprocess"
        )
    return np.load(x_path), np.load(y_path)


def train_classifier(
    modality: str,
    processed_dir: Path,
    checkpoint_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    """Train global Ki on h24 spectrograms."""
    device = get_device()
    print(f"Device: {device}")

    x_raw, y_raw = load_processed(modality, processed_dir)
    x_raw = normalize_spectrograms(x_raw)
    y_raw = labels_to_zero_index(y_raw)

    x_train, y_train, x_val, y_val = train_val_split(x_raw, y_raw)

    # Add channel dimension: (N, H, W) -> (N, 1, H, W) for Conv2d.
    x_train_t = torch.from_numpy(x_train[:, None, :, :])
    x_val_t = torch.from_numpy(x_val[:, None, :, :])
    y_train_t = torch.from_numpy(y_train)
    y_val_t = torch.from_numpy(y_val)

    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
    )

    num_classes = len(GLOBAL_CLASS_NAMES)
    model = SpectrogramCNN(num_classes=num_classes).to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict] = []
    best_val_acc = 0.0
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"k_global_{modality}.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": GLOBAL_CLASS_NAMES,
                    "modality": modality,
                    "val_acc": val_acc,
                },
                ckpt_path,
            )

    summary = {
        "modality": modality,
        "num_train": int(len(x_train)),
        "num_val": int(len(x_val)),
        "best_val_acc": best_val_acc,
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    metrics_path = checkpoint_dir / f"k_global_{modality}_metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ki on h24 M3N-VC spectrograms")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Convert parquet files to cached .npy spectrograms",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the global classifier on cached spectrograms",
    )
    parser.add_argument(
        "--modality",
        choices=["mic", "geo", "both"],
        default="mic",
        help="Sensor modality to train on (both trains two separate models)",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_H24_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.preprocess and not args.train:
        print("Nothing to do. Use --preprocess and/or --train.")
        print("Example: python train_h24.py --preprocess --train --modality mic")
        return

    if args.preprocess:
        preprocess_h24(args.data_dir, args.processed_dir)

    if args.train:
        modalities = ["mic", "geo"] if args.modality == "both" else [args.modality]
        for modality in modalities:
            train_classifier(
                modality=modality,
                processed_dir=args.processed_dir,
                checkpoint_dir=args.checkpoint_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )


if __name__ == "__main__":
    main()
