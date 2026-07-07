"""Load trained Ki classifiers from checkpoint or flat weight files."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from cascade.checkpoint_paths import resolve_registry_checkpoint
from models.dual_modal_cnn import build_ki_model
from training.trainer import get_device
from utils.classifier_registry import ClassifierRegistry
from utils.labels import KI_REGISTRY


def load_state_dict_from_file(path: Path, device: torch.device | None = None) -> dict:
    """Load a state_dict from .pt / .pth (full checkpoint or flat weights)."""
    dev = device or torch.device("cpu")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Weight file not found: {path.resolve()}")

    ckpt = torch.load(path, map_location=dev, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt

    raise ValueError(
        f"{path} does not contain loadable weights. "
        f"Expected 'model_state_dict' or a flat state_dict. "
        f"Keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}. "
        f"Run: python repack_checkpoints.py"
    )


def load_state_dict_for_ki(
    ki_name: str,
    checkpoint_dir: Path,
    *,
    registry_path: Path | None = None,
    checkpoint_field: str | None = None,
    device: torch.device | None = None,
) -> tuple[dict, Path]:
    """Find and load weights for one Ki; returns (state_dict, path_used)."""
    path = resolve_registry_checkpoint(
        checkpoint_field,
        ki_name,
        checkpoint_dir,
        registry_path,
    )
    return load_state_dict_from_file(path, device), path


def load_cascade_models(
    checkpoint_dir: Path,
    registry_path: Path,
) -> tuple[dict[str, nn.Module], ClassifierRegistry, torch.device]:
    """Load all Ki models listed in KI_REGISTRY using registry metadata."""
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    registry_path = Path(registry_path).expanduser().resolve()
    registry = ClassifierRegistry.load(registry_path)
    device = get_device()
    models: dict[str, nn.Module] = {}

    for ki_name in KI_REGISTRY:
        rec = registry.get(ki_name)
        if rec is None:
            raise ValueError(f"No registry record for {ki_name}")

        state_dict, used_path = load_state_dict_for_ki(
            ki_name,
            checkpoint_dir,
            registry_path=registry_path,
            checkpoint_field=rec.checkpoint,
            device=device,
        )

        model = build_ki_model(ki_name, len(rec.class_names)).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models[ki_name] = model
        print(f"Loaded {ki_name} from {used_path} ({sum(t.numel() for t in state_dict.values()):,} params)")

    return models, registry, device
