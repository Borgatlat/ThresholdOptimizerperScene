"""Resolve Ki checkpoint / weight file paths (robust across OS and working directory)."""

from __future__ import annotations

import hashlib
from pathlib import Path

KI_NAMES = [f"K{i}" for i in range(7)] + ["Kdet"]


def file_fingerprint(path: Path | str) -> dict[str, str | int]:
    """Return portable identity metadata for one checkpoint or artifact."""
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def normalize_checkpoint_dir(checkpoint_dir: Path | str) -> Path:
    return Path(checkpoint_dir).expanduser().resolve()


def repo_root_from_registry(registry_path: Path) -> Path:
    """Repo root = parent of checkpoints/ when registry lives at checkpoints/classifier_registry.json."""
    registry_path = Path(registry_path).expanduser().resolve()
    if registry_path.parent.name == "checkpoints":
        return registry_path.parent.parent
    return registry_path.parent


def checkpoint_candidates(
    ki_name: str,
    checkpoint_dir: Path,
    registry_path: Path | None = None,
) -> list[Path]:
    """Ordered search paths for one Ki (first existing file wins)."""
    checkpoint_dir = normalize_checkpoint_dir(checkpoint_dir)
    roots = [checkpoint_dir]
    if registry_path is not None:
        roots.append(repo_root_from_registry(registry_path) / "checkpoints")

    seen: set[Path] = set()
    candidates: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(path)

    for root in roots:
        root = normalize_checkpoint_dir(root)
        weights = root / "weights"
        for name in (
            f"{ki_name}_weights.pt",
            f"{ki_name}_weights.pth",
            f"{ki_name}.pt",
            f"{ki_name}.pth",
        ):
            add(root / name if "weights" not in name else weights / name)
        add(root / f"{ki_name}.pt")
        add(root / f"{ki_name}.pth")
        add(weights / f"{ki_name}_weights.pt")
        add(weights / f"{ki_name}_weights.pth")

    return candidates


def resolve_registry_checkpoint(
    checkpoint_field: str | None,
    ki_name: str,
    checkpoint_dir: Path,
    registry_path: Path | None = None,
) -> Path:
    """Map registry checkpoint string to an existing file on disk."""
    checkpoint_dir = normalize_checkpoint_dir(checkpoint_dir)

    if checkpoint_field:
        raw = Path(str(checkpoint_field).replace("\\", "/"))
        extra = [
            raw,
            checkpoint_dir / raw.name,
            checkpoint_dir / raw,
        ]
        if raw.parts and raw.parts[0] == "checkpoints":
            rest = Path(*raw.parts[1:])
            extra.extend(
                [
                    checkpoint_dir / rest,
                    checkpoint_dir / rest.name,
                ]
            )
        if registry_path is not None:
            repo = repo_root_from_registry(registry_path)
            extra.append(repo / raw)
            if raw.parts and raw.parts[0] == "checkpoints":
                extra.append(repo / Path(*raw.parts))

        for path in extra:
            try:
                if path.is_file():
                    return path.resolve()
            except OSError:
                continue

    for path in checkpoint_candidates(ki_name, checkpoint_dir, registry_path):
        if path.is_file():
            return path.resolve()

    tried = checkpoint_candidates(ki_name, checkpoint_dir, registry_path)
    msg = "\n  ".join(str(p) for p in tried[:8])
    raise FileNotFoundError(
        f"Cannot find weights for {ki_name}. Searched:\n  {msg}\n"
        f"Place {ki_name}.pt in {checkpoint_dir} or run: python repack_checkpoints.py"
    )


def relative_checkpoint_ref(ki_name: str) -> str:
    """Stable registry value: filename only, resolved via checkpoint_dir."""
    return f"{ki_name}.pt"
