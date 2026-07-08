"""Pre-flight checks before running run_all_scenes.py."""

from __future__ import annotations

import sys
from pathlib import Path

from checkpoint_paths import KI_NAMES, checkpoint_candidates
from process_data import KNOWN_SCENES, resolve_scene_raw_dir

CHECKPOINT_DIR = Path("checkpoints")
REGISTRY = CHECKPOINT_DIR / "classifier_registry.json"


def check_checkpoints() -> list[str]:
    errors: list[str] = []
    if not REGISTRY.is_file():
        errors.append(f"Missing registry: {REGISTRY}")
        return errors
    for ki in KI_NAMES:
        paths = checkpoint_candidates(ki, CHECKPOINT_DIR, REGISTRY)
        if not any(p.is_file() for p in paths):
            errors.append(f"No weights found for {ki} (searched {paths[0].parent}/...)")
    return errors


def check_scene_data(datasets_root: Path) -> dict[str, str]:
    status: dict[str, str] = {}
    for scene in KNOWN_SCENES:
        try:
            raw = resolve_scene_raw_dir(scene, datasets_root)
            n_mic = len(list(raw.glob("*_mic.parquet")))
            status[scene] = f"ok ({n_mic} mic files @ {raw})"
        except FileNotFoundError as exc:
            status[scene] = f"missing: {exc}"
    return status


def main() -> int:
    print("=== Checkpoint verification ===")
    ckpt_errors = check_checkpoints()
    if ckpt_errors:
        for err in ckpt_errors:
            print(f"  FAIL: {err}")
    else:
        print(f"  OK: all {len(KI_NAMES)} Ki weights found under {CHECKPOINT_DIR}/")

    print("\n=== Scene raw data (optional until you run preprocessing) ===")
    root = Path("datasets")
    for scene, msg in check_scene_data(root).items():
        print(f"  {scene:>4}: {msg}")

    print("\n=== Processed cache ===")
    proc = root / "processed"
    if proc.exists():
        for scene in KNOWN_SCENES:
            meta = proc / f"{scene}_metadata.parquet"
            print(f"  {scene:>4}: {'ready' if meta.is_file() else 'not built'}")
    else:
        print("  (no datasets/processed/ yet — run_all_scenes will create it)")

    if ckpt_errors:
        print("\nFix checkpoint errors before running empirical_outcomes.")
        return 1
    print("\nCheckpoints OK. Download M3N-VC scenes, then: python run_all_scenes.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
