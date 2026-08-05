"""Rebuild the CIFAR-100 training manifest after independent/HPC jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.cifar100.train import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SPLIT_DIR,
    SMOKE_CONFIG,
    load_training_config,
    rebuild_training_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device-description",
        default="independent jobs; see per-candidate metrics",
    )
    args = parser.parse_args()

    config_path = args.config or (SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG)
    output_dir = args.output_dir
    if args.smoke and output_dir == DEFAULT_OUTPUT_DIR:
        output_dir = Path("checkpoints/cifar100/smoke/training")
    manifest = rebuild_training_manifest(
        config=load_training_config(config_path),
        split_npz=args.split_dir / "cifar100_split_indices.npz",
        split_manifest=args.split_dir / "cifar100_split_indices.json",
        output_dir=output_dir,
        device=args.device_description,
    )
    print(
        json.dumps(
            {
                "manifest": str((output_dir / "training_manifest.json").resolve()),
                "candidate_models": len(manifest["candidates"]),
                "complete_candidate_set": manifest["complete_candidate_set"],
                "initializer_available": manifest["initializer"] is not None,
                "detector_available": manifest["detector"] is not None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
