"""Download/verify CIFAR-100 training data and save the fixed split indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.cifar100.data import DEFAULT_SPLIT_SEED
from experiments.cifar100.train import (
    DEFAULT_DATA_ROOT,
    DEFAULT_SPLIT_DIR,
    prepare_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    dataset, splits, npz_path, manifest_path = prepare_splits(
        data_root=args.data_root,
        split_dir=args.output_dir,
        download=args.download,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "official_training_samples": len(dataset),
                "train": len(splits.train),
                "model_selection": len(splits.model_selection),
                "cascade_validation": len(splits.cascade_validation),
                "indices": str(npz_path),
                "manifest": str(manifest_path),
                "official_test_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
