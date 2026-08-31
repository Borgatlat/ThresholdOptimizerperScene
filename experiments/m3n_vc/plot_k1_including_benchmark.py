"""Plot the K1-enabled DP, threshold-on-DP, and joint-GA benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CHECKPOINT_DIR = Path(
    "checkpoints/k1_including_h24_with_run9_dp_target_paper_sa"
)


def _load_packets(checkpoint_dir: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
    dp_path = checkpoint_dir / "dp_layout_threshold_optimization.json"
    if not dp_path.is_file():
        dp_path = checkpoint_dir / "dp_and_threshold_summary.json"
    dp = json.loads(dp_path.read_text(encoding="utf-8"))
    ga = json.loads((checkpoint_dir / "ga" / "summary.json").read_text(encoding="utf-8"))
    for field in ("outcomes_sha256", "target_accuracy", "split_seed", "split_strategy"):
        if dp["settings"].get(field) != ga["settings"].get(field):
            raise ValueError(f"DP and GA packets disagree on {field}.")
    return dp, ga


def plot_comparison(checkpoint_dir: Path, output_path: Path) -> Path:
    dp, ga = _load_packets(checkpoint_dir)
    methods = (
        dp["methods"]["dp_fixed_thresholds"],
        dp["methods"]["sa_on_dp_layout"],
        ga["winner"],
    )
    labels = (
        "DP layout\n(fixed thresholds)",
        "DP layout\n(optimized thresholds)",
        "Joint genetic\noptimizer",
    )
    validation_costs = [float(method["validation"]["expected_cost"]) for method in methods]
    testing_costs = [float(method["holdout"]["expected_cost"]) for method in methods]
    validation_accuracies = [100.0 * float(method["validation"]["accuracy"]) for method in methods]
    testing_accuracies = [100.0 * float(method["holdout"]["accuracy"]) for method in methods]
    target = 100.0 * float(dp["target_accuracy"])

    x = np.arange(len(methods))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), layout="constrained")
    axes[0].bar(x - width / 2, validation_costs, width, label="Validation", color="#457B9D")
    axes[0].bar(x + width / 2, testing_costs, width, label="Testing", color="#A8DADC")
    axes[0].set_ylabel("Expected cost (ms)")
    axes[0].set_title("Expected cascade cost")
    axes[0].legend(frameon=False)
    axes[0].bar_label(axes[0].containers[0], fmt="%.1f", padding=3, fontsize=8.5)
    axes[0].bar_label(axes[0].containers[1], fmt="%.1f", padding=3, fontsize=8.5)

    axes[1].bar(x - width / 2, validation_accuracies, width, label="Validation", color="#457B9D")
    axes[1].bar(x + width / 2, testing_accuracies, width, label="Testing", color="#A8DADC")
    axes[1].axhline(target, color="#333333", linestyle="--", linewidth=1.1, label="Target")
    axes[1].set_ylim(
        min(validation_accuracies + testing_accuracies) - 0.15,
        max(validation_accuracies + testing_accuracies) + 0.15,
    )
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("End-to-end accuracy")
    axes[1].legend(frameon=False)
    axes[1].bar_label(axes[1].containers[0], fmt="%.3f%%", padding=3, fontsize=8.5)
    axes[1].bar_label(axes[1].containers[1], fmt="%.3f%%", padding=3, fontsize=8.5)

    for axis in axes:
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        f"K1-enabled h24 optimization (DP accuracy target = {target:.3f}%)",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220 if output_path.suffix.lower() == ".png" else None)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.checkpoint_dir / "optimizer_comparison_bar_chart.png"
    print(plot_comparison(args.checkpoint_dir, output))


if __name__ == "__main__":
    main()
