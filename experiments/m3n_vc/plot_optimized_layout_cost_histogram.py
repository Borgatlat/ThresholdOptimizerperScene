"""Plot optimized validation-cost distributions from an exhaustive layout run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CHECKPOINT_DIR = Path("checkpoints/k1_free_full_benchmark_h24_target_090")


def _load_records(path: Path) -> list[dict[str, object]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [int(record["layout_index"]) for record in records]
    if not records or len(indices) != len(set(indices)):
        raise ValueError(f"{path} is empty or contains duplicate layout indices.")
    return records


def plot_histogram(
    checkpoint_dir: Path,
    output_path: Path,
    *,
    zoom_max_ms: float = 20.0,
) -> Path:
    summary_path = checkpoint_dir / "summary.json"
    results_path = checkpoint_dir / "layout_results.jsonl"
    summary: Mapping[str, object] = json.loads(summary_path.read_text(encoding="utf-8"))
    records = _load_records(results_path)

    validations = [record["validation"] for record in records]
    feasible_costs = np.asarray(
        [
            float(validation["expected_cost"])
            for validation in validations
            if bool(validation["feasible"])
        ],
        dtype=float,
    )
    if feasible_costs.size == 0 or np.any(feasible_costs <= 0.0):
        raise ValueError("At least one positive, feasible optimized cost is required.")
    if zoom_max_ms <= float(feasible_costs.min()):
        raise ValueError("zoom_max_ms must exceed the minimum optimized cost.")

    methods = summary["methods"]
    joint_cost = float(methods["exhaustive_joint"]["validation"]["expected_cost"])
    target_accuracy = 100.0 * float(summary["target_accuracy"])
    zoom_costs = feasible_costs[feasible_costs <= zoom_max_ms]

    figure, axes = plt.subplots(1, 2, figsize=(12.2, 5.3))
    full_bins = np.geomspace(
        float(feasible_costs.min()) * 0.995,
        float(feasible_costs.max()) * 1.005,
        48,
    )
    axes[0].hist(feasible_costs, bins=full_bins, color="#4C78A8", edgecolor="white")
    axes[0].set_xscale("log")
    axes[0].set_title("All optimized layouts")
    axes[0].set_xlabel("Best validation cost per layout (ms, log scale)")
    axes[0].set_ylabel("Number of layouts")

    zoom_min_ms = float(np.floor(feasible_costs.min()))
    zoom_span_ms = zoom_max_ms - zoom_min_ms
    zoom_bin_width_ms = max(
        0.25,
        float(np.ceil((zoom_span_ms / 60.0) * 4.0) / 4.0),
    )
    zoom_bins = np.arange(
        zoom_min_ms,
        zoom_max_ms + zoom_bin_width_ms,
        zoom_bin_width_ms,
    )
    axes[1].hist(zoom_costs, bins=zoom_bins, color="#72B7B2")
    axes[1].set_title(f"Low-cost region (≤ {zoom_max_ms:g} ms)")
    axes[1].set_xlabel("Best validation cost per layout (ms)")
    axes[1].set_ylabel("Number of layouts")
    axes[1].set_xlim(zoom_min_ms, zoom_max_ms)
    tick_step_ms = max(1.0, float(np.ceil((zoom_max_ms - zoom_min_ms) / 10.0)))
    axes[1].set_xticks(
        np.arange(zoom_min_ms, zoom_max_ms + 1e-9, tick_step_ms)
    )

    for axis in axes:
        axis.axvline(
            joint_cost,
            color="#D62728",
            linewidth=1.8,
            label=f"Joint optimum: {joint_cost:.3f} ms",
        )
        axis.grid(axis="y", alpha=0.22)
        axis.legend(frameon=False, fontsize=9)

    feasible_count = int(feasible_costs.size)
    axes[0].text(
        0.98,
        0.95,
        "\n".join(
            (
                f"Median: {np.median(feasible_costs):.3f} ms",
                f"Standard deviation: {np.std(feasible_costs):.3f} ms",
                f"90th percentile: {np.quantile(feasible_costs, 0.9):.3f} ms",
            )
        ),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC"},
    )
    axes[1].text(
        0.98,
        0.95,
        f"{zoom_costs.size:,} layouts ({100.0 * zoom_costs.size / feasible_count:.1f}%)",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#CCCCCC"},
    )

    figure.suptitle(
        f"Effect of cascade layout after threshold optimization\n"
        f"h24, K1-free, {target_accuracy:g}% validation-accuracy target",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        "Each value is one layout's best validation cost after best-of-10, "
        "1,000-iteration Chellapilla SA. Testing data are not used.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--zoom-max-ms", type=float, default=20.0)
    args = parser.parse_args()
    output = args.output or args.checkpoint_dir / "optimized_layout_cost_histogram.pdf"
    print(plot_histogram(args.checkpoint_dir, output, zoom_max_ms=args.zoom_max_ms))


if __name__ == "__main__":
    main()
