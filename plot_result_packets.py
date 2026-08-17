"""Generate dataset-neutral cost and routing figures from result packets only."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from result_packets import load_result_packet


def _partition(packet: Mapping[str, object], name: str) -> Mapping[str, object]:
    partitions = packet["partitions"]
    assert isinstance(partitions, Mapping)
    result = partitions[name]
    assert isinstance(result, Mapping)
    return result


def _label(packet: Mapping[str, object]) -> str:
    method = packet["method"]
    assert isinstance(method, Mapping)
    return str(method["label"])


def plot_packets(packet_paths: Sequence[Path], output_dir: Path) -> None:
    packets = [load_result_packet(path) for path in packet_paths]
    if not packets:
        raise ValueError("At least one result packet is required.")
    dataset_profiles = {
        (
            str(packet["dataset"]["id"]),  # type: ignore[index]
            str(packet["dataset"]["profile_fingerprint"]),  # type: ignore[index]
        )
        for packet in packets
    }
    if len(dataset_profiles) != 1:
        raise ValueError("All plotted packets must use the same dataset profile.")
    dataset_id = next(iter(dataset_profiles))[0]
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [_label(packet) for packet in packets]
    positions = np.arange(len(packets))

    for partition in ("validation", "test"):
        metrics = [_partition(packet, partition) for packet in packets]
        costs = [float(item["expected_cost_ms"]) for item in metrics]
        accuracies = [float(item["accuracy"]) for item in metrics]
        figure, axis = plt.subplots(figsize=(max(8.0, len(packets) * 1.8), 5.4))
        bars = axis.bar(positions, costs, color="#4C78A8", edgecolor="#333333")
        axis.set_title(f"{dataset_id} - {partition} expected cost")
        axis.set_ylabel("Expected cost (ms)")
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        for bar, cost, accuracy in zip(bars, costs, accuracies, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{cost:,.1f} ms\n{accuracy:.2%}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        figure.tight_layout()
        figure.savefig(output_dir / f"{partition}_cost.png", dpi=240)
        figure.savefig(output_dir / f"{partition}_cost.pdf")
        plt.close(figure)

        route_ids = sorted(
            {
                str(route)
                for item in metrics
                for route in dict(item["routes"])
            }
        )
        bottoms = np.zeros(len(packets), dtype=float)
        figure, axis = plt.subplots(figsize=(max(8.0, len(packets) * 1.8), 5.7))
        colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(route_ids)))
        for route_id, color in zip(route_ids, colors, strict=True):
            values = np.asarray(
                [
                    100.0
                    * int(dict(item["routes"]).get(route_id, 0))
                    / int(item["samples"])
                    for item in metrics
                ]
            )
            axis.bar(
                positions,
                values,
                bottom=bottoms,
                label=route_id,
                color=color,
                edgecolor="white",
            )
            bottoms += values
        axis.set_title(f"{dataset_id} - {partition} terminal routing")
        axis.set_ylabel("Share of samples")
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.set_ylim(0, 100)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        axis.legend(title="Final decision", frameon=False, ncol=min(6, len(route_ids)))
        figure.tight_layout()
        figure.savefig(output_dir / f"{partition}_routing.png", dpi=240)
        figure.savefig(output_dir / f"{partition}_routing.pdf")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packets", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_packets(args.packets, args.output_dir)


if __name__ == "__main__":
    main()
