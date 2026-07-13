"""Plot paper-Kdet baseline-target threshold-optimization reports.

For every scene report, this creates:
* a baseline-versus-annealed accuracy/expected-cost bar chart; and
* a holdout route-distribution bar chart paired with threshold bars.

The values used for plotting are also collected into ``plot_data.json`` next
to the reports, so figures do not become the only record of the experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_REPORTS_DIR = Path("checkpoints/paper_kdet_baseline_target")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/paper_kdet_baseline_target")
POLICIES = (("baseline", "Baseline"), ("annealing", "Annealed"))
PARTITIONS = ("validation", "holdout")
COLORS = {"Baseline": "#457b9d", "Annealed": "#e76f51"}


def _load_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def _partition_metrics(policy: Mapping[str, object], partition: str) -> Mapping[str, object]:
    """Read current reports and the pre-rename ``optimization`` reports."""
    if partition in policy and isinstance(policy[partition], Mapping):
        return policy[partition]
    if partition == "validation":
        legacy = policy.get("optimization")
        if isinstance(legacy, Mapping):
            return legacy
    raise KeyError(f"Policy has no {partition!r} metrics.")


def _policy_metrics(report: Mapping[str, object], policy_key: str) -> Mapping[str, object]:
    policy = report.get(policy_key)
    if not isinstance(policy, Mapping):
        raise KeyError(f"Report has no {policy_key!r} policy.")
    return policy


def _normalised_routes(metrics: Mapping[str, object]) -> dict[str, float]:
    counts = metrics.get("route_counts", {})
    if not isinstance(counts, Mapping):
        raise ValueError("route_counts must be an object.")
    total = float(sum(int(value) for value in counts.values()))
    if total == 0.0:
        return {str(route): 0.0 for route in counts}
    return {str(route): 100.0 * int(count) / total for route, count in counts.items()}


def _figure_metrics(scene: str, report: Mapping[str, object], output_path: Path) -> None:
    values = {
        label: {
            partition: _partition_metrics(_policy_metrics(report, policy), partition)
            for partition in PARTITIONS
        }
        for policy, label in POLICIES
    }
    positions = np.arange(len(PARTITIONS))
    width = 0.36

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for offset, (_, label) in zip((-width / 2, width / 2), POLICIES, strict=True):
        accuracy = [100.0 * float(values[label][partition]["accuracy"]) for partition in PARTITIONS]
        axes[0].bar(positions + offset, accuracy, width, label=label, color=COLORS[label])
        cost = [float(values[label][partition]["expected_cost"]) for partition in PARTITIONS]
        axes[1].bar(positions + offset, cost, width, label=label, color=COLORS[label])

    axes[0].set_title("End-to-end accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0.0, 100.0)
    axes[1].set_title("Expected cascade cost")
    axes[1].set_ylabel("Expected cost (ms)")
    for axis in axes:
        axis.set_xticks(positions, [partition.title() for partition in PARTITIONS])
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    target = 100.0 * float(report["target_accuracy"])
    figure.suptitle(f"{scene}: paper-Kdet policy metrics (target {target:.2f}%)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _figure_policy(scene: str, report: Mapping[str, object], output_path: Path) -> None:
    baseline = _policy_metrics(report, "baseline")
    annealed = _policy_metrics(report, "annealing")
    baseline_holdout = _partition_metrics(baseline, "holdout")
    annealed_holdout = _partition_metrics(annealed, "holdout")
    baseline_routes = _normalised_routes(baseline_holdout)
    annealed_routes = _normalised_routes(annealed_holdout)
    route_ids = sorted(set(baseline_routes) | set(annealed_routes))

    baseline_thresholds = baseline_holdout.get("thresholds", {})
    annealed_thresholds = annealed_holdout.get("thresholds", {})
    if not isinstance(baseline_thresholds, Mapping) or not isinstance(annealed_thresholds, Mapping):
        raise ValueError("thresholds must be objects.")
    model_ids = sorted(set(baseline_thresholds) | set(annealed_thresholds))
    positions_routes = np.arange(len(route_ids))
    positions_models = np.arange(len(model_ids))
    width = 0.36

    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2), constrained_layout=True)
    axes[0].bar(
        positions_routes - width / 2,
        [baseline_routes.get(route_id, 0.0) for route_id in route_ids],
        width,
        label="Baseline",
        color=COLORS["Baseline"],
    )
    axes[0].bar(
        positions_routes + width / 2,
        [annealed_routes.get(route_id, 0.0) for route_id in route_ids],
        width,
        label="Annealed",
        color=COLORS["Annealed"],
    )
    axes[0].set_title("Holdout terminal-route distribution")
    axes[0].set_ylabel("Samples (%)")
    axes[0].set_xticks(positions_routes, route_ids)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(
        positions_models - width / 2,
        [float(baseline_thresholds.get(model_id, np.nan)) for model_id in model_ids],
        width,
        label="Baseline",
        color=COLORS["Baseline"],
    )
    axes[1].bar(
        positions_models + width / 2,
        [float(annealed_thresholds.get(model_id, np.nan)) for model_id in model_ids],
        width,
        label="Annealed",
        color=COLORS["Annealed"],
    )
    axes[1].set_title("Selected confidence thresholds")
    axes[1].set_ylabel("Threshold")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xticks(positions_models, model_ids)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    figure.suptitle(f"{scene}: paper-Kdet routing and thresholds")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _collect_scene_data(scene: str, report: Mapping[str, object]) -> dict:
    collected: dict[str, object] = {
        "target_accuracy": report["target_accuracy"],
        "target_accuracy_source": report.get("target_accuracy_source"),
        "policies": {},
    }
    for policy_key, label in POLICIES:
        policy = _policy_metrics(report, policy_key)
        policy_data: dict[str, object] = {}
        for partition in PARTITIONS:
            metrics = _partition_metrics(policy, partition)
            policy_data[partition] = {
                "accuracy": metrics["accuracy"],
                "expected_cost": metrics["expected_cost"],
                "route_percent": _normalised_routes(metrics),
                "thresholds": metrics["thresholds"],
            }
        collected["policies"][label.lower()] = policy_data  # type: ignore[index]
    return collected


def plot_reports(
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
) -> dict:
    report_paths = sorted(
        path for path in reports_dir.glob("*.json") if path.name != "summary.json"
    )
    if not report_paths:
        raise FileNotFoundError(f"No scene reports found in {reports_dir}.")

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_data: dict[str, object] = {"reports_dir": str(reports_dir), "scenes": {}}
    for report_path in report_paths:
        report = _load_json(report_path)
        scene = str(report.get("scene", report_path.stem))
        metrics_path = figures_dir / f"{scene}_metrics.png"
        policy_path = figures_dir / f"{scene}_routes_thresholds.png"
        _figure_metrics(scene, report, metrics_path)
        _figure_policy(scene, report, policy_path)
        plot_data["scenes"][scene] = _collect_scene_data(scene, report)  # type: ignore[index]
        print(f"[{scene}] wrote {metrics_path} and {policy_path}")

    data_path = reports_dir / "plot_data.json"
    data_path.write_text(json.dumps(plot_data, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {data_path}")
    return plot_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-scene figures from paper-Kdet optimization reports."
    )
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    args = parser.parse_args()
    plot_reports(args.reports_dir, args.figures_dir)


if __name__ == "__main__":
    main()
