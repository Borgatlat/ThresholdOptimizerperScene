"""Plot the five h24 hierarchy methods used in the joint-search comparison.

The comparison is intentionally narrow and reads only completed experiment
artifacts:

* the preset-threshold DP hierarchy;
* that same hierarchy after threshold annealing;
* the annealed linear K3 -> Kdet cascade;
* the approximate joint layout/threshold search at the ``target_096`` target;
* the exhaustive joint layout/threshold search at the same target.

Five primary figures are written: validation and holdout cost bars, validation
and holdout terminal-routing bars, and a five-panel holdout cascade diagram.
Every figure is saved as both PNG and vector PDF, alongside the exact plotted
values in ``plot_data.json``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np

from empirical_outcomes import load_empirical_outcomes
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import FixedLayoutThresholdEvaluator, split_empirical_outcomes


DEFAULT_BASELINE_REPORT = Path("checkpoints/paper_kdet_baseline_target/h24.json")
DEFAULT_APPROXIMATE_REPORT = Path(
    "checkpoints/joint_ga_k1_free_h24_target_096/summary.json"
)
DEFAULT_BRUTE_FORCE_REPORT = Path(
    "checkpoints/brute_force_k1_free_h24_target_096/"
    "summary_shard_00000_of_00001.json"
)
DEFAULT_BRUTE_FORCE_RESULTS = Path(
    "checkpoints/brute_force_k1_free_h24_target_096/"
    "results_shard_00000_of_00001.jsonl"
)
LINEAR_K3_LAYOUT_ID = "59525b7d992bc2db"
DEFAULT_OUTCOMES = Path("checkpoints/empirical_outcomes.pkl")
DEFAULT_OUTPUT_DIR = Path("checkpoints/figures/h24_five_method_comparison")

PARTITIONS = ("validation", "holdout")
METHOD_COLORS = {
    "baseline": "#6C757D",
    "baseline_annealed": "#2A9D8F",
    "linear_k3": "#8E6CBE",
    "approximate_joint": "#E9A23B",
    "brute_force": "#457B9D",
}
ROUTE_ORDER = ("K0", "K1", "K2", "K3", "K4", "K5", "K6", "detector")
ROUTE_COLORS = {
    "K0": "#4C78A8",
    "K1": "#F58518",
    "K2": "#E45756",
    "K3": "#72B7B2",
    "K4": "#54A24B",
    "K5": "#EECA3B",
    "K6": "#B279A2",
    "detector": "#9D9DA1",
}
NODE_WIDTH = 1.25
NODE_HEIGHT = 0.82


@dataclass(frozen=True)
class MethodResult:
    key: str
    label: str
    short_label: str
    source: Path
    target_accuracy: float
    layout: dict[str, object]
    validation: dict[str, object]
    holdout: dict[str, object]


@dataclass(frozen=True)
class DiagramNode:
    location: str
    candidate_id: str
    x: float
    y: float


@dataclass(frozen=True)
class DiagramEdge:
    source: str
    target: str
    branch_label: str | None = None


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _read_jsonl_record(path: Path, layout_id: str) -> dict[str, object]:
    """Read one layout result without loading the full exhaustive-search file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            if str(payload.get("layout_id")) == layout_id:
                return payload
    raise ValueError(f"Layout {layout_id!r} was not found in {path}.")


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be an object.")
    return value


def _metrics(value: object, description: str) -> dict[str, object]:
    metrics = dict(_mapping(value, description))
    for field in ("accuracy", "expected_cost", "total", "route_counts", "thresholds"):
        if field not in metrics:
            raise ValueError(f"{description} has no {field!r} field.")
    counts = _mapping(metrics["route_counts"], f"{description}.route_counts")
    if sum(int(count) for count in counts.values()) != int(metrics["total"]):
        raise ValueError(f"{description} route counts do not sum to total.")
    _mapping(metrics["thresholds"], f"{description}.thresholds")
    return metrics


def _baseline_layout(report: Mapping[str, object]) -> dict[str, object]:
    split = _mapping(report.get("split"), "baseline split")
    initial = [str(item) for item in split.get("initial_layout", [])]
    specialized_raw = _mapping(
        split.get("specialized_layout", {}), "baseline specialized layout"
    )
    specialized = {
        str(key): [str(item) for item in value]
        for key, value in specialized_raw.items()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    }
    if not initial:
        raise ValueError("Baseline report does not contain its initial layout.")
    return {"initial": initial, "specialized": specialized}


def _winner_layout(winner: Mapping[str, object], description: str) -> dict[str, object]:
    layout = _mapping(winner.get("layout"), f"{description} layout")
    initial = [str(item) for item in layout.get("initial", [])]
    specialized_raw = _mapping(
        layout.get("specialized", {}), f"{description} specialized layout"
    )
    specialized = {
        str(key): [str(item) for item in value]
        for key, value in specialized_raw.items()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    }
    if not initial:
        raise ValueError(f"{description} has an empty initial layout.")
    return {"initial": initial, "specialized": specialized}


def load_methods(
    baseline_report: Path = DEFAULT_BASELINE_REPORT,
    approximate_report: Path = DEFAULT_APPROXIMATE_REPORT,
    brute_force_report: Path = DEFAULT_BRUTE_FORCE_REPORT,
    brute_force_results: Path = DEFAULT_BRUTE_FORCE_RESULTS,
) -> list[MethodResult]:
    baseline_payload = _read_json(baseline_report)
    approximate_payload = _read_json(approximate_report)
    brute_payload = _read_json(brute_force_report)
    linear_k3_payload = _read_jsonl_record(brute_force_results, LINEAR_K3_LAYOUT_ID)

    baseline_policy = _mapping(baseline_payload.get("baseline"), "baseline policy")
    annealed_policy = _mapping(
        baseline_payload.get("annealing"), "baseline annealing policy"
    )
    baseline_layout = _baseline_layout(baseline_payload)
    baseline_target = float(baseline_payload["target_accuracy"])

    approximate_winner = _mapping(
        approximate_payload.get("winner"), "approximate winner"
    )
    approximate_settings = _mapping(
        approximate_payload.get("settings"), "approximate settings"
    )
    brute_winner = _mapping(brute_payload.get("best"), "brute-force winner")
    brute_settings = _mapping(brute_payload.get("settings"), "brute-force settings")
    approximate_target = float(approximate_settings["target_accuracy"])
    brute_target = float(brute_settings["target_accuracy"])
    linear_k3_settings = _mapping(
        linear_k3_payload.get("settings"), "linear K3 settings"
    )
    linear_k3_target = float(linear_k3_settings["target_accuracy"])
    if not np.isclose(approximate_target, brute_target, atol=0.0, rtol=0.0):
        raise ValueError("Approximate and brute-force reports use different targets.")
    if not np.isclose(approximate_target, linear_k3_target, atol=0.0, rtol=0.0):
        raise ValueError("Linear K3 and joint-search reports use different targets.")

    methods = [
        MethodResult(
            key="baseline",
            label="Preset baseline",
            short_label="Preset\nbaseline",
            source=baseline_report,
            target_accuracy=baseline_target,
            layout=baseline_layout,
            validation=_metrics(baseline_policy.get("validation"), "baseline validation"),
            holdout=_metrics(baseline_policy.get("holdout"), "baseline holdout"),
        ),
        MethodResult(
            key="baseline_annealed",
            label="Baseline annealed",
            short_label="Baseline\nannealed",
            source=baseline_report,
            target_accuracy=baseline_target,
            layout=baseline_layout,
            validation=_metrics(
                annealed_policy.get("validation"), "annealed baseline validation"
            ),
            holdout=_metrics(
                annealed_policy.get("holdout"), "annealed baseline holdout"
            ),
        ),
        MethodResult(
            key="linear_k3",
            label="Annealed K3 -> Kdet (0.9662)",
            short_label="Annealed linear\nK3 -> Kdet",
            source=brute_force_results,
            target_accuracy=linear_k3_target,
            layout=_winner_layout(linear_k3_payload, "linear K3 result"),
            validation=_metrics(
                linear_k3_payload.get("validation"), "linear K3 validation"
            ),
            holdout=_metrics(linear_k3_payload.get("holdout"), "linear K3 holdout"),
        ),
        MethodResult(
            key="approximate_joint",
            label="Approx. joint (0.9662)",
            short_label="Approx. joint\n(0.9662)",
            source=approximate_report,
            target_accuracy=approximate_target,
            layout=_winner_layout(approximate_winner, "approximate winner"),
            validation=_metrics(
                approximate_winner.get("validation"), "approximate validation"
            ),
            holdout=_metrics(approximate_winner.get("holdout"), "approximate holdout"),
        ),
        MethodResult(
            key="brute_force",
            label="Brute force (0.9662)",
            short_label="Brute force\n(0.9662)",
            source=brute_force_report,
            target_accuracy=brute_target,
            layout=_winner_layout(brute_winner, "brute-force winner"),
            validation=_metrics(brute_winner.get("validation"), "brute validation"),
            holdout=_metrics(brute_winner.get("holdout"), "brute holdout"),
        ),
    ]
    if methods[0].layout != methods[1].layout:
        raise AssertionError("Preset and annealed baselines must share one topology.")
    return methods


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)


def _partition(method: MethodResult, partition: str) -> dict[str, object]:
    if partition == "validation":
        return method.validation
    if partition == "holdout":
        return method.holdout
    raise ValueError(f"Unknown partition: {partition}")


def plot_costs(
    methods: Sequence[MethodResult], partition: str, output_dir: Path
) -> None:
    metrics = [_partition(method, partition) for method in methods]
    costs = np.asarray([float(item["expected_cost"]) for item in metrics])
    accuracy = np.asarray([float(item["accuracy"]) for item in metrics])
    positions = np.arange(len(methods))

    figure, axis = plt.subplots(figsize=(9.8, 5.6), layout="constrained")
    bars = axis.bar(
        positions,
        costs,
        width=0.68,
        color=[METHOD_COLORS[method.key] for method in methods],
        edgecolor="#2F2F2F",
        linewidth=0.8,
    )
    axis.set_title(f"h24 — {partition.title()} expected cascade cost", fontsize=15, pad=14)
    axis.set_ylabel("Expected cost (ms)")
    axis.set_xticks(positions, [method.short_label for method in methods])
    axis.set_ylim(0.0, float(np.max(costs)) * 1.22)
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, cost, score in zip(bars, costs, accuracy, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + np.max(costs) * 0.025,
            f"{cost:,.1f} ms\n{score:.2%} accuracy",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    _save_figure(figure, output_dir, f"h24_{partition}_cost")


def _route_percent(metrics: Mapping[str, object], route_id: str) -> float:
    counts = _mapping(metrics["route_counts"], "route counts")
    return 100.0 * int(counts.get(route_id, 0)) / int(metrics["total"])


def plot_routing(
    methods: Sequence[MethodResult], partition: str, output_dir: Path
) -> None:
    metrics = [_partition(method, partition) for method in methods]
    used_routes = [
        route_id
        for route_id in ROUTE_ORDER
        if any(int(_mapping(item["route_counts"], "route counts").get(route_id, 0)) for item in metrics)
    ]
    positions = np.arange(len(methods))
    bottoms = np.zeros(len(methods), dtype=float)

    figure, axis = plt.subplots(figsize=(10.4, 6.0), layout="constrained")
    for route_id in used_routes:
        values = np.asarray([_route_percent(item, route_id) for item in metrics])
        bars = axis.bar(
            positions,
            values,
            width=0.68,
            bottom=bottoms,
            color=ROUTE_COLORS[route_id],
            edgecolor="white",
            linewidth=0.6,
            label="Kdet" if route_id == "detector" else route_id,
        )
        for bar, value, bottom in zip(bars, values, bottoms, strict=True):
            if value >= 3.5:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#1A1A1A",
                )
        bottoms += values

    axis.set_title(
        f"h24 — {partition.title()} terminal routing", fontsize=15, pad=14
    )
    axis.set_ylabel("Share of samples")
    axis.set_xticks(positions, [method.short_label for method in methods])
    axis.set_ylim(0.0, 100.0)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=100.0))
    axis.grid(axis="y", alpha=0.2, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(
        title="Final decision",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=min(7, len(used_routes)),
        frameon=False,
    )
    _save_figure(figure, output_dir, f"h24_{partition}_routing")


def _cascade(layout: Mapping[str, object]) -> Cascade:
    initial = tuple(str(item) for item in layout.get("initial", []))
    specialized_raw = _mapping(layout.get("specialized", {}), "specialized layout")
    specialized: dict[tuple[str, str], list[str]] = {}
    for key, chain in specialized_raw.items():
        router_id, group = str(key).split(":", maxsplit=1)
        if not isinstance(chain, Sequence) or isinstance(chain, (str, bytes)):
            raise ValueError(f"Specialized chain {key!r} must be a list.")
        specialized[(router_id, group)] = [str(item) for item in chain]
    return Cascade(
        expected_cost=0.0,
        initial=list(initial),
        specialized=specialized,
        detector="detector",
    )


def _holdout_evaluator(
    layout: Mapping[str, object], holdout_payload: Mapping[str, object]
) -> FixedLayoutThresholdEvaluator:
    optimizer = HierarchyOptimizer(
        dict(holdout_payload),
        detector_mode="paper",
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
    )
    return FixedLayoutThresholdEvaluator(optimizer, _cascade(layout))


def _detector_occurrence(location: str) -> str:
    return f"detector@{location}"


def occurrence_route_counts(
    evaluator: FixedLayoutThresholdEvaluator,
    thresholds: Mapping[str, object],
) -> dict[str, int]:
    """Count the exact cascade occurrence that makes each final decision."""

    threshold_map = evaluator._normalise_thresholds(  # noqa: SLF001
        {str(key): float(value) for key, value in thresholds.items()}
    )
    counts: dict[str, int] = {}

    def finish(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    def run_specialized(sample_id: int, router_id: str, group: str) -> None:
        chain = evaluator.cascade.specialized.get(
            (router_id, group), [evaluator.detector_id]
        )
        for index, candidate_id in enumerate(chain):
            location = f"specialized[{router_id}:{group}][{index}]"
            if candidate_id == evaluator.detector_id:
                finish(_detector_occurrence(location))
                return
            slot_id = evaluator._slot_by_location[location]  # noqa: SLF001
            if evaluator.confidence[slot_id][sample_id] >= threshold_map[slot_id]:
                finish(slot_id)
                return
        finish(_detector_occurrence(f"specialized[{router_id}:{group}][implicit]"))

    for sample_id in range(evaluator.sample_count):
        resolved = False
        for index, candidate_id in enumerate(evaluator.cascade.initial):
            location = f"initial[{index}]"
            if candidate_id == evaluator.detector_id:
                finish(_detector_occurrence(location))
                resolved = True
                break

            slot_id = evaluator._slot_by_location[location]  # noqa: SLF001
            if evaluator.confidence[slot_id][sample_id] < threshold_map[slot_id]:
                continue
            if evaluator._is_identifier(candidate_id):  # noqa: SLF001
                prediction = int(evaluator.prediction[candidate_id][sample_id])
                group = evaluator._intermediate_idx_to_group.get(prediction)  # noqa: SLF001
                if group in evaluator._specialized_groups:  # noqa: SLF001
                    run_specialized(sample_id, candidate_id, str(group))
                elif group in evaluator._global_name_to_idx:  # noqa: SLF001
                    finish(slot_id)
                else:
                    finish(_detector_occurrence(f"unmapped[{slot_id}]"))
            else:
                finish(slot_id)
            resolved = True
            break
        if not resolved:
            finish(_detector_occurrence("initial[implicit]"))

    if sum(counts.values()) != evaluator.sample_count:
        raise AssertionError("Occurrence route counts do not cover every sample.")
    return counts


def _aggregate_occurrence_counts(
    evaluator: FixedLayoutThresholdEvaluator, counts: Mapping[str, int]
) -> dict[str, int]:
    aggregate: dict[str, int] = {}
    for occurrence, count in counts.items():
        if occurrence.startswith("detector@"):
            candidate_id = "detector"
        else:
            candidate_id = evaluator.threshold_candidates[occurrence]
        aggregate[candidate_id] = aggregate.get(candidate_id, 0) + int(count)
    return aggregate


def _diagram_graph(
    layout: Mapping[str, object],
) -> tuple[list[DiagramNode], list[DiagramEdge]]:
    initial = [str(item) for item in layout.get("initial", [])]
    specialized = _mapping(layout.get("specialized", {}), "specialized layout")
    nodes: list[DiagramNode] = []
    edges: list[DiagramEdge] = []
    initial_locations: dict[str, list[tuple[int, str]]] = {}

    for index, candidate_id in enumerate(initial):
        location = f"initial[{index}]"
        nodes.append(DiagramNode(location, candidate_id, float(index), 0.0))
        initial_locations.setdefault(candidate_id, []).append((index, location))
        if index:
            edges.append(DiagramEdge(f"initial[{index - 1}]", location))

    router_ids = [
        candidate_id
        for candidate_id in initial
        if any(str(key).startswith(f"{candidate_id}:") for key in specialized)
    ]
    router_ids = list(dict.fromkeys(router_ids))
    for router_order, router_id in enumerate(router_ids):
        router_index, router_location = initial_locations[router_id][0]
        amplitude = 1.25 if len(router_ids) == 1 else max(0.85, 1.75 - router_order * 0.9)
        for group, sign in (("suv", 1.0), ("coupe", -1.0)):
            key = f"{router_id}:{group}"
            chain = specialized.get(key)
            if not isinstance(chain, Sequence) or isinstance(chain, (str, bytes)):
                continue
            previous = router_location
            for index, candidate_id in enumerate(chain):
                location = f"specialized[{key}][{index}]"
                nodes.append(
                    DiagramNode(
                        location,
                        str(candidate_id),
                        float(router_index + index + 1),
                        sign * amplitude,
                    )
                )
                edges.append(
                    DiagramEdge(
                        previous,
                        location,
                        group.upper() if index == 0 else None,
                    )
                )
                previous = location
    return nodes, edges


def _draw_edge(
    axis: plt.Axes,
    source: DiagramNode,
    target: DiagramNode,
    branch_label: str | None,
) -> None:
    start_x = source.x + NODE_WIDTH / 2
    end_x = target.x - NODE_WIDTH / 2
    if np.isclose(source.y, target.y):
        axis.annotate(
            "",
            xy=(end_x, target.y),
            xytext=(start_x, source.y),
            arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#333333"},
        )
        return

    elbow_x = start_x + max(0.15, (end_x - start_x) * 0.46)
    axis.plot(
        [start_x, elbow_x, elbow_x],
        [source.y, source.y, target.y],
        color="#333333",
        linewidth=1.0,
    )
    axis.annotate(
        "",
        xy=(end_x, target.y),
        xytext=(elbow_x, target.y),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#333333"},
    )
    if branch_label:
        axis.text(
            elbow_x - 0.04,
            (source.y + target.y) / 2,
            branch_label,
            fontsize=7.5,
            ha="right",
            va="center",
            color="#444444",
        )


def _draw_diagram_panel(
    axis: plt.Axes,
    method: MethodResult,
    evaluator: FixedLayoutThresholdEvaluator,
    occurrence_counts: Mapping[str, int],
    cmap: matplotlib.colors.Colormap,
    norm: Normalize,
) -> None:
    nodes, edges = _diagram_graph(method.layout)
    by_location = {node.location: node for node in nodes}
    threshold_map = evaluator._normalise_thresholds(  # noqa: SLF001
        {
            str(key): float(value)
            for key, value in _mapping(
                method.holdout["thresholds"], "holdout thresholds"
            ).items()
        }
    )

    for edge in edges:
        _draw_edge(
            axis,
            by_location[edge.source],
            by_location[edge.target],
            edge.branch_label,
        )

    for node in nodes:
        if node.candidate_id == evaluator.detector_id:
            occurrence_key = _detector_occurrence(node.location)
            threshold_text = "fallback"
            display_name = "Kdet"
        else:
            slot_id = evaluator._slot_by_location[node.location]  # noqa: SLF001
            occurrence_key = slot_id
            threshold_text = f"τ = {threshold_map[slot_id]:.3f}"
            display_name = node.candidate_id
        share = occurrence_counts.get(occurrence_key, 0) / evaluator.sample_count
        facecolor = cmap(norm(share))
        text_color = "white" if share >= 0.58 else "#111111"
        rectangle = Rectangle(
            (node.x - NODE_WIDTH / 2, node.y - NODE_HEIGHT / 2),
            NODE_WIDTH,
            NODE_HEIGHT,
            facecolor=facecolor,
            edgecolor="#2D2D2D",
            linewidth=1.0,
            zorder=3,
        )
        axis.add_patch(rectangle)
        axis.text(
            node.x,
            node.y,
            f"{display_name}\n{threshold_text}\n{share:.1%} exits",
            ha="center",
            va="center",
            fontsize=8.0,
            color=text_color,
            linespacing=1.05,
            zorder=4,
        )

    first = by_location["initial[0]"]
    input_x = first.x - NODE_WIDTH / 2 - 0.8
    axis.text(input_x - 0.05, first.y, "Input", ha="right", va="center", fontsize=8)
    axis.annotate(
        "",
        xy=(first.x - NODE_WIDTH / 2, first.y),
        xytext=(input_x, first.y),
        arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#333333"},
    )

    x_values = [node.x for node in nodes]
    y_values = [node.y for node in nodes]
    axis.set_xlim(min(x_values) - 1.35, max(x_values) + 0.85)
    axis.set_ylim(min(y_values) - 0.75, max(y_values) + 0.75)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    axis.set_title(
        f"{method.label}\n"
        f"{float(method.holdout['expected_cost']):,.1f} ms · "
        f"{float(method.holdout['accuracy']):.2%} accuracy",
        fontsize=11,
        pad=7,
    )


def plot_holdout_diagrams(
    methods: Sequence[MethodResult],
    holdout_payload: Mapping[str, object],
    output_dir: Path,
) -> dict[str, dict[str, int]]:
    rows = 2
    columns = 3
    figure, axes = plt.subplots(
        rows, columns, figsize=(20.0, 10.5), layout="constrained"
    )
    cmap = matplotlib.colormaps["Blues"]
    norm = Normalize(vmin=0.0, vmax=1.0)
    occurrence_by_method: dict[str, dict[str, int]] = {}

    active_axes = list(axes.flat[: len(methods)])
    for unused_axis in axes.flat[len(methods) :]:
        unused_axis.axis("off")

    for axis, method in zip(active_axes, methods, strict=True):
        evaluator = _holdout_evaluator(method.layout, holdout_payload)
        thresholds = _mapping(method.holdout["thresholds"], "holdout thresholds")
        replay = evaluator.evaluate(
            {str(key): float(value) for key, value in thresholds.items()}
        )
        if not np.isclose(
            float(replay["expected_cost"]),
            float(method.holdout["expected_cost"]),
            atol=1e-9,
        ):
            raise AssertionError(f"{method.label} holdout cost did not replay exactly.")
        if not np.isclose(
            float(replay["accuracy"]), float(method.holdout["accuracy"]), atol=1e-12
        ):
            raise AssertionError(f"{method.label} holdout accuracy did not replay exactly.")

        occurrence_counts = occurrence_route_counts(evaluator, thresholds)
        aggregate = _aggregate_occurrence_counts(evaluator, occurrence_counts)
        expected_routes = {
            str(key): int(value)
            for key, value in _mapping(
                method.holdout["route_counts"], "holdout route counts"
            ).items()
        }
        if aggregate != expected_routes:
            raise AssertionError(
                f"{method.label} occurrence routing does not match its saved aggregate: "
                f"{aggregate} != {expected_routes}"
            )
        occurrence_by_method[method.key] = occurrence_counts
        _draw_diagram_panel(axis, method, evaluator, occurrence_counts, cmap, norm)

    colorbar = figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=active_axes,
        orientation="horizontal",
        fraction=0.035,
        pad=0.035,
        aspect=45,
    )
    colorbar.set_label("Share of holdout samples whose final decision occurs at this box")
    colorbar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    figure.suptitle(
        "h24 — Holdout cascade layouts, confidence thresholds, and routing reliance",
        fontsize=16,
    )
    _save_figure(figure, output_dir, "h24_holdout_cascade_diagrams")
    return occurrence_by_method


def _plot_data(
    methods: Sequence[MethodResult],
    occurrence_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, object]:
    result: dict[str, object] = {
        "dataset": "h24",
        "detector_mode": "paper",
        "detector_cost_ms": PAPER_DETECTOR_COST_MS,
        "routing_bar_semantics": "terminal model producing the final decision",
        "diagram_shading_semantics": "terminal cascade occurrence producing the final decision",
        "methods": {},
    }
    method_output = result["methods"]
    assert isinstance(method_output, dict)
    for method in methods:
        method_output[method.key] = {
            "label": method.label,
            "source": str(method.source.resolve()),
            "target_accuracy": method.target_accuracy,
            "layout": method.layout,
            "validation": {
                "accuracy": method.validation["accuracy"],
                "expected_cost": method.validation["expected_cost"],
                "route_counts": method.validation["route_counts"],
                "thresholds": method.validation["thresholds"],
            },
            "holdout": {
                "accuracy": method.holdout["accuracy"],
                "expected_cost": method.holdout["expected_cost"],
                "route_counts": method.holdout["route_counts"],
                "thresholds": method.holdout["thresholds"],
                "occurrence_route_counts": occurrence_counts[method.key],
            },
        }
    return result


def plot_comparison(
    *,
    baseline_report: Path = DEFAULT_BASELINE_REPORT,
    approximate_report: Path = DEFAULT_APPROXIMATE_REPORT,
    brute_force_report: Path = DEFAULT_BRUTE_FORCE_REPORT,
    brute_force_results: Path = DEFAULT_BRUTE_FORCE_RESULTS,
    outcomes: Path = DEFAULT_OUTCOMES,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    methods = load_methods(
        baseline_report,
        approximate_report,
        brute_force_report,
        brute_force_results,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_empirical_outcomes(outcomes)
    _, holdout_payload, split = split_empirical_outcomes(
        payload,
        holdout_fraction=0.20,
        split_strategy="blocked_per_run",
        random_seed=0,
    )
    if int(split["holdout_samples"]) != int(methods[0].holdout["total"]):
        raise AssertionError("Outcome table does not reproduce the saved holdout split.")

    for partition in PARTITIONS:
        plot_costs(methods, partition, output_dir)
        plot_routing(methods, partition, output_dir)
    occurrence_counts = plot_holdout_diagrams(methods, holdout_payload, output_dir)

    plot_data = _plot_data(methods, occurrence_counts)
    data_path = output_dir / "plot_data.json"
    data_path.write_text(
        json.dumps(plot_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {data_path}")
    return plot_data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument(
        "--approximate-report", type=Path, default=DEFAULT_APPROXIMATE_REPORT
    )
    parser.add_argument(
        "--brute-force-report", type=Path, default=DEFAULT_BRUTE_FORCE_REPORT
    )
    parser.add_argument(
        "--brute-force-results", type=Path, default=DEFAULT_BRUTE_FORCE_RESULTS
    )
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    plot_comparison(
        baseline_report=args.baseline_report,
        approximate_report=args.approximate_report,
        brute_force_report=args.brute_force_report,
        brute_force_results=args.brute_force_results,
        outcomes=args.outcomes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
