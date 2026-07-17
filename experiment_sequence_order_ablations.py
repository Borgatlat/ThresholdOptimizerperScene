"""Sequence-order ablations for fixed-layout threshold optimization.

Research question
-----------------
For a fixed *set* of classifiers on the initial chain, how much does ORDER
matter once we retune thresholds?  Is the DP-chosen order uniquely good, or
do several permutations match it after annealing?

Why order can change cost (even with the same models)
-----------------------------------------------------
Cascade replay is short-circuiting: the first Ki that *accepts* ends the
path for that sample (or routes into a specialized branch).  Putting a
cheap-but-selective model early can skip expensive later stages; putting a
frequently-accepting model first can starve later stages of samples.  The
threshold anneal can partially compensate by raising/lowering accept rates,
but it cannot invent a different evaluation order — so topology order is a
real degree of freedom.

Method
------
1. Split outcomes ``blocked_per_run`` 80/20 (shared across all orders).
2. Synthesize the DP cascade on validation; that initial chain is ``dp_order``.
3. Build a capped set of alternative initial-chain orders (permutations +
   a few fixed reference chains).
4. For each order: anneal thresholds on validation, freeze, score holdout.
5. Rank vs DP order by holdout cost / accuracy / feasibility.

Usage
-----
    python experiment_sequence_order_ablations.py
    python experiment_sequence_order_ablations.py --scenes h24 --detector-modes paper
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from traceback import format_exc
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from empirical_outcomes import load_empirical_outcomes
from experiment_threshold_variants import make_cascade
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


ALL_SCENES = ("h24", "h08", "s31", "a06", "i29")
DEFAULT_OUTCOMES_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/sequence_order")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")
DETECTOR = "detector"
# Cap how many *extra* random permutations we keep when the DP chain is long.
# What if you raise this? More coverage, much longer wall-clock (each order
# needs a full anneal on validation).
MAX_RANDOM_PERMS = 8
# Exhaustive permutations only when the non-detector chain is this short.
# 4! = 24 is still practical; 5! = 120 would dominate the experiment budget.
MAX_EXHAUSTIVE_CHAIN_LEN = 4
# Fixed reference orders (used when their nodes are available).
REFERENCE_ORDERS: tuple[tuple[str, ...], ...] = (
    ("K3",),
    ("K2", "K3"),
    ("K0", "K3"),
    ("K0", "K2", "K3"),
    ("K3", "K0", "K2"),
)


def outcome_path_for_scene(outcomes_dir: Path, scene: str) -> Path:
    if scene == "h24":
        return outcomes_dir / "empirical_outcomes.pkl"
    return outcomes_dir / f"empirical_outcomes_{scene}.pkl"


def _strip_detector(chain: list[str]) -> list[str]:
    """Drop trailing (or any) detector tokens from an initial chain."""
    return [c for c in chain if c != DETECTOR]


def _order_key(order: list[str] | tuple[str, ...]) -> str:
    """Stable string id for an order, e.g. ``K0>K3>K2``."""
    nodes = _strip_detector(list(order))
    return ">".join(nodes) if nodes else "(detector_only)"


def cascade_to_dict(cascade: Cascade) -> dict[str, Any]:
    return {
        "expected_cost_dp": float(cascade.expected_cost),
        "initial": list(cascade.initial),
        "specialized": {
            f"{router_id}:{group}": list(chain)
            for (router_id, group), chain in cascade.specialized.items()
        },
        "detector": cascade.detector,
    }


def specialized_for_order(
    dp_specialized: dict[tuple[str, str], list[str]],
    order: list[str],
) -> dict[tuple[str, str], list[str]]:
    """Keep DP specialized branches only when their router is still on the trunk.

    Syntax: ``dp_specialized`` keys are ``(router_id, group)`` tuples.
    If a permutation drops ``K0``, every ``(K0, *)`` branch becomes unreachable
    and must be removed — otherwise ``FixedLayoutThresholdEvaluator`` would
    still try to tune thresholds for specialists that the layout never calls.

    What if you kept orphan specialized branches?  The evaluator would treat
    those Kis as tunable even though replay never reaches them, wasting search
    budget and muddying the "order only" ablation.
    """
    present = set(_strip_detector(order))
    return {
        key: list(chain)
        for key, chain in dp_specialized.items()
        if key[0] in present
    }


def build_order_candidates(
    dp_initial: list[str],
    *,
    random_seed: int,
    max_exhaustive_len: int = MAX_EXHAUSTIVE_CHAIN_LEN,
    max_random_perms: int = MAX_RANDOM_PERMS,
) -> list[tuple[str, list[str]]]:
    """Return ``[(label, order_without_detector), ...]`` with DP first.

    Labels document *why* each order is in the set (perm / reverse / ref / …)
    so the COMPARISON table stays readable.
    """
    dp_nodes = _strip_detector(dp_initial)
    labeled: list[tuple[str, list[str]]] = []
    seen: set[tuple[str, ...]] = set()

    def add(label: str, order: list[str] | tuple[str, ...]) -> None:
        nodes = _strip_detector(list(order))
        key = tuple(nodes)
        if key in seen:
            return
        # Empty chain (= detector only) is valid but uninteresting for an
        # order ablation; skip unless DP itself was empty (shouldn't happen).
        if not nodes and dp_nodes:
            return
        seen.add(key)
        labeled.append((label, nodes))

    add("dp_order", dp_nodes)

    n = len(dp_nodes)
    if n >= 2:
        # Structured variants that are easy to interpret in a paper figure.
        add("reverse", list(reversed(dp_nodes)))
        add("rotate_left", dp_nodes[1:] + dp_nodes[:1])
        add("rotate_right", dp_nodes[-1:] + dp_nodes[:-1])
        # Adjacent swaps: local perturbations of DP order.
        for i in range(n - 1):
            swapped = list(dp_nodes)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            add(f"swap_{i}_{i+1}", swapped)

    if 1 < n <= max_exhaustive_len:
        # Exhaustive permutations of the DP membership set.
        # ``itertools.permutations`` yields tuples; we already de-dupe via ``seen``.
        for perm in itertools.permutations(dp_nodes):
            add(f"perm_{_order_key(perm)}", list(perm))
    elif n > max_exhaustive_len:
        # Chain too long for n!: keep structured variants + a seeded random sample.
        rng = np.random.default_rng(random_seed)
        # Cap attempts so we don't spin forever on collisions with structured set.
        attempts = 0
        added_random = 0
        while added_random < max_random_perms and attempts < max_random_perms * 20:
            attempts += 1
            perm = list(rng.permutation(dp_nodes))
            before = len(seen)
            add(f"random_{added_random}", perm)
            if len(seen) > before:
                added_random += 1

    # Fixed reference orders: only nodes that are either in the DP chain or
    # in the small global/identifier pool {K0..K3}.  Never invent K4 on the
    # trunk here — specialists belong in specialized branches.
    allowed = set(dp_nodes) | {"K0", "K1", "K2", "K3"}
    for ref in REFERENCE_ORDERS:
        if all(node in allowed for node in ref):
            add(f"ref_{_order_key(ref)}", list(ref))

    return labeled


def _speedup(baseline_cost: float | None, opt_cost: float | None) -> float | None:
    if baseline_cost is None or opt_cost is None:
        return None
    if float(opt_cost) <= 0.0:
        return None
    return float(baseline_cost) / float(opt_cost)


def evaluate_order(
    validation_payload: dict,
    holdout_payload: dict,
    *,
    order: list[str],
    specialized: dict[tuple[str, str], list[str]],
    detector_mode: str,
    detector_cost_ms: float,
    annealing_iterations: int,
    quantile_points: int,
    random_seed: int,
    anneal: bool,
) -> dict:
    """Build cascade for ``order``, optionally anneal, score shared holdout."""
    cascade = make_cascade(list(order), specialized=specialized)

    val_opt = HierarchyOptimizer(
        validation_payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    hold_opt = HierarchyOptimizer(
        holdout_payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    val_eval = FixedLayoutThresholdEvaluator(val_opt, cascade)
    hold_eval = FixedLayoutThresholdEvaluator(hold_opt, cascade)

    # Collection-threshold baseline for THIS order (no anneal yet).
    baseline_val = val_eval.evaluate()
    baseline_hold = hold_eval.evaluate()
    target = float(baseline_val["accuracy"])

    if anneal:
        annealed = optimize_fixed_layout_thresholds_simulated_annealing(
            val_eval,
            target,
            quantile_points=quantile_points,
            n_iterations=annealing_iterations,
            random_seed=random_seed,
        )
        thresholds = {str(k): float(v) for k, v in annealed["thresholds"].items()}
        opt_val = {
            "accuracy": float(annealed["accuracy"]),
            "expected_cost": float(annealed["expected_cost"]),
            "feasible": bool(annealed["feasible"]),
            "thresholds": thresholds,
        }
        opt_hold = hold_eval.evaluate(thresholds)
    else:
        thresholds = dict(val_eval.default_thresholds)
        opt_val = {
            "accuracy": float(baseline_val["accuracy"]),
            "expected_cost": float(baseline_val["expected_cost"]),
            "feasible": True,
            "thresholds": thresholds,
        }
        opt_hold = baseline_hold

    return {
        "order": list(order),
        "order_key": _order_key(order),
        "layout": cascade_to_dict(cascade),
        "target_accuracy": target,
        "collection_validation": {
            "accuracy": float(baseline_val["accuracy"]),
            "expected_cost": float(baseline_val["expected_cost"]),
        },
        "collection_holdout": {
            "accuracy": float(baseline_hold["accuracy"]),
            "expected_cost": float(baseline_hold["expected_cost"]),
        },
        "optimized_validation": opt_val,
        "optimized_holdout": {
            "accuracy": float(opt_hold["accuracy"]),
            "expected_cost": float(opt_hold["expected_cost"]),
            "macro_accuracy": float(opt_hold.get("macro_accuracy", float("nan"))),
            "worst_class_accuracy": float(
                opt_hold.get("worst_class_accuracy", float("nan"))
            ),
            "route_counts": opt_hold.get("route_counts"),
            "thresholds": thresholds,
        },
        "holdout_feasible": bool(float(opt_hold["accuracy"]) >= target),
        "holdout_speedup_vs_collection": _speedup(
            float(baseline_hold["expected_cost"]),
            float(opt_hold["expected_cost"]),
        ),
        "annealed": bool(anneal),
    }


def run_scene_detector(
    scene: str,
    outcomes_path: Path,
    *,
    detector_mode: str,
    detector_cost_ms: float,
    annealing_iterations: int,
    quantile_points: int,
    holdout_fraction: float,
    split_strategy: str,
    random_seed: int,
    include_collection_baseline: bool,
    max_exhaustive_chain_len: int = MAX_EXHAUSTIVE_CHAIN_LEN,
    max_random_perms: int = MAX_RANDOM_PERMS,
) -> dict:
    """One scene × detector_mode: shared split, DP order, then every ablation."""
    payload = load_empirical_outcomes(outcomes_path)
    validation_payload, holdout_payload, split_meta = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=random_seed,
    )

    # DP reference structure — validation only (holdout must not pick topology).
    dp_optimizer = HierarchyOptimizer(
        validation_payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    dp_cascade = dp_optimizer.synthesize()
    dp_nodes = _strip_detector(dp_cascade.initial)
    dp_specialized = {
        key: list(chain) for key, chain in dp_cascade.specialized.items()
    }

    candidates = build_order_candidates(
        dp_cascade.initial,
        random_seed=random_seed,
        max_exhaustive_len=max_exhaustive_chain_len,
        max_random_perms=max_random_perms,
    )
    print(
        f"  DP initial={dp_nodes}  ({len(candidates)} orders to evaluate)",
        flush=True,
    )

    orders_out: dict[str, Any] = {}
    for idx, (label, order) in enumerate(candidates):
        spec = specialized_for_order(dp_specialized, order)
        # Distinct anneal seeds per order so two perms are not identical walks.
        seed = random_seed + 17 * (idx + 1)
        print(f"  [{label}] order={order} ...", flush=True)
        result = evaluate_order(
            validation_payload,
            holdout_payload,
            order=order,
            specialized=spec,
            detector_mode=detector_mode,
            detector_cost_ms=detector_cost_ms,
            annealing_iterations=annealing_iterations,
            quantile_points=quantile_points,
            random_seed=seed,
            anneal=True,
        )
        result["label"] = label
        result["is_dp_order"] = label == "dp_order"
        orders_out[label] = result
        opt = result["optimized_holdout"]
        print(
            f"    holdout acc={opt['accuracy']:.4f}  "
            f"cost={opt['expected_cost']:.2f}ms  "
            f"feasible={result['holdout_feasible']}",
            flush=True,
        )

        if include_collection_baseline and label == "dp_order":
            # Cheap "before retune" snapshot for the DP order only (full
            # collection baselines for every perm would nearly double runtime
            # without much extra science — collection metrics are already
            # stored inside each annealed result).
            pass

    # Rank annealed orders: feasible first, then lower holdout cost.
    ranking = sorted(
        orders_out.values(),
        key=lambda r: (
            0 if r["holdout_feasible"] else 1,
            float(r["optimized_holdout"]["expected_cost"]),
            -float(r["optimized_holdout"]["accuracy"]),
        ),
    )
    dp_result = orders_out["dp_order"]
    best = ranking[0]
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
        row["delta_cost_vs_dp_ms"] = float(
            row["optimized_holdout"]["expected_cost"]
        ) - float(dp_result["optimized_holdout"]["expected_cost"])
        row["delta_acc_vs_dp"] = float(row["optimized_holdout"]["accuracy"]) - float(
            dp_result["optimized_holdout"]["accuracy"]
        )

    return {
        "scene": scene,
        "detector_mode": detector_mode,
        "outcomes_path": str(outcomes_path),
        "split": split_meta,
        "dp_layout": cascade_to_dict(dp_cascade),
        "n_orders": len(orders_out),
        "orders": orders_out,
        "best_label": best["label"],
        "best_beats_dp_cost": bool(
            float(best["optimized_holdout"]["expected_cost"])
            < float(dp_result["optimized_holdout"]["expected_cost"]) - 1e-9
        ),
        "ranking": [r["label"] for r in ranking],
    }


def write_comparison_md(summary: dict, output_dir: Path) -> Path:
    rows: list[dict] = []
    for key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        dp = report["orders"]["dp_order"]
        best_label = report["best_label"]
        best = report["orders"][best_label]
        # Count how many orders are within 1% cost of DP among feasible.
        dp_cost = float(dp["optimized_holdout"]["expected_cost"])
        near_dp = 0
        better_cost = 0
        for label, block in report["orders"].items():
            if label == "dp_order":
                continue
            if not block["holdout_feasible"]:
                continue
            cost = float(block["optimized_holdout"]["expected_cost"])
            if cost <= dp_cost * 1.01:
                near_dp += 1
            if cost < dp_cost - 1e-9:
                better_cost += 1
        rows.append(
            {
                "scene": report["scene"],
                "detector_mode": report["detector_mode"],
                "n_orders": report["n_orders"],
                "dp_order": dp["order"],
                "dp_holdout_acc": dp["optimized_holdout"]["accuracy"],
                "dp_holdout_cost_ms": dp["optimized_holdout"]["expected_cost"],
                "dp_feasible": dp["holdout_feasible"],
                "best_label": best_label,
                "best_order": best["order"],
                "best_holdout_acc": best["optimized_holdout"]["accuracy"],
                "best_holdout_cost_ms": best["optimized_holdout"]["expected_cost"],
                "best_feasible": best["holdout_feasible"],
                "delta_cost_best_vs_dp_ms": best["delta_cost_vs_dp_ms"],
                "delta_acc_best_vs_dp": best["delta_acc_vs_dp"],
                "n_feasible_near_dp_1pct": near_dp,
                "n_feasible_cheaper_than_dp": better_cost,
            }
        )

    md = [
        "# Sequence-Order Ablations — Comparison",
        "",
        "Question: for a fixed set of classifiers on the initial chain, how much "
        "does **order** matter after threshold retuning?",
        "",
        "All orders for a given (scene, detector_mode) share the same "
        "`blocked_per_run` holdout. DP order is always included.",
        "",
        "Ranking: feasible policies first, then lower holdout cost. "
        "Reference orders (`ref_*`) may change membership (not only permutation).",
        "",
        "| scene | detector | #orders | DP order | DP cost | DP feas | best label | "
        "best order | best cost | best feas | Δcost vs DP | #feas cheaper | #near DP |",
        "|---|---|---:|---|---:|---|---|---|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            "| {scene} | {det} | {n} | `{dp}` | {dpc} | {dpf} | {bl} | `{bo}` | {bc} | "
            "{bf} | {dc} | {cheaper} | {near} |".format(
                scene=row["scene"],
                det=row["detector_mode"],
                n=row["n_orders"],
                dp="→".join(row["dp_order"]),
                dpc=_fmt(row["dp_holdout_cost_ms"]),
                dpf=row["dp_feasible"],
                bl=row["best_label"],
                bo="→".join(row["best_order"]),
                bc=_fmt(row["best_holdout_cost_ms"]),
                bf=row["best_feasible"],
                dc=_fmt(row["delta_cost_best_vs_dp_ms"]),
                cheaper=row["n_feasible_cheaper_than_dp"],
                near=row["n_feasible_near_dp_1pct"],
            )
        )

    n_dp_best = sum(1 for r in rows if r["best_label"] == "dp_order")
    n_alt_best = len(rows) - n_dp_best
    n_feas_win = sum(
        1
        for r in rows
        if r["best_label"] != "dp_order"
        and r["best_feasible"]
        and float(r["delta_cost_best_vs_dp_ms"]) < -1e-9
    )
    md.extend(
        [
            "",
            "## Verdict",
            "",
            f"Across {len(rows)} (scene, detector) settings:",
            f"- DP order ranked best in **{n_dp_best}** settings",
            f"- Some other order ranked above DP in **{n_alt_best}** settings",
            f"- Of those, **{n_feas_win}** were feasible *and* cheaper than DP",
            "",
            "Δcost < 0 means the ranked-best policy was cheaper than DP. "
            "When both DP and best are infeasible, treat cost wins cautiously "
            "(they may buy speed by missing the accuracy target). "
            "Many near-DP feasible orders ⇒ order is weakly identified after annealing.",
            "",
        ]
    )

    path = output_dir / "COMPARISON.md"
    path.write_text("\n".join(md) + "\n")
    (output_dir / "COMPARISON.json").write_text(
        json.dumps({"table": rows}, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(f"Wrote {path}")
    return path


def _fmt(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def plot_sequence_order_figures(summary: dict, figures_dir: Path) -> list[Path]:
    """Paper figures: cost-by-order for h24, and best-vs-DP deltas across scenes."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )
    c_dp = "#4C6A80"
    c_other = "#C45C26"
    c_best = "#2F5D50"
    written: list[Path] = []

    # --- Fig A/B: h24 cost-by-order bars (paper / trained) ---
    for detector_mode in ("paper", "trained"):
        key = f"h24__{detector_mode}"
        report = summary.get("runs", {}).get(key)
        if not report or report.get("status") != "ok":
            continue
        # Sort by holdout cost for a readable waterfall-like bar chart.
        items = sorted(
            report["orders"].values(),
            key=lambda r: float(r["optimized_holdout"]["expected_cost"]),
        )
        labels = [r["label"] for r in items]
        costs = [float(r["optimized_holdout"]["expected_cost"]) for r in items]
        colors = []
        for r in items:
            if r["label"] == "dp_order":
                colors.append(c_dp)
            elif r["label"] == report["best_label"]:
                colors.append(c_best)
            else:
                colors.append(c_other)

        fig_w = max(7.0, 0.35 * len(labels) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, 4.0))
        x = np.arange(len(labels))
        ax.bar(x, costs, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
        ax.set_ylabel("Holdout expected cost (ms)")
        ax.set_title(f"h24 sequence-order ablation ({detector_mode} Kdet)")
        # Tiny legend via proxy artists.
        ax.bar([], [], color=c_dp, label="DP order")
        ax.bar([], [], color=c_best, label="Best order")
        ax.bar([], [], color=c_other, label="Other")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        path = figures_dir / f"fig_sequence_order_h24_cost_{detector_mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

    # --- Fig C/D: Δcost(best vs DP) across scenes ---
    for detector_mode in ("paper", "trained"):
        scenes: list[str] = []
        deltas: list[float] = []
        for scene in ALL_SCENES:
            key = f"{scene}__{detector_mode}"
            report = summary.get("runs", {}).get(key)
            if not report or report.get("status") != "ok":
                continue
            scenes.append(scene)
            best = report["orders"][report["best_label"]]
            deltas.append(float(best["delta_cost_vs_dp_ms"]))

        if not scenes:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        x = np.arange(len(scenes))
        colors = [c_best if d < -1e-9 else c_dp for d in deltas]
        ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.4)
        ax.axhline(0.0, color="#6B7280", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.set_ylabel("Δ holdout cost: best − DP (ms)")
        ax.set_xlabel("Scene")
        ax.set_title(f"Best order vs DP order — {detector_mode} Kdet")
        fig.tight_layout()
        path = figures_dir / f"fig_sequence_order_delta_vs_dp_{detector_mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

    # --- Fig E/F: accuracy vs cost scatter of all orders (h24) ---
    for detector_mode in ("paper", "trained"):
        key = f"h24__{detector_mode}"
        report = summary.get("runs", {}).get(key)
        if not report or report.get("status") != "ok":
            continue
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        for r in report["orders"].values():
            acc = float(r["optimized_holdout"]["accuracy"])
            cost = float(r["optimized_holdout"]["expected_cost"])
            if r["label"] == "dp_order":
                ax.scatter(
                    cost, acc, s=70, c=c_dp, marker="D", zorder=3, label="DP order"
                )
            elif r["label"] == report["best_label"]:
                ax.scatter(
                    cost, acc, s=70, c=c_best, marker="*", zorder=3, label="Best"
                )
            else:
                ax.scatter(cost, acc, s=28, c=c_other, alpha=0.75, zorder=2)
        ax.set_xlabel("Holdout expected cost (ms)")
        ax.set_ylabel("Holdout accuracy")
        ax.set_title(f"h24 order cloud ({detector_mode} Kdet)")
        # De-dupe legend entries.
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), frameon=False)
        fig.tight_layout()
        path = figures_dir / f"fig_sequence_order_h24_scatter_{detector_mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

    for path in written:
        print(f"Wrote {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=list(ALL_SCENES))
    parser.add_argument(
        "--detector-modes",
        nargs="+",
        choices=("paper", "trained"),
        default=("paper", "trained"),
    )
    parser.add_argument("--outcomes-dir", type=Path, default=DEFAULT_OUTCOMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--iterations", type=int, default=8_000)
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument(
        "--split-strategy",
        choices=("blocked_per_run", "random_per_run"),
        default="blocked_per_run",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--detector-cost-ms", type=float, default=PAPER_DETECTOR_COST_MS)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write JSON/MD only (smoke tests).",
    )
    parser.add_argument(
        "--max-exhaustive-chain-len",
        type=int,
        default=MAX_EXHAUSTIVE_CHAIN_LEN,
        help="If DP chain length exceeds this, sample random perms instead of n!.",
    )
    parser.add_argument(
        "--max-random-perms",
        type=int,
        default=MAX_RANDOM_PERMS,
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "experiment": "sequence_order_ablations",
        "question": (
            "For a fixed set of classifiers on the initial chain, how much does "
            "ORDER matter once we retune thresholds?"
        ),
        "annealing_iterations": args.iterations,
        "quantile_points": args.quantile_points,
        "split_strategy": args.split_strategy,
        "holdout_fraction": args.holdout_fraction,
        "random_seed": args.seed,
        "max_exhaustive_chain_len": args.max_exhaustive_chain_len,
        "max_random_perms": args.max_random_perms,
        "runs": {},
    }

    ordered_scenes = [s for s in ALL_SCENES if s in args.scenes]
    ordered_scenes += [s for s in args.scenes if s not in ordered_scenes]

    for detector_mode in args.detector_modes:
        for scene in ordered_scenes:
            key = f"{scene}__{detector_mode}"
            outcomes = outcome_path_for_scene(args.outcomes_dir, scene)
            print(f"\n=== {key} ===", flush=True)
            if not outcomes.is_file():
                summary["runs"][key] = {
                    "status": "skipped",
                    "reason": f"missing {outcomes}",
                }
                print(f"  skipped: {outcomes}")
                continue
            try:
                report = run_scene_detector(
                    scene,
                    outcomes,
                    detector_mode=detector_mode,
                    detector_cost_ms=args.detector_cost_ms,
                    annealing_iterations=args.iterations,
                    quantile_points=args.quantile_points,
                    holdout_fraction=args.holdout_fraction,
                    split_strategy=args.split_strategy,
                    random_seed=args.seed,
                    include_collection_baseline=True,
                    max_exhaustive_chain_len=args.max_exhaustive_chain_len,
                    max_random_perms=args.max_random_perms,
                )
                report["status"] = "ok"
                run_path = args.output_dir / f"{key}.json"
                run_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
                )
                report["report_path"] = str(run_path)
                summary["runs"][key] = report
                print(
                    f"  best={report['best_label']}  "
                    f"beats_dp={report['best_beats_dp_cost']}  "
                    f"Wrote {run_path}",
                    flush=True,
                )
            except Exception as error:
                summary["runs"][key] = {
                    "status": "failed",
                    "scene": scene,
                    "detector_mode": detector_mode,
                    "error": str(error),
                    "traceback": format_exc(),
                }
                print(f"  FAILED: {error}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(f"\nWrote {summary_path}")
    write_comparison_md(summary, args.output_dir)
    if not args.skip_plots:
        plot_sequence_order_figures(summary, args.figures_dir)


if __name__ == "__main__":
    main()
