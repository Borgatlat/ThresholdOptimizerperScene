"""Detector-cost sensitivity: re-synthesize + retune as paper-Kdet cost varies.

PROMPT (experiment — README #7)
-------------------------------
GOAL
If we change the assumed paper-Kdet cost, re-synthesize the DP cascade, and
retune thresholds, how do layout / holdout accuracy / expected cost change?

CONTEXT
- HierarchyOptimizer(detector_mode="paper", detector_cost_ms=...) already
  controls synthetic Kdet cost; trained mode uses the measured detector cost.
- compare_kdet_costs() prints structure only — this experiment ALSO anneals
  thresholds after each structure change.
- DO NOT train classifiers; DO NOT scene-switch; DO NOT redo prior suites.

METHOD
For scene=h24 first (then h08,s31,a06,i29 — not i22):
  1) Split blocked_per_run 80/20 (shared across all Kdet costs).
  2) For each detector_cost_ms in the sweep (paper mode):
       a) synthesize DP cascade on validation with that cost
       b) anneal thresholds (target = collection-threshold micro accuracy)
       c) freeze (cascade, thresholds); evaluate on the SHARED holdout
          using the SAME detector_cost_ms for cost accounting
  3) Also report one trained-Kdet reference point (natural measured cost).

DELIVERABLES
- this script
- checkpoints/threshold_experiments/detector_cost/
- COMPARISON.md + paper PNGs
- commit + push + update PR

Why this matters (beginner intuition)
-------------------------------------
Paper Kdet is a *synthetic* expensive always-correct fallback.  If you set its
cost very high, the DP is willing to chain through more cheap Kis first.  If
you set it low, the optimal cascade collapses toward "just call Kdet".
Threshold annealing then reshapes accept rates on whatever layout the DP
chose — so cost and structure move together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from traceback import format_exc
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from empirical_outcomes import load_empirical_outcomes
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


ALL_SCENES = ("h24", "h08", "s31", "a06", "i29")
DEFAULT_OUTCOMES_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/detector_cost")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")

# Sweep spans "barely more than a Ki" → paper default 10_000 ms.
# 10.85 / ~28 appear in compare_kdet_costs as near-trained references.
DEFAULT_COST_SWEEP_MS: tuple[float, ...] = (
    100.0,
    250.0,
    500.0,
    1_000.0,
    2_500.0,
    5_000.0,
    10_000.0,
)


def outcome_path_for_scene(outcomes_dir: Path, scene: str) -> Path:
    if scene == "h24":
        return outcomes_dir / "empirical_outcomes.pkl"
    return outcomes_dir / f"empirical_outcomes_{scene}.pkl"


def cascade_to_dict(cascade: Cascade) -> dict[str, Any]:
    return {
        "expected_cost_dp": float(cascade.expected_cost),
        "initial": list(cascade.initial),
        "specialized": {
            f"{router_id}:{group}": list(chain)
            for (router_id, group), chain in cascade.specialized.items()
        },
        "detector": cascade.detector,
        "initial_len_excl_detector": len(
            [c for c in cascade.initial if c != cascade.detector]
        ),
    }


def _speedup(baseline_cost: float, opt_cost: float) -> float | None:
    if opt_cost <= 0:
        return None
    return float(baseline_cost) / float(opt_cost)


def run_one_cost(
    validation_payload: dict,
    holdout_payload: dict,
    *,
    detector_mode: str,
    detector_cost_ms: float,
    annealing_iterations: int,
    quantile_points: int,
    random_seed: int,
) -> dict:
    """Synthesize → anneal → holdout for one (mode, Kdet cost) pair.

    Holdout evaluator must use the same ``detector_cost_ms`` as synthesis,
    or expected-cost numbers would mix two different fallback prices.
    """
    val_opt = HierarchyOptimizer(
        validation_payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    cascade = val_opt.synthesize()
    # Rebuild evaluator-bound optimizer so FixedLayoutThresholdEvaluator sees
    # the same detector_cost (HierarchyOptimizer stores it on the instance).
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

    collection_val = val_eval.evaluate()
    collection_hold = hold_eval.evaluate()
    target = float(collection_val["accuracy"])

    annealed = optimize_fixed_layout_thresholds_simulated_annealing(
        val_eval,
        target,
        quantile_points=quantile_points,
        n_iterations=annealing_iterations,
        random_seed=random_seed,
        constraint_metric="micro",
    )
    thresholds = {str(k): float(v) for k, v in annealed["thresholds"].items()}
    opt_val = val_eval.evaluate(thresholds)
    opt_hold = hold_eval.evaluate(thresholds)

    return {
        "detector_mode": detector_mode,
        "detector_cost_ms": float(val_opt.detector_cost),
        "requested_detector_cost_ms": float(detector_cost_ms),
        "layout": cascade_to_dict(cascade),
        "target_accuracy": target,
        "collection_validation": {
            "accuracy": float(collection_val["accuracy"]),
            "macro_accuracy": float(collection_val["macro_accuracy"]),
            "worst_class_accuracy": float(collection_val["worst_class_accuracy"]),
            "expected_cost": float(collection_val["expected_cost"]),
        },
        "collection_holdout": {
            "accuracy": float(collection_hold["accuracy"]),
            "macro_accuracy": float(collection_hold["macro_accuracy"]),
            "worst_class_accuracy": float(collection_hold["worst_class_accuracy"]),
            "expected_cost": float(collection_hold["expected_cost"]),
        },
        "optimized_validation": {
            "accuracy": float(opt_val["accuracy"]),
            "macro_accuracy": float(opt_val["macro_accuracy"]),
            "worst_class_accuracy": float(opt_val["worst_class_accuracy"]),
            "expected_cost": float(opt_val["expected_cost"]),
            "feasible": bool(opt_val["accuracy"] >= target),
        },
        "optimized_holdout": {
            "accuracy": float(opt_hold["accuracy"]),
            "macro_accuracy": float(opt_hold["macro_accuracy"]),
            "worst_class_accuracy": float(opt_hold["worst_class_accuracy"]),
            "expected_cost": float(opt_hold["expected_cost"]),
            "route_counts": opt_hold.get("route_counts"),
        },
        "thresholds": thresholds,
        "holdout_feasible": bool(float(opt_hold["accuracy"]) >= target),
        "holdout_speedup_vs_collection": _speedup(
            float(collection_hold["expected_cost"]),
            float(opt_hold["expected_cost"]),
        ),
    }


def run_scene(
    scene: str,
    outcomes_path: Path,
    *,
    cost_sweep_ms: tuple[float, ...],
    include_trained_reference: bool,
    annealing_iterations: int,
    quantile_points: int,
    holdout_fraction: float,
    split_strategy: str,
    random_seed: int,
) -> dict:
    payload = load_empirical_outcomes(outcomes_path)
    validation_payload, holdout_payload, split_meta = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=random_seed,
    )
    measured_det_cost = float(payload["detector"]["cost"])

    report: dict[str, Any] = {
        "scene": scene,
        "outcomes_path": str(outcomes_path),
        "split": split_meta,
        "measured_detector_cost_ms": measured_det_cost,
        "paper_costs": {},
        "trained_reference": None,
    }

    for i, cost in enumerate(cost_sweep_ms):
        label = f"paper_{_cost_tag(cost)}"
        print(f"  [{label}] detector_cost_ms={cost:g} ...", flush=True)
        block = run_one_cost(
            validation_payload,
            holdout_payload,
            detector_mode="paper",
            detector_cost_ms=float(cost),
            annealing_iterations=annealing_iterations,
            quantile_points=quantile_points,
            random_seed=random_seed + 41 * (i + 1),
        )
        report["paper_costs"][label] = block
        lay = block["layout"]["initial"]
        print(
            f"    layout={lay}  "
            f"holdout acc={block['optimized_holdout']['accuracy']:.4f}  "
            f"cost={block['optimized_holdout']['expected_cost']:.2f}ms  "
            f"speedup={block['holdout_speedup_vs_collection'] or float('nan'):.3f}x  "
            f"feas={block['holdout_feasible']}",
            flush=True,
        )

    if include_trained_reference:
        print(
            f"  [trained_ref] measured detector_cost_ms={measured_det_cost:.4f} ...",
            flush=True,
        )
        # detector_cost_ms arg is ignored for trained mode inside HierarchyOptimizer;
        # pass measured value for bookkeeping clarity.
        trained = run_one_cost(
            validation_payload,
            holdout_payload,
            detector_mode="trained",
            detector_cost_ms=measured_det_cost,
            annealing_iterations=annealing_iterations,
            quantile_points=quantile_points,
            random_seed=random_seed + 7,
        )
        report["trained_reference"] = trained
        print(
            f"    layout={trained['layout']['initial']}  "
            f"holdout acc={trained['optimized_holdout']['accuracy']:.4f}  "
            f"cost={trained['optimized_holdout']['expected_cost']:.2f}ms",
            flush=True,
        )

    return report


def _cost_tag(cost: float) -> str:
    """Filename-safe tag: 1000 -> 1000, 10000 -> 10000."""
    if float(cost).is_integer():
        return str(int(cost))
    return str(cost).replace(".", "p")


def write_comparison_md(summary: dict, output_dir: Path) -> Path:
    rows: list[dict] = []
    for _key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        scene = report["scene"]
        for label, block in report.get("paper_costs", {}).items():
            rows.append(
                {
                    "scene": scene,
                    "mode": "paper",
                    "detector_cost_ms": block["detector_cost_ms"],
                    "label": label,
                    "layout": block["layout"]["initial"],
                    "chain_len": block["layout"]["initial_len_excl_detector"],
                    "holdout_acc": block["optimized_holdout"]["accuracy"],
                    "holdout_cost_ms": block["optimized_holdout"]["expected_cost"],
                    "speedup": block.get("holdout_speedup_vs_collection"),
                    "feasible": block.get("holdout_feasible"),
                    "collection_holdout_cost_ms": block["collection_holdout"][
                        "expected_cost"
                    ],
                }
            )
        trained = report.get("trained_reference")
        if trained:
            rows.append(
                {
                    "scene": scene,
                    "mode": "trained",
                    "detector_cost_ms": trained["detector_cost_ms"],
                    "label": "trained_ref",
                    "layout": trained["layout"]["initial"],
                    "chain_len": trained["layout"]["initial_len_excl_detector"],
                    "holdout_acc": trained["optimized_holdout"]["accuracy"],
                    "holdout_cost_ms": trained["optimized_holdout"]["expected_cost"],
                    "speedup": trained.get("holdout_speedup_vs_collection"),
                    "feasible": trained.get("holdout_feasible"),
                    "collection_holdout_cost_ms": trained["collection_holdout"][
                        "expected_cost"
                    ],
                }
            )

    md = [
        "# Detector-Cost Sensitivity — Comparison",
        "",
        "Question: as assumed paper-Kdet cost changes, how do **DP layout**, "
        "holdout accuracy, and expected cost change after threshold retuning?",
        "",
        "All paper costs for a scene share the same `blocked_per_run` holdout. "
        "`trained_ref` uses the measured Kdet cost (structure + replay).",
        "",
        "| scene | mode | Kdet cost (ms) | chain len | layout | holdout acc | "
        "opt cost (ms) | speedup | feasible |",
        "|---|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        md.append(
            "| {scene} | {mode} | {cost} | {n} | `{lay}` | {acc} | {oc} | {spd} | {feas} |".format(
                scene=row["scene"],
                mode=row["mode"],
                cost=_fmt(row["detector_cost_ms"]),
                n=row["chain_len"],
                lay="→".join(row["layout"]),
                acc=_fmt(row["holdout_acc"]),
                oc=_fmt(row["holdout_cost_ms"]),
                spd=_fmt(row["speedup"]),
                feas=row["feasible"],
            )
        )

    # Verdict: does chain length grow with Kdet cost on h24?
    h24 = summary.get("runs", {}).get("h24")
    verdict_lines = ["", "## Verdict", ""]
    if h24 and h24.get("status") == "ok":
        pairs = sorted(
            (
                float(b["detector_cost_ms"]),
                int(b["layout"]["initial_len_excl_detector"]),
                float(b["optimized_holdout"]["expected_cost"]),
            )
            for b in h24["paper_costs"].values()
        )
        lens = [p[1] for p in pairs]
        nondec = all(lens[i] <= lens[i + 1] for i in range(len(lens) - 1))
        verdict_lines.append(
            f"On **h24/paper**, chain length (excl. detector) across the sweep: "
            f"{lens} for Kdet costs {[p[0] for p in pairs]}."
        )
        verdict_lines.append(
            "- Chain length is **non-decreasing** in Kdet cost."
            if nondec
            else "- Chain length is **not** monotone in Kdet cost (see table)."
        )
        verdict_lines.append(
            "Higher assumed Kdet cost usually deepens the cascade and raises "
            "end-to-end expected cost even after threshold retune — unless "
            "annealing finds aggressive early accepts."
        )
    else:
        verdict_lines.append("h24 run missing; see per-scene rows above.")
    verdict_lines.append("")
    md.extend(verdict_lines)

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


def plot_figures(summary: dict, figures_dir: Path) -> list[Path]:
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
    c_cost = "#C45C26"
    c_acc = "#2F5D50"
    c_len = "#4C6A80"
    written: list[Path] = []

    # --- h24 primary curves ---
    h24 = summary.get("runs", {}).get("h24")
    if h24 and h24.get("status") == "ok":
        blocks = sorted(
            h24["paper_costs"].values(),
            key=lambda b: float(b["detector_cost_ms"]),
        )
        xs = [float(b["detector_cost_ms"]) for b in blocks]
        costs = [float(b["optimized_holdout"]["expected_cost"]) for b in blocks]
        accs = [float(b["optimized_holdout"]["accuracy"]) for b in blocks]
        lens = [int(b["layout"]["initial_len_excl_detector"]) for b in blocks]

        fig, ax1 = plt.subplots(figsize=(6.8, 3.8))
        ax1.plot(xs, costs, "o-", color=c_cost, label="Holdout opt. cost")
        ax1.set_xscale("log")
        ax1.set_xlabel("Assumed paper-Kdet cost (ms)")
        ax1.set_ylabel("Holdout expected cost (ms)", color=c_cost)
        ax1.tick_params(axis="y", labelcolor=c_cost)
        ax2 = ax1.twinx()
        ax2.plot(xs, accs, "s--", color=c_acc, label="Holdout accuracy")
        ax2.set_ylabel("Holdout accuracy", color=c_acc)
        ax2.tick_params(axis="y", labelcolor=c_acc)
        ax2.set_ylim(0.0, 1.05)
        ax1.set_title("h24: threshold-retuned cascade vs Kdet cost")
        fig.tight_layout()
        path = figures_dir / "fig_detector_cost_h24_cost_acc.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.step(xs, lens, where="mid", color=c_len, linewidth=2)
        ax.plot(xs, lens, "o", color=c_len)
        ax.set_xscale("log")
        ax.set_xlabel("Assumed paper-Kdet cost (ms)")
        ax.set_ylabel("DP initial chain length (excl. det)")
        ax.set_title("h24: cascade depth vs Kdet cost")
        ax.set_yticks(sorted(set(lens)))
        fig.tight_layout()
        path = figures_dir / "fig_detector_cost_h24_chain_len.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

    # --- all scenes: opt cost vs Kdet cost ---
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    scene_colors = {
        "h24": "#2F5D50",
        "h08": "#C45C26",
        "s31": "#4C6A80",
        "a06": "#8B3A3A",
        "i29": "#B08968",
    }
    for scene in ALL_SCENES:
        report = summary.get("runs", {}).get(scene)
        if not report or report.get("status") != "ok":
            continue
        blocks = sorted(
            report["paper_costs"].values(),
            key=lambda b: float(b["detector_cost_ms"]),
        )
        xs = [float(b["detector_cost_ms"]) for b in blocks]
        ys = [float(b["optimized_holdout"]["expected_cost"]) for b in blocks]
        ax.plot(xs, ys, "o-", color=scene_colors.get(scene, "#333"), label=scene)
    ax.set_xscale("log")
    ax.set_xlabel("Assumed paper-Kdet cost (ms)")
    ax.set_ylabel("Holdout expected cost after anneal (ms)")
    ax.set_title("Detector-cost sensitivity — all scenes (paper Kdet)")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    path = figures_dir / "fig_detector_cost_all_scenes.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    written.append(path)

    # --- all scenes: accuracy vs Kdet cost ---
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for scene in ALL_SCENES:
        report = summary.get("runs", {}).get(scene)
        if not report or report.get("status") != "ok":
            continue
        blocks = sorted(
            report["paper_costs"].values(),
            key=lambda b: float(b["detector_cost_ms"]),
        )
        xs = [float(b["detector_cost_ms"]) for b in blocks]
        ys = [float(b["optimized_holdout"]["accuracy"]) for b in blocks]
        ax.plot(xs, ys, "o-", color=scene_colors.get(scene, "#333"), label=scene)
    ax.set_xscale("log")
    ax.set_xlabel("Assumed paper-Kdet cost (ms)")
    ax.set_ylabel("Holdout accuracy after anneal")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Detector-cost sensitivity — holdout accuracy")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    path = figures_dir / "fig_detector_cost_all_scenes_acc.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    written.append(path)

    for path in written:
        print(f"Wrote {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=list(ALL_SCENES))
    parser.add_argument("--outcomes-dir", type=Path, default=DEFAULT_OUTCOMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument(
        "--costs",
        type=float,
        nargs="+",
        default=list(DEFAULT_COST_SWEEP_MS),
        help="Paper-Kdet costs (ms) to sweep.",
    )
    parser.add_argument(
        "--skip-trained-reference",
        action="store_true",
        help="Do not run the measured-Kdet trained reference point.",
    )
    parser.add_argument("--iterations", type=int, default=8_000)
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument(
        "--split-strategy",
        choices=("blocked_per_run", "random_per_run"),
        default="blocked_per_run",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cost_sweep = tuple(float(c) for c in args.costs)

    summary: dict[str, Any] = {
        "experiment": "detector_cost_sensitivity",
        "question": (
            "As assumed paper-Kdet cost changes, how do DP layout, holdout "
            "accuracy, and expected cost change after threshold retuning?"
        ),
        "cost_sweep_ms": list(cost_sweep),
        "paper_default_ms": PAPER_DETECTOR_COST_MS,
        "annealing_iterations": args.iterations,
        "quantile_points": args.quantile_points,
        "split_strategy": args.split_strategy,
        "holdout_fraction": args.holdout_fraction,
        "random_seed": args.seed,
        "runs": {},
    }

    ordered_scenes = [s for s in ALL_SCENES if s in args.scenes]
    ordered_scenes += [s for s in args.scenes if s not in ordered_scenes]

    for scene in ordered_scenes:
        outcomes = outcome_path_for_scene(args.outcomes_dir, scene)
        print(f"\n=== {scene} ===", flush=True)
        if not outcomes.is_file():
            summary["runs"][scene] = {
                "status": "skipped",
                "reason": f"missing {outcomes}",
            }
            continue
        try:
            report = run_scene(
                scene,
                outcomes,
                cost_sweep_ms=cost_sweep,
                include_trained_reference=not args.skip_trained_reference,
                annealing_iterations=args.iterations,
                quantile_points=args.quantile_points,
                holdout_fraction=args.holdout_fraction,
                split_strategy=args.split_strategy,
                random_seed=args.seed,
            )
            report["status"] = "ok"
            run_path = args.output_dir / f"{scene}.json"
            run_path.write_text(
                json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
            )
            report["report_path"] = str(run_path)
            summary["runs"][scene] = report
            print(f"  Wrote {run_path}")
        except Exception as error:
            summary["runs"][scene] = {
                "status": "failed",
                "scene": scene,
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
        plot_figures(summary, args.figures_dir)


if __name__ == "__main__":
    main()
