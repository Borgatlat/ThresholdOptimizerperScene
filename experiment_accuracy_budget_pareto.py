"""Accuracy-budget Pareto: how much speed for a small accuracy give?

Research question
-----------------
If we allow the annealer's micro-accuracy floor to drop by a *controlled*
budget (0%, 0.5%, 1%, 2%, …) below the collection baseline, how much
holdout expected-cost / speedup do we buy, and does accuracy actually fall
by about that much on holdout?

This is the practical "speed without major accuracy loss" sweep: same DP
layout, same holdout, only the accuracy floor changes.

Usage
-----
    python experiment_accuracy_budget_pareto.py
    python experiment_accuracy_budget_pareto.py --scenes h24 --detector-modes paper
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
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/accuracy_budget")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")

# Accuracy *shortfall* allowed below collection baseline micro accuracy.
# 0.0 = protect baseline (current default). 0.01 = allow 1 percentage point drop.
DEFAULT_BUDGETS: tuple[float, ...] = (0.0, 0.005, 0.01, 0.02, 0.03)


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
    }


def _budget_label(budget: float) -> str:
    """0.0 -> budget_0pp, 0.005 -> budget_0p5pp, 0.01 -> budget_1pp."""
    pp = budget * 100.0
    if abs(pp - round(pp)) < 1e-9:
        return f"budget_{int(round(pp))}pp"
    # one decimal place, 'p' for the decimal point (filename-safe)
    return f"budget_{str(pp).replace('.', 'p')}pp"


def _speedup(baseline_cost: float, opt_cost: float) -> float | None:
    if opt_cost <= 0:
        return None
    return float(baseline_cost) / float(opt_cost)


def run_scene_detector(
    scene: str,
    outcomes_path: Path,
    *,
    detector_mode: str,
    detector_cost_ms: float,
    budgets: tuple[float, ...],
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

    # Freeze DP layout once — only the accuracy floor varies.
    dp_opt = HierarchyOptimizer(
        validation_payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    cascade = dp_opt.synthesize()

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
    baseline_micro = float(collection_val["accuracy"])

    budgets_out: dict[str, Any] = {
        "collection": {
            "status": "ok",
            "annealed": False,
            "budget": None,
            "floor": None,
            "validation": {
                "accuracy": float(collection_val["accuracy"]),
                "macro_accuracy": float(collection_val["macro_accuracy"]),
                "worst_class_accuracy": float(collection_val["worst_class_accuracy"]),
                "expected_cost": float(collection_val["expected_cost"]),
            },
            "holdout": {
                "accuracy": float(collection_hold["accuracy"]),
                "macro_accuracy": float(collection_hold["macro_accuracy"]),
                "worst_class_accuracy": float(collection_hold["worst_class_accuracy"]),
                "expected_cost": float(collection_hold["expected_cost"]),
            },
            "thresholds": {
                str(k): float(v) for k, v in collection_val["thresholds"].items()
            },
            "holdout_feasible": True,
            "holdout_speedup_vs_collection": 1.0,
        }
    }

    zero_label = None
    for i, budget in enumerate(budgets):
        label = _budget_label(budget)
        if abs(budget) < 1e-15:
            zero_label = label
        # Floor cannot go below 0; if baseline is already tiny, budget may
        # collapse the floor to 0 (accept almost anything) — still report it.
        floor = max(0.0, baseline_micro - float(budget))
        print(
            f"  [{label}] floor={floor:.4f} "
            f"(baseline={baseline_micro:.4f} - {budget:.4f}) ...",
            flush=True,
        )
        annealed = optimize_fixed_layout_thresholds_simulated_annealing(
            val_eval,
            float(floor),
            quantile_points=quantile_points,
            n_iterations=annealing_iterations,
            random_seed=random_seed + 53 * (i + 1),
            constraint_metric="micro",
        )
        thresholds = {str(k): float(v) for k, v in annealed["thresholds"].items()}
        opt_val = val_eval.evaluate(thresholds)
        opt_hold = hold_eval.evaluate(thresholds)
        block = {
            "status": "ok",
            "annealed": True,
            "budget": float(budget),
            "floor": float(floor),
            "baseline_micro_validation": baseline_micro,
            "validation": {
                "accuracy": float(opt_val["accuracy"]),
                "macro_accuracy": float(opt_val["macro_accuracy"]),
                "worst_class_accuracy": float(opt_val["worst_class_accuracy"]),
                "expected_cost": float(opt_val["expected_cost"]),
            },
            "holdout": {
                "accuracy": float(opt_hold["accuracy"]),
                "macro_accuracy": float(opt_hold["macro_accuracy"]),
                "worst_class_accuracy": float(opt_hold["worst_class_accuracy"]),
                "expected_cost": float(opt_hold["expected_cost"]),
                "route_counts": opt_hold.get("route_counts"),
            },
            "thresholds": thresholds,
            "validation_feasible": bool(float(opt_val["accuracy"]) >= floor),
            "holdout_feasible": bool(float(opt_hold["accuracy"]) >= floor),
            "holdout_speedup_vs_collection": _speedup(
                float(collection_hold["expected_cost"]),
                float(opt_hold["expected_cost"]),
            ),
            "anneal_feasible": bool(annealed.get("feasible")),
        }
        budgets_out[label] = block
        print(
            f"    holdout acc={block['holdout']['accuracy']:.4f}  "
            f"cost={block['holdout']['expected_cost']:.2f}ms  "
            f"speedup={block['holdout_speedup_vs_collection'] or float('nan'):.3f}x  "
            f"feas={block['holdout_feasible']}",
            flush=True,
        )

    # Deltas vs budget_0pp (protect baseline) when present.
    ref = budgets_out.get(zero_label) if zero_label else None
    if ref and ref.get("status") == "ok":
        ref_hold = ref["holdout"]
        for label, block in budgets_out.items():
            if not block.get("annealed"):
                continue
            h = block["holdout"]
            block["delta_vs_budget_0pp"] = {
                "cost_ms": float(h["expected_cost"]) - float(ref_hold["expected_cost"]),
                "accuracy": float(h["accuracy"]) - float(ref_hold["accuracy"]),
                "speedup_ratio": (
                    float(ref_hold["expected_cost"]) / float(h["expected_cost"])
                    if float(h["expected_cost"]) > 0
                    else None
                ),
            }

    return {
        "scene": scene,
        "detector_mode": detector_mode,
        "outcomes_path": str(outcomes_path),
        "split": split_meta,
        "layout": cascade_to_dict(cascade),
        "baseline_micro_validation": baseline_micro,
        "budgets": budgets_out,
    }


def write_comparison_md(summary: dict, output_dir: Path) -> Path:
    rows: list[dict] = []
    for _key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        for label, block in report["budgets"].items():
            if label == "collection" or block.get("status") != "ok":
                continue
            d = block.get("delta_vs_budget_0pp") or {}
            rows.append(
                {
                    "scene": report["scene"],
                    "detector_mode": report["detector_mode"],
                    "budget_label": label,
                    "budget": block.get("budget"),
                    "floor": block.get("floor"),
                    "holdout_acc": block["holdout"]["accuracy"],
                    "holdout_cost_ms": block["holdout"]["expected_cost"],
                    "speedup_vs_collection": block.get("holdout_speedup_vs_collection"),
                    "feasible": block.get("holdout_feasible"),
                    "d_cost_vs_0pp": d.get("cost_ms"),
                    "d_acc_vs_0pp": d.get("accuracy"),
                    "speedup_vs_0pp": d.get("speedup_ratio"),
                }
            )

    md = [
        "# Accuracy-Budget Pareto — Comparison",
        "",
        "Question: how much **speed** do we buy by allowing a small, controlled "
        "drop in the micro-accuracy floor below the collection baseline?",
        "",
        "Same DP layout + same `blocked_per_run` holdout within each "
        "(scene, detector_mode). `budget_0pp` = protect baseline (current default).",
        "",
        "| scene | detector | budget | floor | holdout acc | cost (ms) | "
        "speedup vs collect | feas | Δcost vs 0pp | Δacc vs 0pp | speedup vs 0pp |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            "| {scene} | {det} | {lab} | {floor} | {acc} | {cost} | {spd} | {feas} | "
            "{dc} | {da} | {sv} |".format(
                scene=row["scene"],
                det=row["detector_mode"],
                lab=row["budget_label"],
                floor=_fmt(row["floor"]),
                acc=_fmt(row["holdout_acc"]),
                cost=_fmt(row["holdout_cost_ms"]),
                spd=_fmt(row["speedup_vs_collection"]),
                feas=row["feasible"],
                dc=_fmt(row["d_cost_vs_0pp"]),
                da=_fmt(row["d_acc_vs_0pp"]),
                sv=_fmt(row["speedup_vs_0pp"]),
            )
        )

    # Verdict: at 1pp budget, median speedup vs 0pp on paper mode.
    speedups_1pp_paper: list[float] = []
    for row in rows:
        if (
            row["detector_mode"] == "paper"
            and row["budget_label"] == "budget_1pp"
            and row.get("speedup_vs_0pp") is not None
        ):
            speedups_1pp_paper.append(float(row["speedup_vs_0pp"]))

    md.extend(["", "## Verdict", ""])
    if speedups_1pp_paper:
        med = float(np.median(speedups_1pp_paper))
        md.append(
            f"At a **1pp** accuracy budget (paper Kdet), median speedup vs "
            f"`budget_0pp` across scenes: **{med:.2f}×** "
            f"(n={len(speedups_1pp_paper)})."
        )
    md.append(
        "Negative Δcost vs 0pp means the relaxed floor found a cheaper policy. "
        "Check Δacc: a large unexpected accuracy drop means the budget was "
        "spent aggressively; a small drop with big speedup is the sweet spot."
    )
    md.append("")

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
    scene_colors = {
        "h24": "#2F5D50",
        "h08": "#C45C26",
        "s31": "#4C6A80",
        "a06": "#8B3A3A",
        "i29": "#B08968",
    }
    written: list[Path] = []

    for detector_mode in ("paper", "trained"):
        # Pareto: holdout acc vs cost, one curve per scene
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        any_drawn = False
        for scene in ALL_SCENES:
            key = f"{scene}__{detector_mode}"
            report = summary.get("runs", {}).get(key)
            if not report or report.get("status") != "ok":
                continue
            points = []
            for label, block in report["budgets"].items():
                if not block.get("annealed"):
                    continue
                points.append(
                    (
                        float(block["budget"]),
                        float(block["holdout"]["expected_cost"]),
                        float(block["holdout"]["accuracy"]),
                    )
                )
            if not points:
                continue
            points.sort(key=lambda p: p[0])
            costs = [p[1] for p in points]
            accs = [p[2] for p in points]
            ax.plot(
                costs,
                accs,
                "o-",
                color=scene_colors.get(scene, "#333"),
                label=scene,
            )
            any_drawn = True
        if any_drawn:
            ax.set_xlabel("Holdout expected cost (ms)")
            ax.set_ylabel("Holdout accuracy")
            ax.set_title(f"Accuracy–cost Pareto by accuracy budget ({detector_mode})")
            ax.legend(frameon=False)
            fig.tight_layout()
            path = figures_dir / f"fig_accuracy_budget_pareto_{detector_mode}.png"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
            written.append(path)
        else:
            plt.close(fig)

        # Speedup vs budget (grouped / lines)
        fig, ax = plt.subplots(figsize=(6.8, 3.8))
        any_drawn = False
        for scene in ALL_SCENES:
            key = f"{scene}__{detector_mode}"
            report = summary.get("runs", {}).get(key)
            if not report or report.get("status") != "ok":
                continue
            xs, ys = [], []
            for label, block in report["budgets"].items():
                if not block.get("annealed"):
                    continue
                xs.append(float(block["budget"]) * 100.0)
                ys.append(float(block["holdout_speedup_vs_collection"] or 0.0))
            order = np.argsort(xs)
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax.plot(xs, ys, "o-", color=scene_colors.get(scene, "#333"), label=scene)
            any_drawn = True
        if any_drawn:
            ax.set_xlabel("Accuracy budget (percentage points below baseline)")
            ax.set_ylabel("Speedup vs collection thresholds")
            ax.set_title(f"Speedup vs accuracy budget ({detector_mode})")
            ax.legend(frameon=False, ncol=3)
            fig.tight_layout()
            path = figures_dir / f"fig_accuracy_budget_speedup_{detector_mode}.png"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
            written.append(path)
        else:
            plt.close(fig)

        # h24 bar: cost by budget
        key = f"h24__{detector_mode}"
        report = summary.get("runs", {}).get(key)
        if report and report.get("status") == "ok":
            items = sorted(
                (
                    (float(b["budget"]), float(b["holdout"]["expected_cost"]), lab)
                    for lab, b in report["budgets"].items()
                    if b.get("annealed")
                ),
                key=lambda t: t[0],
            )
            if items:
                fig, ax = plt.subplots(figsize=(6.2, 3.6))
                labels = [f"{b*100:g}pp" for b, _, _ in items]
                costs = [c for _, c, _ in items]
                ax.bar(
                    np.arange(len(labels)),
                    costs,
                    color="#C45C26",
                    edgecolor="white",
                    linewidth=0.4,
                )
                ax.set_xticks(np.arange(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_xlabel("Accuracy budget")
                ax.set_ylabel("Holdout expected cost (ms)")
                ax.set_title(f"h24 holdout cost vs accuracy budget ({detector_mode})")
                fig.tight_layout()
                path = figures_dir / f"fig_accuracy_budget_h24_cost_{detector_mode}.png"
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
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help="Accuracy shortfalls below baseline (e.g. 0 0.005 0.01 0.02).",
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
    parser.add_argument("--detector-cost-ms", type=float, default=PAPER_DETECTOR_COST_MS)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    budgets = tuple(float(b) for b in args.budgets)
    if any(b < 0 for b in budgets):
        raise SystemExit("budgets must be >= 0")

    summary: dict[str, Any] = {
        "experiment": "accuracy_budget_pareto",
        "question": (
            "How much holdout speedup do we buy by allowing a small controlled "
            "drop in the micro-accuracy floor below the collection baseline?"
        ),
        "budgets": list(budgets),
        "annealing_iterations": args.iterations,
        "quantile_points": args.quantile_points,
        "split_strategy": args.split_strategy,
        "holdout_fraction": args.holdout_fraction,
        "random_seed": args.seed,
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
                continue
            try:
                report = run_scene_detector(
                    scene,
                    outcomes,
                    detector_mode=detector_mode,
                    detector_cost_ms=args.detector_cost_ms,
                    budgets=budgets,
                    annealing_iterations=args.iterations,
                    quantile_points=args.quantile_points,
                    holdout_fraction=args.holdout_fraction,
                    split_strategy=args.split_strategy,
                    random_seed=args.seed,
                )
                report["status"] = "ok"
                run_path = args.output_dir / f"{key}.json"
                run_path.write_text(
                    json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
                )
                report["report_path"] = str(run_path)
                summary["runs"][key] = report
                print(f"  Wrote {run_path}")
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
        plot_figures(summary, args.figures_dir)


if __name__ == "__main__":
    main()
