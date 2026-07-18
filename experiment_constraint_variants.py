"""Constraint variants: micro vs macro vs worst-class accuracy floors.

Research question
-----------------
If we protect MACRO accuracy or WORST-CLASS accuracy (not only micro /
overall accuracy) while annealing thresholds, how do holdout accuracy,
fairness (worst class), and expected cost change vs the current
micro-accuracy floor?

Why worst-class can force higher cost than micro
------------------------------------------------
Micro accuracy is an average over samples.  A policy can look great overall
while crushing one rare class.  A worst-class floor forbids that: the annealer
must keep every class above the floor, which often means fewer aggressive
early accepts (higher thresholds / more traffic to expensive later Kis), so
expected cost rises.  Macro sits in between (each class weighted equally).

Usage
-----
    python experiment_constraint_variants.py
    python experiment_constraint_variants.py --scenes h24 --detector-modes paper
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
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/constraint_variants")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")

# (variant_name, constraint_metric, floor_mode)
# floor_mode: "baseline_<metric>" or a float literal.
VARIANT_SPECS: tuple[tuple[str, str, str | float], ...] = (
    ("micro_baseline", "micro", "baseline_micro"),
    ("macro_baseline", "macro", "baseline_macro"),
    ("worst_baseline", "worst_class", "baseline_worst"),
    ("micro_0.95", "micro", 0.95),
    ("macro_0.95", "macro", 0.95),
    ("worst_0.90", "worst_class", 0.90),
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
    }


def _acc_bundle(metrics: dict) -> dict[str, float]:
    return {
        "micro": float(metrics["accuracy"]),
        "macro": float(metrics["macro_accuracy"]),
        "worst_class": float(metrics["worst_class_accuracy"]),
        "expected_cost": float(metrics["expected_cost"]),
    }


def _speedup(baseline_cost: float, opt_cost: float) -> float | None:
    if opt_cost <= 0:
        return None
    return float(baseline_cost) / float(opt_cost)


def _baseline_key(constraint_metric: str) -> str:
    return "micro" if constraint_metric == "micro" else constraint_metric


def _resolve_baseline_floor(floor_spec: str, baselines: dict[str, float]) -> float:
    if floor_spec == "baseline_micro":
        return float(baselines["micro"])
    if floor_spec == "baseline_macro":
        return float(baselines["macro"])
    if floor_spec == "baseline_worst":
        return float(baselines["worst_class"])
    raise ValueError(f"Unknown floor_spec: {floor_spec!r}")


def _metric_from_bundle(bundle: dict[str, float], constraint_metric: str) -> float:
    if constraint_metric == "micro":
        return float(bundle["micro"])
    if constraint_metric == "macro":
        return float(bundle["macro"])
    return float(bundle["worst_class"])


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
) -> dict:
    payload = load_empirical_outcomes(outcomes_path)
    validation_payload, holdout_payload, split_meta = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=random_seed,
    )

    # Freeze DP layout once (validation only).
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

    collection_val = val_eval.evaluate()  # includes class metrics by default
    collection_hold = hold_eval.evaluate()
    baselines = {
        "micro": float(collection_val["accuracy"]),
        "macro": float(collection_val["macro_accuracy"]),
        "worst_class": float(collection_val["worst_class_accuracy"]),
    }

    variants_out: dict[str, Any] = {
        "collection": {
            "status": "ok",
            "annealed": False,
            "constraint_metric": None,
            "floor": None,
            "validation": _acc_bundle(collection_val),
            "holdout": _acc_bundle(collection_hold),
            "thresholds": {
                str(k): float(v) for k, v in collection_val["thresholds"].items()
            },
            "validation_feasible": True,
            "holdout_feasible": True,
            "holdout_speedup_vs_collection": 1.0,
        }
    }

    for idx, (name, constraint_metric, floor_spec) in enumerate(VARIANT_SPECS):
        print(f"  [{name}] metric={constraint_metric} ...", flush=True)

        # Fixed floors: skip if collection baseline of THAT metric is already
        # below the requested floor (starts infeasible at defaults).
        if isinstance(floor_spec, float):
            baseline_for_metric = float(baselines[_baseline_key(constraint_metric)])
            if baseline_for_metric + 1e-12 < float(floor_spec):
                variants_out[name] = {
                    "status": "skipped",
                    "reason": (
                        f"collection {constraint_metric}={baseline_for_metric:.4f} "
                        f"< requested floor {float(floor_spec):.4f}"
                    ),
                    "constraint_metric": constraint_metric,
                    "floor": float(floor_spec),
                    "baseline_metric_value": baseline_for_metric,
                }
                print(f"    SKIPPED: {variants_out[name]['reason']}", flush=True)
                continue
            floor = float(floor_spec)
        else:
            floor = _resolve_baseline_floor(str(floor_spec), baselines)

        annealed = optimize_fixed_layout_thresholds_simulated_annealing(
            val_eval,
            float(floor),
            quantile_points=quantile_points,
            n_iterations=annealing_iterations,
            random_seed=random_seed + 31 * (idx + 1),
            constraint_metric=constraint_metric,
        )
        thresholds = {str(k): float(v) for k, v in annealed["thresholds"].items()}
        # Full holdout / validation metrics with class stats.
        val_metrics = val_eval.evaluate(thresholds)
        hold_metrics = hold_eval.evaluate(thresholds)
        val_bundle = _acc_bundle(val_metrics)
        hold_bundle = _acc_bundle(hold_metrics)
        constrained_val = _metric_from_bundle(val_bundle, constraint_metric)
        constrained_hold = _metric_from_bundle(hold_bundle, constraint_metric)

        block = {
            "status": "ok",
            "annealed": True,
            "constraint_metric": constraint_metric,
            "floor": float(floor),
            "validation": val_bundle,
            "holdout": hold_bundle,
            "thresholds": thresholds,
            "validation_feasible": bool(constrained_val >= float(floor)),
            "holdout_feasible": bool(constrained_hold >= float(floor)),
            "holdout_speedup_vs_collection": _speedup(
                float(collection_hold["expected_cost"]),
                float(hold_metrics["expected_cost"]),
            ),
            "anneal_feasible": bool(annealed.get("feasible")),
        }
        variants_out[name] = block
        print(
            f"    holdout micro={block['holdout']['micro']:.4f}  "
            f"macro={block['holdout']['macro']:.4f}  "
            f"worst={block['holdout']['worst_class']:.4f}  "
            f"cost={block['holdout']['expected_cost']:.2f}ms  "
            f"feas={block['holdout_feasible']}",
            flush=True,
        )

    # Deltas vs micro_baseline (when present).
    micro_base = variants_out.get("micro_baseline")
    if micro_base and micro_base.get("status") == "ok":
        mb_hold = micro_base["holdout"]
        for name, block in variants_out.items():
            if block.get("status") != "ok" or "holdout" not in block:
                continue
            h = block["holdout"]
            block["delta_vs_micro_baseline"] = {
                "cost_ms": float(h["expected_cost"]) - float(mb_hold["expected_cost"]),
                "micro": float(h["micro"]) - float(mb_hold["micro"]),
                "macro": float(h["macro"]) - float(mb_hold["macro"]),
                "worst_class": float(h["worst_class"]) - float(mb_hold["worst_class"]),
            }

    return {
        "scene": scene,
        "detector_mode": detector_mode,
        "outcomes_path": str(outcomes_path),
        "split": split_meta,
        "layout": cascade_to_dict(cascade),
        "collection_baselines_validation": baselines,
        "variants": variants_out,
    }


def write_comparison_md(summary: dict, output_dir: Path) -> Path:
    rows: list[dict] = []
    for _key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        for name, block in report["variants"].items():
            if name == "collection":
                continue
            if block.get("status") == "skipped":
                rows.append(
                    {
                        "scene": report["scene"],
                        "detector_mode": report["detector_mode"],
                        "variant": name,
                        "status": "skipped",
                        "reason": block.get("reason"),
                    }
                )
                continue
            if block.get("status") != "ok":
                continue
            d = block.get("delta_vs_micro_baseline") or {}
            rows.append(
                {
                    "scene": report["scene"],
                    "detector_mode": report["detector_mode"],
                    "variant": name,
                    "status": "ok",
                    "metric": block.get("constraint_metric"),
                    "floor": block.get("floor"),
                    "holdout_micro": block["holdout"]["micro"],
                    "holdout_macro": block["holdout"]["macro"],
                    "holdout_worst": block["holdout"]["worst_class"],
                    "holdout_cost_ms": block["holdout"]["expected_cost"],
                    "holdout_feasible": block.get("holdout_feasible"),
                    "speedup": block.get("holdout_speedup_vs_collection"),
                    "d_cost_vs_micro": d.get("cost_ms"),
                    "d_worst_vs_micro": d.get("worst_class"),
                    "d_macro_vs_micro": d.get("macro"),
                }
            )

    md = [
        "# Constraint Variants — Comparison",
        "",
        "Question: if we protect **macro** or **worst-class** accuracy (not only "
        "micro) while annealing, how do holdout accuracy, fairness, and cost change?",
        "",
        "Same DP layout + same `blocked_per_run` holdout within each "
        "(scene, detector_mode). Skipped = collection baseline already below "
        "the requested fixed floor.",
        "",
        "| scene | detector | variant | floor | holdout micro | macro | worst | "
        "cost (ms) | feas | Δcost vs micro_base | Δworst vs micro_base |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        if row.get("status") == "skipped":
            md.append(
                f"| {row['scene']} | {row['detector_mode']} | {row['variant']} | "
                f"| skipped | | | | | | |"
            )
            continue
        md.append(
            "| {scene} | {det} | {var} | {floor} | {mic} | {mac} | {wst} | {cost} | "
            "{feas} | {dc} | {dw} |".format(
                scene=row["scene"],
                det=row["detector_mode"],
                var=row["variant"],
                floor=_fmt(row.get("floor")),
                mic=_fmt(row.get("holdout_micro")),
                mac=_fmt(row.get("holdout_macro")),
                wst=_fmt(row.get("holdout_worst")),
                cost=_fmt(row.get("holdout_cost_ms")),
                feas=row.get("holdout_feasible"),
                dc=_fmt(row.get("d_cost_vs_micro")),
                dw=_fmt(row.get("d_worst_vs_micro")),
            )
        )

    # Verdict: among baseline_* variants, does worst/macro cost more than micro?
    n_worst_costlier = 0
    n_macro_costlier = 0
    n_pairs = 0
    for _key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        v = report["variants"]
        micro = v.get("micro_baseline")
        macro = v.get("macro_baseline")
        worst = v.get("worst_baseline")
        if not (micro and macro and worst):
            continue
        if micro.get("status") != "ok":
            continue
        n_pairs += 1
        mc = float(micro["holdout"]["expected_cost"])
        if macro.get("status") == "ok" and float(macro["holdout"]["expected_cost"]) > mc + 1e-9:
            n_macro_costlier += 1
        if worst.get("status") == "ok" and float(worst["holdout"]["expected_cost"]) > mc + 1e-9:
            n_worst_costlier += 1

    md.extend(
        [
            "",
            "## Verdict",
            "",
            f"Among {n_pairs} settings with all three `*_baseline` variants:",
            f"- **macro_baseline** costlier than micro_baseline in **{n_macro_costlier}**",
            f"- **worst_baseline** costlier than micro_baseline in **{n_worst_costlier}**",
            "",
            "Δcost > 0 vs micro_baseline means the stricter fairness floor bought "
            "equity at a runtime price. Infeasible / skipped fixed floors are "
            "reported explicitly (negative result is fine).",
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
    c_micro = "#4C6A80"
    c_macro = "#C45C26"
    c_worst = "#2F5D50"
    c_collect = "#6B7280"
    colors = {
        "collection": c_collect,
        "micro_baseline": c_micro,
        "macro_baseline": c_macro,
        "worst_baseline": c_worst,
    }
    written: list[Path] = []
    baseline_names = ("collection", "micro_baseline", "macro_baseline", "worst_baseline")

    for detector_mode in ("paper", "trained"):
        scenes = [
            s
            for s in ALL_SCENES
            if summary.get("runs", {}).get(f"{s}__{detector_mode}", {}).get("status")
            == "ok"
        ]
        if not scenes:
            continue

        # Cost grouped bars for baseline_* (+ collection)
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        x = np.arange(len(scenes))
        width = 0.2
        for i, name in enumerate(baseline_names):
            costs = []
            for scene in scenes:
                block = summary["runs"][f"{scene}__{detector_mode}"]["variants"].get(name)
                if not block or block.get("status") != "ok":
                    costs.append(np.nan)
                else:
                    costs.append(float(block["holdout"]["expected_cost"]))
            ax.bar(
                x + (i - 1.5) * width,
                costs,
                width=width,
                color=colors[name],
                label=name,
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.set_ylabel("Holdout expected cost (ms)")
        ax.set_title(f"Constraint variants — cost ({detector_mode})")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = figures_dir / f"fig_constraint_cost_{detector_mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

        # Worst-class accuracy bars
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        for i, name in enumerate(baseline_names):
            vals = []
            for scene in scenes:
                block = summary["runs"][f"{scene}__{detector_mode}"]["variants"].get(name)
                if not block or block.get("status") != "ok":
                    vals.append(np.nan)
                else:
                    vals.append(float(block["holdout"]["worst_class"]))
            ax.bar(
                x + (i - 1.5) * width,
                vals,
                width=width,
                color=colors[name],
                label=name,
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.set_ylabel("Holdout worst-class accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Constraint variants — worst-class ({detector_mode})")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = figures_dir / f"fig_constraint_worst_{detector_mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

        # h24 scatter: cost vs worst-class for all ok variants
        key = f"h24__{detector_mode}"
        report = summary.get("runs", {}).get(key)
        if report and report.get("status") == "ok":
            fig, ax = plt.subplots(figsize=(5.8, 4.2))
            for name, block in report["variants"].items():
                if block.get("status") != "ok":
                    continue
                cost = float(block["holdout"]["expected_cost"])
                worst = float(block["holdout"]["worst_class"])
                micro = float(block["holdout"]["micro"])
                color = colors.get(name, "#8B3A3A")
                ax.scatter(cost, worst, s=55, c=color, zorder=3)
                ax.annotate(name, (cost, worst), textcoords="offset points", xytext=(4, 3), fontsize=7)
                _ = micro
            ax.set_xlabel("Holdout expected cost (ms)")
            ax.set_ylabel("Holdout worst-class accuracy")
            ax.set_title(f"h24 constraint Pareto-ish cloud ({detector_mode})")
            fig.tight_layout()
            path = figures_dir / f"fig_constraint_h24_scatter_{detector_mode}.png"
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
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "experiment": "constraint_variants",
        "question": (
            "If we protect macro or worst-class accuracy (not only micro) while "
            "annealing thresholds, how do holdout accuracy, fairness, and cost change?"
        ),
        "variants": [v[0] for v in VARIANT_SPECS],
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
