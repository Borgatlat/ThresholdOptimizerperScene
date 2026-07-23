"""Multi-seed stability of accuracy-budget (+ winning stacked) speedups.

Research question
-----------------
Are the large speedups from 2–3pp accuracy budgets (and paper's stacked
lower-Kdet + 2pp recipe) **stable across random seeds**, or lucky on one
holdout split?

Why multi-seed matters
----------------------
- One ``blocked_per_run`` split is deterministic (seed does not change the
  holdout mask) — so re-running with seed=1 on blocked would NOT test
  partition variance.
- ``random_per_run`` + different seeds *does* change who is in holdout.
- Annealing is also stochastic; we tie anneal seed to the split seed so
  each trial is a full end-to-end resample.

Usage
-----
    python experiment_multiseed_stability.py
    python experiment_multiseed_stability.py --scenes h24 --detector-modes paper --seeds 0 1 2
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
from experiment_threshold_variants import make_cascade
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


ALL_SCENES = ("h24", "h08")  # stability check — keep small
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_OUTCOMES_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/multiseed_stability")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")
PAPER_KDET = float(PAPER_DETECTOR_COST_MS)  # 10_000
LOW_KDET = 1_000.0


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


def _strip_detector(chain: list[str]) -> list[str]:
    return [c for c in chain if c != "detector"]


def _speedup(baseline_cost: float | None, opt_cost: float | None) -> float | None:
    if baseline_cost is None or opt_cost is None or float(opt_cost) <= 0:
        return None
    return float(baseline_cost) / float(opt_cost)


def _mean_std(values: list[float]) -> dict[str, float | None]:
    """Mean and sample std (ddof=1). Empty list → Nones.

    What if you used ddof=0?
        Population std (divide by n).  For n=5 seeds, sample std (n-1) is the
        usual "how much would this bounce on a new seed" estimator.
    """
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def anneal_recipe(
    validation_payload: dict,
    holdout_payload: dict,
    *,
    cascade: Cascade,
    detector_mode: str,
    detector_cost_ms: float,
    budget: float,
    annealing_iterations: int,
    quantile_points: int,
    random_seed: int,
) -> dict:
    """Anneal one layout at floor = collection_micro − budget; score holdout."""
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
    # floor = how low micro may go on validation while still "feasible".
    # budget=0.02 → allow 2 percentage points below collection.
    floor = max(0.0, baseline_micro - float(budget))

    annealed = optimize_fixed_layout_thresholds_simulated_annealing(
        val_eval,
        float(floor),
        quantile_points=quantile_points,
        n_iterations=annealing_iterations,
        random_seed=random_seed,
        constraint_metric="micro",
    )
    thresholds = {str(k): float(v) for k, v in annealed["thresholds"].items()}
    opt_val = val_eval.evaluate(thresholds)
    opt_hold = hold_eval.evaluate(thresholds)
    return {
        "budget": float(budget),
        "floor": float(floor),
        "baseline_micro_validation": baseline_micro,
        "layout": cascade_to_dict(cascade),
        "thresholds": thresholds,
        "collection_holdout": {
            "accuracy": float(collection_hold["accuracy"]),
            "macro_accuracy": float(collection_hold["macro_accuracy"]),
            "worst_class_accuracy": float(collection_hold["worst_class_accuracy"]),
            "expected_cost": float(collection_hold["expected_cost"]),
        },
        "validation": {
            "accuracy": float(opt_val["accuracy"]),
            "expected_cost": float(opt_val["expected_cost"]),
            "feasible": bool(float(opt_val["accuracy"]) >= floor),
        },
        "holdout": {
            "accuracy": float(opt_hold["accuracy"]),
            "macro_accuracy": float(opt_hold["macro_accuracy"]),
            "worst_class_accuracy": float(opt_hold["worst_class_accuracy"]),
            "expected_cost": float(opt_hold["expected_cost"]),
        },
        "holdout_feasible": bool(float(opt_hold["accuracy"]) >= floor),
        "anneal_feasible": bool(annealed.get("feasible")),
    }


def run_one_seed(
    scene: str,
    outcomes_path: Path,
    *,
    detector_mode: str,
    seed: int,
    split_strategy: str,
    annealing_iterations: int,
    quantile_points: int,
    holdout_fraction: float,
) -> dict:
    """One (scene, mode, seed): shared split for every recipe on that seed."""
    payload = load_empirical_outcomes(outcomes_path)
    validation_payload, holdout_payload, split_meta = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=seed,
    )

    default_cost = (
        PAPER_KDET
        if detector_mode == "paper"
        else float(payload["detector"]["cost"])
    )

    # DP layout once at default Kdet (shared by A/B/C).
    dp_cascade = HierarchyOptimizer(
        validation_payload,
        detector_mode=detector_mode,
        detector_cost_ms=default_cost,
    ).synthesize()
    dp_order = _strip_detector(dp_cascade.initial)
    dp_spec = {k: list(v) for k, v in dp_cascade.specialized.items()}
    dp_made = make_cascade(dp_order, specialized=dp_spec)

    recipes: dict[str, Any] = {}

    def go(name: str, cascade: Cascade, cost: float, budget: float, seed_off: int) -> dict:
        print(
            f"    [{name}] cost={cost:g} budget={budget} ...",
            flush=True,
        )
        block = anneal_recipe(
            validation_payload,
            holdout_payload,
            cascade=cascade,
            detector_mode=detector_mode,
            detector_cost_ms=cost,
            budget=budget,
            annealing_iterations=annealing_iterations,
            quantile_points=quantile_points,
            random_seed=seed * 1000 + seed_off,
        )
        block["recipe"] = name
        block["requested_detector_cost_ms"] = float(cost)
        print(
            f"      holdout acc={block['holdout']['accuracy']:.4f}  "
            f"cost={block['holdout']['expected_cost']:.2f}ms",
            flush=True,
        )
        return block

    recipes["baseline_protect"] = go("baseline_protect", dp_made, default_cost, 0.0, 1)
    recipes["budget_2pp"] = go("budget_2pp", dp_made, default_cost, 0.02, 2)
    recipes["budget_3pp"] = go("budget_3pp", dp_made, default_cost, 0.03, 3)

    if detector_mode == "paper":
        # Re-synthesize DP at lower Kdet — structure can change (shorter chain).
        low_cascade = HierarchyOptimizer(
            validation_payload,
            detector_mode="paper",
            detector_cost_ms=LOW_KDET,
        ).synthesize()
        low_made = make_cascade(
            _strip_detector(low_cascade.initial),
            specialized={k: list(v) for k, v in low_cascade.specialized.items()},
        )
        recipes["stacked_kdet1000_budget_2pp"] = go(
            "stacked_kdet1000_budget_2pp", low_made, LOW_KDET, 0.02, 4
        )
    else:
        recipes["stacked_kdet1000_budget_2pp"] = {
            "status": "skipped",
            "reason": "paper-only (trained Kdet is measured, not synthetic)",
        }

    base = recipes["baseline_protect"]
    base_acc = float(base["holdout"]["accuracy"])
    base_cost = float(base["holdout"]["expected_cost"])
    for name, block in recipes.items():
        if block.get("status") == "skipped" or "holdout" not in block:
            continue
        h = block["holdout"]
        block["delta_vs_baseline_protect"] = {
            "accuracy": float(h["accuracy"]) - base_acc,
            "cost_ms": float(h["expected_cost"]) - base_cost,
            "speedup": _speedup(base_cost, float(h["expected_cost"])),
        }

    return {
        "scene": scene,
        "detector_mode": detector_mode,
        "seed": int(seed),
        "split": split_meta,
        "default_detector_cost_ms": default_cost,
        "recipes": recipes,
    }


def aggregate_seeds(seed_reports: list[dict]) -> dict:
    """Collapse per-seed recipe metrics into mean±std + stability fractions."""
    recipe_names: list[str] = []
    for report in seed_reports:
        for name in report["recipes"]:
            if name not in recipe_names:
                recipe_names.append(name)

    out: dict[str, Any] = {}
    for name in recipe_names:
        accs, costs, speedups, d_accs = [], [], [], []
        within_3pp = 0
        speedup_ge_2 = 0
        n_ok = 0
        n_skipped = 0
        for report in seed_reports:
            block = report["recipes"].get(name)
            if not block:
                continue
            if block.get("status") == "skipped":
                n_skipped += 1
                continue
            if "holdout" not in block:
                continue
            n_ok += 1
            d = block.get("delta_vs_baseline_protect") or {}
            accs.append(float(block["holdout"]["accuracy"]))
            costs.append(float(block["holdout"]["expected_cost"]))
            sp = d.get("speedup")
            if sp is not None:
                speedups.append(float(sp))
                if float(sp) >= 2.0:
                    speedup_ge_2 += 1
            da = d.get("accuracy")
            if da is not None:
                d_accs.append(float(da))
                if float(da) >= -0.03:
                    within_3pp += 1

        out[name] = {
            "n_ok": n_ok,
            "n_skipped": n_skipped,
            "holdout_accuracy": _mean_std(accs),
            "holdout_expected_cost": _mean_std(costs),
            "speedup_vs_baseline_protect": _mean_std(speedups),
            "delta_acc_vs_baseline_protect": _mean_std(d_accs),
            "fraction_seeds_delta_acc_ge_neg3pp": (
                float(within_3pp) / n_ok if n_ok else None
            ),
            "fraction_seeds_speedup_ge_2x": (
                float(speedup_ge_2) / n_ok if n_ok else None
            ),
            "stable_speedup": bool(
                n_ok >= 2
                and _mean_std(speedups)["mean"] is not None
                and float(_mean_std(speedups)["mean"]) >= 2.0
                and float(_mean_std(speedups)["std"] or 0.0)
                <= 0.5 * float(_mean_std(speedups)["mean"])
                and (within_3pp / n_ok) >= 0.8
            )
            if n_ok
            else False,
        }
    return out


def run_scene_mode(
    scene: str,
    outcomes_path: Path,
    *,
    detector_mode: str,
    seeds: tuple[int, ...],
    annealing_iterations: int,
    quantile_points: int,
    holdout_fraction: float,
) -> dict:
    seed_reports: list[dict] = []
    for seed in seeds:
        print(f"  --- seed={seed} (random_per_run) ---", flush=True)
        seed_reports.append(
            run_one_seed(
                scene,
                outcomes_path,
                detector_mode=detector_mode,
                seed=seed,
                split_strategy="random_per_run",
                annealing_iterations=annealing_iterations,
                quantile_points=quantile_points,
                holdout_fraction=holdout_fraction,
            )
        )

    # Optional blocked reference at seed 0 (matches prior experiment protocol).
    print("  --- blocked_per_run reference (seed=0) ---", flush=True)
    blocked_ref = run_one_seed(
        scene,
        outcomes_path,
        detector_mode=detector_mode,
        seed=0,
        split_strategy="blocked_per_run",
        annealing_iterations=annealing_iterations,
        quantile_points=quantile_points,
        holdout_fraction=holdout_fraction,
    )

    return {
        "scene": scene,
        "detector_mode": detector_mode,
        "split_strategy_primary": "random_per_run",
        "seeds": list(seeds),
        "note": (
            "Primary multi-seed uses random_per_run because blocked_per_run "
            "ignores the seed when building the holdout mask."
        ),
        "per_seed": seed_reports,
        "aggregate": aggregate_seeds(seed_reports),
        "blocked_per_run_seed0_reference": blocked_ref,
    }


def write_comparison_md(summary: dict, output_dir: Path) -> Path:
    md = [
        "# Multi-seed Stability — Comparison",
        "",
        "Question: are accuracy-budget / stacked-Kdet speedups **stable** "
        "across `random_per_run` seeds, or seed-0 luck?",
        "",
        "Protocol: 5 seeds, `random_per_run` 80/20, anneal 8000 iters. "
        "`blocked_per_run` seed=0 kept only as a reference (that split does "
        "**not** vary with seed).",
        "",
        "| scene | detector | recipe | mean±std speedup | mean±std Δacc | "
        "frac Δacc≥−3pp | frac ≥2× | stable? |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    recipe_order = [
        "baseline_protect",
        "budget_2pp",
        "budget_3pp",
        "stacked_kdet1000_budget_2pp",
    ]
    for _key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        agg = report.get("aggregate") or {}
        for name in recipe_order:
            block = agg.get(name)
            if not block:
                continue
            if block.get("n_skipped") and not block.get("n_ok"):
                md.append(
                    f"| {report['scene']} | {report['detector_mode']} | {name} | "
                    f"skipped | | | | |"
                )
                continue
            sp = block["speedup_vs_baseline_protect"]
            da = block["delta_acc_vs_baseline_protect"]
            md.append(
                "| {scene} | {det} | {rec} | {sp} | {da} | {f3} | {f2} | {st} |".format(
                    scene=report["scene"],
                    det=report["detector_mode"],
                    rec=name,
                    sp=_fmt_mean_std(sp),
                    da=_fmt_mean_std(da),
                    f3=_fmt(block.get("fraction_seeds_delta_acc_ge_neg3pp")),
                    f2=_fmt(block.get("fraction_seeds_speedup_ge_2x")),
                    st="YES" if block.get("stable_speedup") else "no",
                )
            )

    md.extend(["", "## Verdict", ""])
    for _key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        agg = report.get("aggregate") or {}
        stable = [
            name
            for name, block in agg.items()
            if name != "baseline_protect" and block.get("stable_speedup")
        ]
        md.append(
            f"- **{report['scene']}/{report['detector_mode']}** stable recipes: "
            f"{', '.join(f'`{n}`' for n in stable) if stable else '_none_'}"
        )
    md.extend(
        [
            "",
            "If `budget_3pp` or `stacked_kdet1000_budget_2pp` is **not** stable, "
            "treat the single-seed vacation numbers as provisional for a paper.",
            "",
        ]
    )
    path = output_dir / "COMPARISON.md"
    path.write_text("\n".join(md) + "\n")
    print(f"Wrote {path}")
    return path


def _fmt(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_mean_std(block: dict | None) -> str:
    if not block or block.get("mean") is None:
        return ""
    mean = float(block["mean"])
    std = block.get("std")
    if std is None:
        return f"{mean:.4f}"
    return f"{mean:.4f}±{float(std):.4f}"


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
    colors = {
        "baseline_protect": "#6B7280",
        "budget_2pp": "#B08968",
        "budget_3pp": "#C45C26",
        "stacked_kdet1000_budget_2pp": "#2F5D50",
    }
    written: list[Path] = []

    for mode in ("paper", "trained"):
        key = f"h24__{mode}"
        report = summary.get("runs", {}).get(key)
        if not report or report.get("status") != "ok":
            continue
        recipe_order = [
            n
            for n in (
                "budget_2pp",
                "budget_3pp",
                "stacked_kdet1000_budget_2pp",
            )
            if (report["aggregate"].get(n) or {}).get("n_ok")
        ]
        if not recipe_order:
            continue

        # Speedup strip/box across seeds
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        data, labels, cols = [], [], []
        for name in recipe_order:
            ys = []
            for seed_rep in report["per_seed"]:
                block = seed_rep["recipes"].get(name) or {}
                if block.get("status") == "skipped":
                    continue
                sp = (block.get("delta_vs_baseline_protect") or {}).get("speedup")
                if sp is not None:
                    ys.append(float(sp))
            if not ys:
                continue
            data.append(ys)
            labels.append(name.replace("stacked_", "S:"))
            cols.append(colors.get(name, "#333"))
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
            for patch, col in zip(bp["boxes"], cols):
                patch.set_facecolor(col)
                patch.set_alpha(0.55)
            # Overlay individual seeds (strip).
            for i, ys in enumerate(data):
                x = np.full(len(ys), i + 1, dtype=float)
                # jitter: small noise so overlapping points are visible
                x = x + (np.random.default_rng(0).uniform(-0.08, 0.08, size=len(ys)))
                ax.scatter(x, ys, s=28, c=cols[i], zorder=3, edgecolors="white", linewidths=0.4)
            ax.axhline(1.0, color="#6B7280", linewidth=0.8)
            ax.set_ylabel("Speedup vs baseline_protect")
            ax.set_title(f"h24 multi-seed speedup ({mode}, random_per_run)")
            fig.tight_layout()
            path = figures_dir / f"fig_multiseed_h24_speedup_{mode}.png"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
            written.append(path)

        # Δacc strip
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        data, labels, cols = [], [], []
        for name in recipe_order:
            ys = []
            for seed_rep in report["per_seed"]:
                block = seed_rep["recipes"].get(name) or {}
                if block.get("status") == "skipped":
                    continue
                da = (block.get("delta_vs_baseline_protect") or {}).get("accuracy")
                if da is not None:
                    ys.append(float(da) * 100.0)  # percentage points
            if not ys:
                continue
            data.append(ys)
            labels.append(name.replace("stacked_", "S:"))
            cols.append(colors.get(name, "#333"))
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
            for patch, col in zip(bp["boxes"], cols):
                patch.set_facecolor(col)
                patch.set_alpha(0.55)
            for i, ys in enumerate(data):
                x = np.full(len(ys), i + 1, dtype=float)
                x = x + (np.random.default_rng(1).uniform(-0.08, 0.08, size=len(ys)))
                ax.scatter(x, ys, s=28, c=cols[i], zorder=3, edgecolors="white", linewidths=0.4)
            ax.axhline(0.0, color="#6B7280", linewidth=0.8)
            ax.axhline(-3.0, color="#C45C26", linewidth=0.8, linestyle="--")
            ax.set_ylabel("Δacc vs baseline_protect (pp)")
            ax.set_title(f"h24 multi-seed Δacc ({mode}, random_per_run)")
            fig.tight_layout()
            path = figures_dir / f"fig_multiseed_h24_delta_acc_{mode}.png"
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
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--outcomes-dir", type=Path, default=DEFAULT_OUTCOMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--iterations", type=int, default=8_000)
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "experiment": "multiseed_stability",
        "question": (
            "Are accuracy-budget / stacked-Kdet speedups stable across "
            "random_per_run seeds?"
        ),
        "annealing_iterations": args.iterations,
        "seeds": list(args.seeds),
        "prompt": "prompts/multiseed_budget_stability.md",
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
                report = run_scene_mode(
                    scene,
                    outcomes,
                    detector_mode=detector_mode,
                    seeds=tuple(args.seeds),
                    annealing_iterations=args.iterations,
                    quantile_points=args.quantile_points,
                    holdout_fraction=args.holdout_fraction,
                )
                report["status"] = "ok"
                path = args.output_dir / f"{key}.json"
                path.write_text(
                    json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
                )
                report["report_path"] = str(path)
                summary["runs"][key] = report
                print(f"  Wrote {path}")
                for name, agg in report["aggregate"].items():
                    if not agg.get("n_ok"):
                        continue
                    sp = agg["speedup_vs_baseline_protect"]
                    print(
                        f"  agg {name}: speedup={_fmt_mean_std(sp)} "
                        f"stable={agg.get('stable_speedup')}"
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
        plot_figures(summary, args.figures_dir)


if __name__ == "__main__":
    main()
