"""Cross-scene tune → select → test (README experiment #8).

Research question
-----------------
Do thresholds + cascade layout tuned on scene A transfer?  We:
  1) TUNE   on scene A (DP synthesize + anneal on A's validation split)
  2) SELECT the best source policy using scene B (no further annealing)
  3) TEST   the selected policy on scene C

This is stricter than single-scene holdout and stronger than plain zero-shot
transfer (which skips the select step).

Also includes a speed-oriented candidate set: budget_0pp (protect baseline)
and budget_2pp (allow 2pp micro shortfall) per tune scene.

Usage
-----
    python experiment_cross_scene_tune_select_test.py
    python experiment_cross_scene_tune_select_test.py --scenes h24 h08 s31 --detector-modes paper
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
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


ALL_SCENES = ("h24", "h08", "s31", "a06", "i29")
DEFAULT_OUTCOMES_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/cross_scene")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")
# Candidates per tune scene: protect baseline, and a small speed budget.
DEFAULT_BUDGETS: tuple[float, ...] = (0.0, 0.02)


def outcome_path_for_scene(outcomes_dir: Path, scene: str) -> Path:
    if scene == "h24":
        return outcomes_dir / "empirical_outcomes.pkl"
    return outcomes_dir / f"empirical_outcomes_{scene}.pkl"


def cascade_from_layout(layout: dict) -> Cascade:
    specialized = {
        tuple(key.split(":", 1)): list(chain)
        for key, chain in layout["specialized"].items()
    }
    return Cascade(
        expected_cost=float(layout.get("expected_cost_dp", 0.0)),
        initial=list(layout["initial"]),
        specialized=specialized,  # type: ignore[arg-type]
        detector=layout.get("detector", "detector"),
    )


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


def _budget_tag(budget: float) -> str:
    pp = budget * 100.0
    if abs(pp - round(pp)) < 1e-9:
        return f"{int(round(pp))}pp"
    return f"{str(pp).replace('.', 'p')}pp"


def tune_policy(
    outcomes_path: Path,
    *,
    detector_mode: str,
    detector_cost_ms: float,
    budget: float,
    annealing_iterations: int,
    quantile_points: int,
    holdout_fraction: float,
    split_strategy: str,
    random_seed: int,
) -> dict:
    """Anneal on one scene's validation split; return frozen layout+thresholds."""
    payload = load_empirical_outcomes(outcomes_path)
    validation_payload, holdout_payload, split_meta = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=random_seed,
    )
    val_opt = HierarchyOptimizer(
        validation_payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    cascade = val_opt.synthesize()
    val_eval = FixedLayoutThresholdEvaluator(val_opt, cascade)
    hold_opt = HierarchyOptimizer(
        holdout_payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    hold_eval = FixedLayoutThresholdEvaluator(hold_opt, cascade)

    collection_val = val_eval.evaluate()
    baseline = float(collection_val["accuracy"])
    floor = max(0.0, baseline - float(budget))
    annealed = optimize_fixed_layout_thresholds_simulated_annealing(
        val_eval,
        floor,
        quantile_points=quantile_points,
        n_iterations=annealing_iterations,
        random_seed=random_seed,
        constraint_metric="micro",
    )
    thresholds = {str(k): float(v) for k, v in annealed["thresholds"].items()}
    opt_hold = hold_eval.evaluate(thresholds)
    return {
        "layout": cascade_to_dict(cascade),
        "thresholds": thresholds,
        "budget": float(budget),
        "floor": float(floor),
        "baseline_micro_validation": baseline,
        "tune_split": split_meta,
        "tune_holdout": {
            "accuracy": float(opt_hold["accuracy"]),
            "expected_cost": float(opt_hold["expected_cost"]),
            "macro_accuracy": float(opt_hold["macro_accuracy"]),
            "worst_class_accuracy": float(opt_hold["worst_class_accuracy"]),
        },
        "anneal_feasible": bool(annealed.get("feasible")),
    }


def evaluate_policy_on_scene(
    outcomes_path: Path,
    policy: dict,
    *,
    detector_mode: str,
    detector_cost_ms: float,
) -> dict:
    """Replay a frozen (layout, thresholds) on an entire scene (selection/test).

    We use the full scene table here (not a fresh split) so selection and test
    sets are large and stable.  Tuning already held out data inside the tune
    scene; selection/test scenes are never used for annealing.
    """
    payload = load_empirical_outcomes(outcomes_path)
    opt = HierarchyOptimizer(
        payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    cascade = cascade_from_layout(policy["layout"])
    evaluator = FixedLayoutThresholdEvaluator(opt, cascade)
    # Collection baseline on this scene with the *transferred* layout
    # (not this scene's own DP) — fair "same wiring" comparison.
    collection = evaluator.evaluate()
    # Only keep threshold keys that exist on this layout.
    usable = {
        k: float(v)
        for k, v in policy["thresholds"].items()
        if k in evaluator.tunable_ids
    }
    # Fill any missing tunable ids with this scene's defaults.
    for kid in evaluator.tunable_ids:
        usable.setdefault(kid, float(evaluator.default_thresholds[kid]))
    optimized = evaluator.evaluate(usable)
    return {
        "collection_accuracy": float(collection["accuracy"]),
        "collection_expected_cost": float(collection["expected_cost"]),
        "accuracy": float(optimized["accuracy"]),
        "expected_cost": float(optimized["expected_cost"]),
        "macro_accuracy": float(optimized["macro_accuracy"]),
        "worst_class_accuracy": float(optimized["worst_class_accuracy"]),
        "feasible_vs_scene_collection": bool(
            float(optimized["accuracy"]) >= float(collection["accuracy"]) - 1e-12
        ),
        "speedup_vs_collection_same_layout": (
            float(collection["expected_cost"]) / float(optimized["expected_cost"])
            if float(optimized["expected_cost"]) > 0
            else None
        ),
        "thresholds_used": usable,
    }


def select_best_policy(
    candidates: list[dict],
    select_scores: list[dict],
) -> int:
    """Pick candidate index: prefer feasible-on-select, then higher acc, then lower cost.

    What if you only maximized accuracy?  You might pick a very expensive
    near-collection policy.  What if you only minimized cost?  You might
    destroy accuracy on the select scene.  This lexicographic rule balances both.
    """
    best_i = 0
    best_key = None
    for i, score in enumerate(select_scores):
        key = (
            0 if score["feasible_vs_scene_collection"] else 1,
            -float(score["accuracy"]),
            float(score["expected_cost"]),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_i = i
    return best_i


def run_detector_mode(
    *,
    scenes: list[str],
    outcomes_dir: Path,
    detector_mode: str,
    detector_cost_ms: float,
    budgets: tuple[float, ...],
    annealing_iterations: int,
    quantile_points: int,
    holdout_fraction: float,
    split_strategy: str,
    random_seed: int,
) -> dict:
    # --- tune all source policies ---
    candidates: list[dict] = []
    for s_i, tune_scene in enumerate(scenes):
        path = outcome_path_for_scene(outcomes_dir, tune_scene)
        if not path.is_file():
            print(f"  skip tune {tune_scene}: missing {path}")
            continue
        for b_i, budget in enumerate(budgets):
            tag = f"{tune_scene}_{_budget_tag(budget)}"
            print(f"  tune [{tag}] ...", flush=True)
            policy = tune_policy(
                path,
                detector_mode=detector_mode,
                detector_cost_ms=detector_cost_ms,
                budget=budget,
                annealing_iterations=annealing_iterations,
                quantile_points=quantile_points,
                holdout_fraction=holdout_fraction,
                split_strategy=split_strategy,
                random_seed=random_seed + 17 * s_i + 3 * b_i,
            )
            policy["candidate_id"] = tag
            policy["tune_scene"] = tune_scene
            candidates.append(policy)
            th = policy["tune_holdout"]
            print(
                f"    tune-holdout acc={th['accuracy']:.4f}  "
                f"cost={th['expected_cost']:.2f}ms  "
                f"layout={policy['layout']['initial']}",
                flush=True,
            )

    # Cache full-scene evaluations: candidate × scene
    eval_cache: dict[str, dict[str, dict]] = {}
    for cand in candidates:
        cid = cand["candidate_id"]
        eval_cache[cid] = {}
        for scene in scenes:
            path = outcome_path_for_scene(outcomes_dir, scene)
            if not path.is_file():
                continue
            eval_cache[cid][scene] = evaluate_policy_on_scene(
                path,
                cand,
                detector_mode=detector_mode,
                detector_cost_ms=detector_cost_ms,
            )

    # --- for every (select, test) with select != test, pick best tune source ---
    triples: list[dict] = []
    for select_scene, test_scene in itertools.permutations(scenes, 2):
        # Eligible candidates: not tuned on select or test (avoid leakage).
        # If that empties the pool, fall back to "not tuned on test" only.
        strict = [
            c
            for c in candidates
            if c["tune_scene"] not in {select_scene, test_scene}
        ]
        pool = strict if strict else [
            c for c in candidates if c["tune_scene"] != test_scene
        ]
        if not pool:
            continue
        select_scores = [eval_cache[c["candidate_id"]][select_scene] for c in pool]
        best_i = select_best_policy(pool, select_scores)
        best = pool[best_i]
        select_metrics = select_scores[best_i]
        test_metrics = eval_cache[best["candidate_id"]][test_scene]

        # Baselines on test: (1) test scene's own tuned budget_0pp if present
        # (2) h24 budget_0pp zero-shot if present
        own = next(
            (
                eval_cache[c["candidate_id"]][test_scene]
                for c in candidates
                if c["tune_scene"] == test_scene and abs(c["budget"]) < 1e-15
            ),
            None,
        )
        h24_zs = next(
            (
                eval_cache[c["candidate_id"]][test_scene]
                for c in candidates
                if c["tune_scene"] == "h24" and abs(c["budget"]) < 1e-15
            ),
            None,
        )
        triples.append(
            {
                "select_scene": select_scene,
                "test_scene": test_scene,
                "selected_candidate": best["candidate_id"],
                "selected_tune_scene": best["tune_scene"],
                "selected_budget": best["budget"],
                "pool_size": len(pool),
                "pool_strict_no_select_test": bool(strict),
                "select_metrics": select_metrics,
                "test_metrics": test_metrics,
                "oracle_test_own_0pp": own,
                "h24_zero_shot_on_test": h24_zs,
                "delta_acc_vs_oracle": (
                    float(test_metrics["accuracy"]) - float(own["accuracy"])
                    if own
                    else None
                ),
                "delta_cost_vs_oracle": (
                    float(test_metrics["expected_cost"]) - float(own["expected_cost"])
                    if own
                    else None
                ),
                "delta_acc_vs_h24_zs": (
                    float(test_metrics["accuracy"]) - float(h24_zs["accuracy"])
                    if h24_zs
                    else None
                ),
                "delta_cost_vs_h24_zs": (
                    float(test_metrics["expected_cost"]) - float(h24_zs["expected_cost"])
                    if h24_zs
                    else None
                ),
            }
        )

    return {
        "detector_mode": detector_mode,
        "scenes": scenes,
        "budgets": list(budgets),
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "tune_scene": c["tune_scene"],
                "budget": c["budget"],
                "layout": c["layout"],
                "thresholds": c["thresholds"],
                "tune_holdout": c["tune_holdout"],
            }
            for c in candidates
        ],
        "evaluations": eval_cache,
        "triples": triples,
    }


def write_comparison_md(summary: dict, output_dir: Path) -> Path:
    rows: list[dict] = []
    for mode_key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        for t in report["triples"]:
            rows.append(
                {
                    "detector_mode": report["detector_mode"],
                    "select": t["select_scene"],
                    "test": t["test_scene"],
                    "chosen": t["selected_candidate"],
                    "test_acc": t["test_metrics"]["accuracy"],
                    "test_cost": t["test_metrics"]["expected_cost"],
                    "d_acc_vs_oracle": t.get("delta_acc_vs_oracle"),
                    "d_cost_vs_oracle": t.get("delta_cost_vs_oracle"),
                    "d_acc_vs_h24zs": t.get("delta_acc_vs_h24_zs"),
                    "d_cost_vs_h24zs": t.get("delta_cost_vs_h24_zs"),
                }
            )

    md = [
        "# Cross-Scene Tune → Select → Test — Comparison",
        "",
        "Question: can we tune on scene A, **select** on scene B, and still do "
        "well on scene C — vs oracle (tune on C) and vs h24 zero-shot?",
        "",
        "Candidates per tune scene: `0pp` (protect baseline) and `2pp` (speed budget). "
        "Selection prefers feasible-on-B, then higher accuracy, then lower cost. "
        "Strict pool = not tuned on B or C.",
        "",
        "| detector | select B | test C | chosen (from A) | test acc | test cost | "
        "Δacc vs oracle-C | Δcost vs oracle-C | Δacc vs h24→C | Δcost vs h24→C |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            "| {det} | {sel} | {test} | `{ch}` | {acc} | {cost} | {da} | {dc} | "
            "{dah} | {dch} |".format(
                det=row["detector_mode"],
                sel=row["select"],
                test=row["test"],
                ch=row["chosen"],
                acc=_fmt(row["test_acc"]),
                cost=_fmt(row["test_cost"]),
                da=_fmt(row["d_acc_vs_oracle"]),
                dc=_fmt(row["d_cost_vs_oracle"]),
                dah=_fmt(row["d_acc_vs_h24zs"]),
                dch=_fmt(row["d_cost_vs_h24zs"]),
            )
        )

    # Verdict aggregates
    md.extend(["", "## Verdict", ""])
    for mode in ("paper", "trained"):
        mode_rows = [r for r in rows if r["detector_mode"] == mode]
        if not mode_rows:
            continue
        beat_h24 = sum(
            1
            for r in mode_rows
            if r["d_acc_vs_h24zs"] is not None and float(r["d_acc_vs_h24zs"]) > 1e-9
        )
        within_2pp = sum(
            1
            for r in mode_rows
            if r["d_acc_vs_oracle"] is not None and float(r["d_acc_vs_oracle"]) >= -0.02
        )
        md.append(
            f"**{mode}:** {beat_h24}/{len(mode_rows)} select→test triples beat "
            f"h24 zero-shot accuracy on C; {within_2pp}/{len(mode_rows)} stay "
            f"within 2pp of oracle-C accuracy."
        )
    md.append("")
    md.append(
        "Δacc vs oracle < 0 means transferred+selected policy is less accurate "
        "than tuning on the test scene itself. Large gaps ⇒ you still need "
        "per-scene thresholds (or training) for that scene."
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
    written: list[Path] = []

    for mode in ("paper", "trained"):
        report = summary.get("runs", {}).get(mode)
        if not report or report.get("status") != "ok":
            continue
        triples = report["triples"]
        if not triples:
            continue

        # Heatmap: test acc of selected policy for (select, test)
        scenes = list(report["scenes"])
        idx = {s: i for i, s in enumerate(scenes)}
        mat = np.full((len(scenes), len(scenes)), np.nan)
        for t in triples:
            mat[idx[t["select_scene"]], idx[t["test_scene"]]] = float(
                t["test_metrics"]["accuracy"]
            )
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(mat, cmap="YlOrBr", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(scenes)))
        ax.set_yticks(range(len(scenes)))
        ax.set_xticklabels(scenes)
        ax.set_yticklabels(scenes)
        ax.set_xlabel("Test scene C")
        ax.set_ylabel("Select scene B")
        ax.set_title(f"Selected-policy test accuracy ({mode})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        path = figures_dir / f"fig_cross_scene_acc_heatmap_{mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

        # Scatter: Δacc vs h24 zero-shot
        fig, ax = plt.subplots(figsize=(5.8, 4.0))
        xs, ys = [], []
        for t in triples:
            if t.get("delta_acc_vs_h24_zs") is None:
                continue
            xs.append(float(t["delta_cost_vs_h24_zs"] or 0.0))
            ys.append(float(t["delta_acc_vs_h24_zs"]))
        if xs:
            ax.scatter(xs, ys, c="#2F5D50", s=36, alpha=0.85)
            ax.axhline(0.0, color="#6B7280", linewidth=0.8)
            ax.axvline(0.0, color="#6B7280", linewidth=0.8)
            ax.set_xlabel("Δ cost vs h24 zero-shot on C (ms)")
            ax.set_ylabel("Δ accuracy vs h24 zero-shot on C")
            ax.set_title(f"Select-on-B vs h24→C ({mode})")
            fig.tight_layout()
            path = figures_dir / f"fig_cross_scene_vs_h24zs_{mode}.png"
            fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)
            written.append(path)
        else:
            plt.close(fig)

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
        help="Accuracy shortfalls for tune candidates (e.g. 0 0.02).",
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
    scenes = [s for s in ALL_SCENES if s in args.scenes]
    scenes += [s for s in args.scenes if s not in scenes]

    summary: dict[str, Any] = {
        "experiment": "cross_scene_tune_select_test",
        "question": (
            "Tune on A, select on B, test on C — does selection beat h24 "
            "zero-shot and approach oracle per-scene thresholds?"
        ),
        "budgets": list(budgets),
        "annealing_iterations": args.iterations,
        "scenes": scenes,
        "runs": {},
    }

    for detector_mode in args.detector_modes:
        print(f"\n======== {detector_mode} ========", flush=True)
        try:
            report = run_detector_mode(
                scenes=scenes,
                outcomes_dir=args.outcomes_dir,
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
            path = args.output_dir / f"{detector_mode}.json"
            path.write_text(
                json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
            )
            report["report_path"] = str(path)
            summary["runs"][detector_mode] = report
            print(f"Wrote {path}")
        except Exception as error:
            summary["runs"][detector_mode] = {
                "status": "failed",
                "detector_mode": detector_mode,
                "error": str(error),
                "traceback": format_exc(),
            }
            print(f"FAILED {detector_mode}: {error}")

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
