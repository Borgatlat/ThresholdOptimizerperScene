"""Alternating structure ↔ thresholds experiment.

Research question
-----------------
One-shot (current pipeline) does:
    1) DP-synthesize a cascade from *collection-time* accept masks
    2) anneal thresholds for that *fixed* cascade

But after thresholds change, which samples each Ki "accepts" changes too.
The DP's IDK / routing probabilities are built from those accept masks
(``HierarchyOptimizer.accepted``).  So the cascade that was optimal under
the old masks may no longer be optimal under the new ones.

This experiment asks: if we *re-synthesize* the cascade after each threshold
update (alternating for N rounds), do we get better holdout accuracy and/or
lower expected cost than one-shot?

Method (same holdout for every method)
--------------------------------------
For each (scene, detector_mode):
  1. Split empirical outcomes with ``blocked_per_run`` 80/20 (seed fixed).
  2. On the validation split only:
       for iter in 1..N:
         a) synthesize cascade with HierarchyOptimizer (current accept masks)
         b) anneal thresholds for that fixed cascade
            target = baseline validation accuracy of the *first* cascade
            under collection thresholds (same target one-shot uses)
         c) rebuild accept masks from cached confidences:
                accepted = confidence >= new_threshold
            (do NOT re-run neural nets)
  3. Freeze final (cascade, thresholds); evaluate on the shared holdout.

Baselines compared on that same holdout:
  - one-shot  (N=1)
  - alternating N=2
  - alternating N=3

Why accept masks must be rebuilt (read this twice)
--------------------------------------------------
``HierarchyOptimizer.__init__`` copies the outcomes table's ``accepted``
column into ``self.accepted``.  ``synthesize()`` never looks at the
numeric threshold — it only sees booleans.  If we anneal new thresholds
but leave ``accepted`` at the collection-time values, the next DP call
would still believe the *old* accept/reject pattern and could not react
to the new policy.  Rebuilding ``accepted = confidence >= t`` from the
cached confidence column is exactly how the collector defined acceptance
in the first place (see ``empirical_outcomes._run_one_classifier``), so
we stay consistent without re-inference.

Outputs
-------
* ``checkpoints/threshold_experiments/alternating/``  — per-run JSON + summary
* ``checkpoints/figures/threshold_experiments/``      — paper PNGs
* ``COMPARISON.md``                                   — short table

Usage
-----
    python experiment_alternating_structure_thresholds.py
    python experiment_alternating_structure_thresholds.py --scenes h24 --detector-modes paper
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from traceback import format_exc
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from empirical_outcomes import load_empirical_outcomes
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    FixedLayoutThresholdEvaluator,
    optimize_fixed_layout_thresholds_simulated_annealing,
    split_empirical_outcomes,
)


# i22 intentionally omitted: no single-vehicle empirical outcomes yet.
ALL_SCENES = ("h24", "h08", "s31", "a06", "i29")
DEFAULT_OUTCOMES_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/alternating")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")
# Methods to report. one-shot == alternating with N=1 (explicit name for the table).
METHODS = ("one_shot", "alternating_n2", "alternating_n3")
METHOD_TO_N = {"one_shot": 1, "alternating_n2": 2, "alternating_n3": 3}


def outcome_path_for_scene(outcomes_dir: Path, scene: str) -> Path:
    if scene == "h24":
        return outcomes_dir / "empirical_outcomes.pkl"
    return outcomes_dir / f"empirical_outcomes_{scene}.pkl"


def clone_payload(payload: dict) -> dict:
    """Deep-copy the four payload pieces we mutate (accept masks / thresholds).

    Why not ``copy.deepcopy(payload)`` alone?  It works, but being explicit
    about the four keys documents which fields the alternating loop touches.
    If you added a fifth key and forgot to clone it, a shallow bug would let
    two methods silently share state across the same holdout split.
    """
    return {
        "labels": payload["labels"].copy(deep=True),
        "candidates": payload["candidates"].copy(deep=True),
        "outcomes": payload["outcomes"].copy(deep=True),
        "detector": deepcopy(payload["detector"]),
    }


def rebuild_accept_masks_from_thresholds(
    payload: dict,
    thresholds: dict[str, float],
) -> None:
    """Update ``outcomes.accepted`` after thresholds change (in-place).

    Syntax note for beginners
    -------------------------
    ``payload["outcomes"]`` is a long pandas DataFrame with one row per
    (sample, classifier).  We loop over each classifier id whose threshold
    we just optimized.  For that classifier's rows we set:

        accepted = (confidence >= new_threshold)

    ``confidence`` is already in the table from the original neural-net
    forward pass.  Changing the comparison threshold does NOT require
    re-running the network — confidence is a cached float.

    What happens if you skip this rebuild?
    --------------------------------------
    The next ``HierarchyOptimizer(payload).synthesize()`` would still read
    the old boolean ``accepted`` column.  The DP would then pick a cascade
    that is "optimal" for the *previous* threshold policy, defeating the
    whole point of alternating.  Threshold annealing (via
    ``FixedLayoutThresholdEvaluator``) already recomputes acceptance on the
    fly from confidences; only the DP side needs this explicit rebuild
    because it consumes the DataFrame column, not the threshold number.
    """
    outcomes = payload["outcomes"]
    candidates = payload["candidates"]

    for candidate_id, threshold in thresholds.items():
        # Boolean mask selecting every row belonging to this Ki.
        # Changing ``==`` to ``!=`` would update the wrong classifiers.
        row_mask = outcomes["candidate_id"] == candidate_id
        if not bool(row_mask.any()):
            continue
        # Vectorized comparison: one bool per sample for this Ki.
        outcomes.loc[row_mask, "accepted"] = (
            outcomes.loc[row_mask, "confidence"].to_numpy(dtype=float) >= float(threshold)
        )
        # Keep the candidates table consistent so default_thresholds / describe()
        # reflect the policy we just chose.  If you leave this stale, the next
        # FixedLayoutThresholdEvaluator would start annealing from the old
        # default even though accepted masks already moved.
        cand_mask = candidates["id"] == candidate_id
        if bool(cand_mask.any()):
            candidates.loc[cand_mask, "threshold"] = float(threshold)


def cascade_to_dict(cascade: Cascade) -> dict[str, Any]:
    """JSON-friendly layout snapshot (tuples → 'router:group' strings)."""
    return {
        "expected_cost_dp": float(cascade.expected_cost),
        "initial": list(cascade.initial),
        "specialized": {
            f"{router_id}:{group}": list(chain)
            for (router_id, group), chain in cascade.specialized.items()
        },
        "detector": cascade.detector,
    }


def _speedup(baseline_cost: float | None, opt_cost: float | None) -> float | None:
    if baseline_cost is None or opt_cost is None:
        return None
    if float(opt_cost) <= 0.0:
        return None
    return float(baseline_cost) / float(opt_cost)


def run_alternating(
    validation_payload: dict,
    holdout_payload: dict,
    *,
    n_rounds: int,
    detector_mode: str,
    detector_cost_ms: float,
    annealing_iterations: int,
    quantile_points: int,
    random_seed: int,
) -> dict:
    """Run N rounds of (synthesize → anneal → rebuild masks) on validation.

    Holdout is touched only at the end (and for the collection-threshold
    baseline of the final cascade), so topology/threshold search never peeks.
    """
    if n_rounds < 1:
        raise ValueError("n_rounds must be >= 1")

    # Work on a private copy so one_shot / N=2 / N=3 do not pollute each other.
    working = clone_payload(validation_payload)

    iteration_log: list[dict] = []
    target_accuracy: float | None = None
    final_cascade: Cascade | None = None
    final_thresholds: dict[str, float] | None = None
    final_validation_metrics: dict | None = None

    for round_idx in range(1, n_rounds + 1):
        # --- (a) synthesize cascade from CURRENT accept masks -------------
        # HierarchyOptimizer reads working["outcomes"]["accepted"] here.
        optimizer = HierarchyOptimizer(
            working,
            detector_mode=detector_mode,
            detector_cost_ms=detector_cost_ms,
        )
        cascade = optimizer.synthesize()
        evaluator = FixedLayoutThresholdEvaluator(optimizer, cascade)

        # Collection / current-default thresholds for THIS cascade layout.
        # On round 1 these are the registry thresholds baked into the pkl.
        # On later rounds they are the previous anneal's thresholds (because
        # we wrote them back into candidates + accepted).
        round_baseline = evaluator.evaluate()

        # Freeze the accuracy target from round 1 so every method (one-shot
        # and alternating) optimizes toward the SAME accuracy floor.
        # What if you recomputed the target every round?  Then a cascade that
        # is more accurate under its defaults would demand a higher bar, and
        # one-shot vs alternating would no longer be comparable.
        if target_accuracy is None:
            target_accuracy = float(round_baseline["accuracy"])

        # --- (b) anneal thresholds for this fixed cascade -----------------
        # Seed shifts by round so later rounds are not an exact replay of
        # round 1's random walk (same search budget, independent exploration).
        annealed = optimize_fixed_layout_thresholds_simulated_annealing(
            evaluator,
            float(target_accuracy),
            quantile_points=quantile_points,
            n_iterations=annealing_iterations,
            random_seed=random_seed + round_idx,
        )
        thresholds = {
            str(k): float(v) for k, v in annealed["thresholds"].items()
        }

        iteration_log.append(
            {
                "round": round_idx,
                "layout": cascade_to_dict(cascade),
                "baseline_validation_accuracy": float(round_baseline["accuracy"]),
                "baseline_validation_cost_ms": float(round_baseline["expected_cost"]),
                "annealed_validation_accuracy": float(annealed["accuracy"]),
                "annealed_validation_cost_ms": float(annealed["expected_cost"]),
                "annealed_feasible": bool(annealed["feasible"]),
                "thresholds": thresholds,
            }
        )

        final_cascade = cascade
        final_thresholds = thresholds
        final_validation_metrics = annealed

        # --- (c) rebuild accept masks from cached confidences -------------
        # Skip on the last round: we freeze (cascade, thresholds) and evaluate
        # holdout next; no further DP call will read these masks.
        if round_idx < n_rounds:
            rebuild_accept_masks_from_thresholds(working, thresholds)

    assert final_cascade is not None
    assert final_thresholds is not None
    assert final_validation_metrics is not None
    assert target_accuracy is not None

    # Holdout evaluation uses FixedLayoutThresholdEvaluator, which derives
    # accepted = confidence >= threshold itself — so we do NOT need to mutate
    # the holdout payload's accepted column for scoring.
    holdout_optimizer = HierarchyOptimizer(
        holdout_payload,
        detector_mode=detector_mode,
        detector_cost_ms=detector_cost_ms,
    )
    holdout_evaluator = FixedLayoutThresholdEvaluator(
        holdout_optimizer,
        final_cascade,
    )
    # Baseline = same final layout, but collection / default thresholds.
    # (Holdout evaluator's default_thresholds come from holdout candidates,
    # which still hold the original registry values — correct for "vs baseline
    # thresholds" speedup.)
    baseline_holdout = holdout_evaluator.evaluate()
    optimized_holdout = holdout_evaluator.evaluate(final_thresholds)

    return {
        "n_rounds": n_rounds,
        "target_accuracy": float(target_accuracy),
        "target_accuracy_source": "round1_baseline_validation",
        "iterations": iteration_log,
        "final_layout": cascade_to_dict(final_cascade),
        "final_thresholds": final_thresholds,
        "validation": {
            "accuracy": float(final_validation_metrics["accuracy"]),
            "expected_cost": float(final_validation_metrics["expected_cost"]),
            "feasible": bool(final_validation_metrics["feasible"]),
        },
        "baseline_holdout": {
            "accuracy": float(baseline_holdout["accuracy"]),
            "expected_cost": float(baseline_holdout["expected_cost"]),
        },
        "optimized_holdout": {
            "accuracy": float(optimized_holdout["accuracy"]),
            "expected_cost": float(optimized_holdout["expected_cost"]),
            "macro_accuracy": float(optimized_holdout.get("macro_accuracy", float("nan"))),
            "worst_class_accuracy": float(
                optimized_holdout.get("worst_class_accuracy", float("nan"))
            ),
            "route_counts": optimized_holdout.get("route_counts"),
        },
        "holdout_feasible": bool(
            float(optimized_holdout["accuracy"]) >= float(target_accuracy)
        ),
        "holdout_speedup_vs_baseline_thresholds": _speedup(
            float(baseline_holdout["expected_cost"]),
            float(optimized_holdout["expected_cost"]),
        ),
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
    methods: tuple[str, ...] = METHODS,
) -> dict:
    """One scene × detector_mode: shared split, then every method on that split."""
    payload = load_empirical_outcomes(outcomes_path)
    validation_payload, holdout_payload, split_meta = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=random_seed,
    )

    report: dict[str, Any] = {
        "scene": scene,
        "detector_mode": detector_mode,
        "outcomes_path": str(outcomes_path),
        "split": split_meta,
        "methods": {},
    }

    for method_name in methods:
        n_rounds = METHOD_TO_N[method_name]
        print(
            f"  [{scene}/{detector_mode}] {method_name} (N={n_rounds}) ...",
            flush=True,
        )
        result = run_alternating(
            validation_payload,
            holdout_payload,
            n_rounds=n_rounds,
            detector_mode=detector_mode,
            detector_cost_ms=detector_cost_ms,
            annealing_iterations=annealing_iterations,
            quantile_points=quantile_points,
            random_seed=random_seed,
        )
        report["methods"][method_name] = result
        opt = result["optimized_holdout"]
        print(
            f"    holdout acc={opt['accuracy']:.4f}  "
            f"cost={opt['expected_cost']:.2f}ms  "
            f"speedup={result['holdout_speedup_vs_baseline_thresholds'] or float('nan'):.3f}x  "
            f"feasible={result['holdout_feasible']}  "
            f"layout={result['final_layout']['initial']}",
            flush=True,
        )

    return report


def write_comparison_md(summary: dict, output_dir: Path) -> Path:
    """Short paper-facing table: does alternating beat one-shot?"""
    rows: list[dict] = []
    for key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        scene = report["scene"]
        detector_mode = report["detector_mode"]
        methods = report.get("methods", {})
        one = methods.get("one_shot", {})
        for method_name, block in methods.items():
            opt = block.get("optimized_holdout", {})
            one_opt = one.get("optimized_holdout", {})
            cost_delta = None
            acc_delta = None
            if one_opt and method_name != "one_shot":
                # Negative cost_delta => alternating is cheaper (good).
                cost_delta = float(opt["expected_cost"]) - float(one_opt["expected_cost"])
                acc_delta = float(opt["accuracy"]) - float(one_opt["accuracy"])
            rows.append(
                {
                    "scene": scene,
                    "detector_mode": detector_mode,
                    "method": method_name,
                    "holdout_acc": opt.get("accuracy"),
                    "holdout_cost_ms": opt.get("expected_cost"),
                    "speedup_vs_baseline_thresholds": block.get(
                        "holdout_speedup_vs_baseline_thresholds"
                    ),
                    "feasible": block.get("holdout_feasible"),
                    "layout": block.get("final_layout", {}).get("initial"),
                    "acc_delta_vs_one_shot": acc_delta,
                    "cost_delta_vs_one_shot_ms": cost_delta,
                    "thresholds": block.get("final_thresholds"),
                }
            )

    md = [
        "# Alternating Structure ↔ Thresholds — Comparison",
        "",
        "Question: after thresholds change, does re-synthesizing the cascade "
        "(alternating) beat one-shot (DP once → thresholds once)?",
        "",
        "All methods for a given (scene, detector_mode) share the **same** "
        "`blocked_per_run` holdout split. Negative result is reported if "
        "alternating does not improve.",
        "",
        "| scene | detector | method | holdout acc | cost (ms) | speedup | feasible | "
        "Δacc vs one-shot | Δcost vs one-shot (ms) | layout |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        layout = "→".join(row["layout"] or [])
        md.append(
            "| {scene} | {det} | {method} | {acc} | {cost} | {spd} | {feas} | "
            "{dacc} | {dcost} | `{layout}` |".format(
                scene=row["scene"],
                det=row["detector_mode"],
                method=row["method"],
                acc=_fmt(row["holdout_acc"]),
                cost=_fmt(row["holdout_cost_ms"]),
                spd=_fmt(row["speedup_vs_baseline_thresholds"]),
                feas=row["feasible"],
                dacc=_fmt(row["acc_delta_vs_one_shot"]),
                dcost=_fmt(row["cost_delta_vs_one_shot_ms"]),
                layout=layout,
            )
        )

    # Verdict paragraph: count wins.
    wins_cost = 0
    wins_acc = 0
    compared = 0
    for row in rows:
        if row["method"] == "one_shot":
            continue
        if row["cost_delta_vs_one_shot_ms"] is None:
            continue
        compared += 1
        if float(row["cost_delta_vs_one_shot_ms"]) < -1e-9:
            wins_cost += 1
        if row["acc_delta_vs_one_shot"] is not None and float(row["acc_delta_vs_one_shot"]) > 1e-9:
            wins_acc += 1

    md.extend(
        [
            "",
            "## Verdict",
            "",
            f"Among {compared} alternating runs (N=2 and N=3, both detector modes / scenes):",
            f"- **{wins_cost}** lowered holdout expected cost vs one-shot",
            f"- **{wins_acc}** raised holdout accuracy vs one-shot",
            "",
            "Δcost < 0 means alternating is cheaper; Δacc > 0 means more accurate.",
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


def plot_alternating_figures(summary: dict, figures_dir: Path) -> list[Path]:
    """Paper-ready accuracy / cost comparisons for alternating vs one-shot."""
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
    c_one = "#4C6A80"
    c_n2 = "#C45C26"
    c_n3 = "#2F5D50"
    method_colors = {
        "one_shot": c_one,
        "alternating_n2": c_n2,
        "alternating_n3": c_n3,
    }
    method_labels = {
        "one_shot": "One-shot",
        "alternating_n2": "Alt. N=2",
        "alternating_n3": "Alt. N=3",
    }

    written: list[Path] = []

    for detector_mode in ("paper", "trained"):
        scenes: list[str] = []
        series: dict[str, dict[str, list[float]]] = {
            m: {"acc": [], "cost": [], "speedup": []} for m in METHODS
        }
        for scene in ALL_SCENES:
            key = f"{scene}__{detector_mode}"
            report = summary.get("runs", {}).get(key)
            if not report or report.get("status") != "ok":
                continue
            scenes.append(scene)
            for method_name in METHODS:
                block = report["methods"][method_name]
                opt = block["optimized_holdout"]
                series[method_name]["acc"].append(float(opt["accuracy"]))
                series[method_name]["cost"].append(float(opt["expected_cost"]))
                series[method_name]["speedup"].append(
                    float(block["holdout_speedup_vs_baseline_thresholds"] or 0.0)
                )

        if not scenes:
            continue

        x = np.arange(len(scenes))
        width = 0.25

        # Figure A: holdout expected cost grouped bars
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for i, method_name in enumerate(METHODS):
            ax.bar(
                x + (i - 1) * width,
                series[method_name]["cost"],
                width=width,
                color=method_colors[method_name],
                label=method_labels[method_name],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.set_ylabel("Holdout expected cost (ms)")
        ax.set_xlabel("Scene")
        ax.set_title(f"Alternating vs one-shot — cost ({detector_mode} Kdet)")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = figures_dir / f"fig_alternating_cost_{detector_mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

        # Figure B: holdout accuracy grouped bars
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for i, method_name in enumerate(METHODS):
            ax.bar(
                x + (i - 1) * width,
                series[method_name]["acc"],
                width=width,
                color=method_colors[method_name],
                label=method_labels[method_name],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.set_ylabel("Holdout accuracy")
        ax.set_xlabel("Scene")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Alternating vs one-shot — accuracy ({detector_mode} Kdet)")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = figures_dir / f"fig_alternating_acc_{detector_mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

        # Figure C: Δcost vs one-shot (negative = alternating cheaper)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for i, method_name in enumerate(("alternating_n2", "alternating_n3")):
            deltas = []
            for j, scene in enumerate(scenes):
                key = f"{scene}__{detector_mode}"
                methods = summary["runs"][key]["methods"]
                one_cost = float(methods["one_shot"]["optimized_holdout"]["expected_cost"])
                alt_cost = float(methods[method_name]["optimized_holdout"]["expected_cost"])
                deltas.append(alt_cost - one_cost)
            ax.bar(
                x + (i - 0.5) * width,
                deltas,
                width=width,
                color=method_colors[method_name],
                label=method_labels[method_name],
                edgecolor="white",
                linewidth=0.4,
            )
        ax.axhline(0.0, color="#6B7280", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.set_ylabel("Δ holdout cost vs one-shot (ms)")
        ax.set_xlabel("Scene")
        ax.set_title(
            f"Cost change from alternating (neg. = cheaper) — {detector_mode}"
        )
        ax.legend(frameon=False)
        fig.tight_layout()
        path = figures_dir / f"fig_alternating_delta_cost_{detector_mode}.png"
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
    parser.add_argument(
        "--detector-cost-ms",
        type=float,
        default=PAPER_DETECTOR_COST_MS,
        help="Synthetic paper-Kdet cost (ignored for detector_mode=trained).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write JSON/MD only (useful for quick smoke tests).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "experiment": "alternating_structure_thresholds",
        "question": (
            "If we re-synthesize the cascade AFTER thresholds change, do we "
            "get better accuracy and/or lower expected cost than one-shot?"
        ),
        "methods": list(METHODS),
        "annealing_iterations": args.iterations,
        "quantile_points": args.quantile_points,
        "split_strategy": args.split_strategy,
        "holdout_fraction": args.holdout_fraction,
        "random_seed": args.seed,
        "runs": {},
    }

    # h24 first (as requested), then the remaining scenes in ALL_SCENES order.
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
        plot_alternating_figures(summary, args.figures_dir)


if __name__ == "__main__":
    main()
