"""Independent vs joint threshold calibration.

PROMPT USED (experiment 4 — README item 5)
------------------------------------------
Repo: ThresholdOptimizerperScene, branch cursor/threshold-optimizer-experiments-8590.

GOAL
Compare *independent* per-Ki threshold calibration (precision / P(IDK)
matching) against the current *joint* end-to-end anneal on a fixed cascade.
Question: does joint cascade-aware tuning beat calibrating each classifier
alone, on the same holdout?

CONTEXT (already done — do not redo)
- threshold_optimizer.py anneal + FixedLayoutThresholdEvaluator
- hierarchy_optimizer.py DP synthesize
- empirical_outcomes*.pkl cached; alternating + sequence-order experiments done
- DO NOT train per-scene classifiers; DO NOT scene-switch

METHOD
For scene=h24 first (then h08,s31,a06,i29 — not i22), detector_mode in
{paper, trained}:
  1) Split blocked_per_run 80/20 (shared across ALL methods).
  2) Synthesize DP cascade on validation only; freeze layout for everyone.
  3) Methods (same layout, same holdout):
       a) collection          — registry / collection thresholds
       b) indep_precision     — per Ki: lowest t s.t. precision(accepts) >=
                                paper level target (0.90 global, 0.95 other)
       c) indep_precision_match — per Ki: match collection-time precision
       d) indep_p_idk_match   — per Ki: match collection-time P(IDK)
       e) indep_p_idk_fixed   — per Ki: threshold for P(IDK)=0.10
       f) joint_anneal        — end-to-end SA (target=collection val accuracy)
  4) Report holdout acc, cost, speedup vs collection, feasibility, thresholds.

DELIVERABLES
- this script
- checkpoints/threshold_experiments/independent_vs_joint/
- COMPARISON.md + paper PNGs
- commit + push + update PR

ACCEPTANCE
- every method shares the same holdout + same DP layout
- independent methods never look at cascade cost during calibration
- negative result OK if independent ≈ joint

------------------------------------------------------------------------
Why "independent" vs "joint" matters (beginner intuition)
--------------------------------------------------------
Joint anneal asks: "what threshold *vector* makes the whole cascade fast
while staying accurate?"  Controllers interact — raising K0's threshold
changes which samples reach K3.

Independent calibration asks each Ki alone: "when *you* accept, how often
are you right?" or "how often do you IDK?"  That ignores routing.  If
independent matches joint, cascade interactions are weak and we can
calibrate cheaper.  If joint wins, the interactions matter.
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
from utils.labels import (
    GLOBAL_CLASS_NAMES,
    INTERMEDIATE_CLASS_NAMES,
    KI_REGISTRY,
    PAPER_THRESHOLD_HI_BY_LEVEL,
)


ALL_SCENES = ("h24", "h08", "s31", "a06", "i29")
DEFAULT_OUTCOMES_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/independent_vs_joint")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")
FIXED_P_IDK = 0.10

# Method ids → short labels for tables/figures.
METHODS = (
    "collection",
    "indep_precision",
    "indep_precision_match",
    "indep_p_idk_match",
    "indep_p_idk_fixed",
    "joint_anneal",
)
METHOD_LABELS = {
    "collection": "Collection H_i",
    "indep_precision": "Indep. precision@paper",
    "indep_precision_match": "Indep. precision@collect",
    "indep_p_idk_match": "Indep. P(IDK)@collect",
    "indep_p_idk_fixed": f"Indep. P(IDK)={FIXED_P_IDK:.0%}",
    "joint_anneal": "Joint anneal",
}


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


def _true_label_vector(payload: dict, candidate_id: str) -> np.ndarray:
    """Ground-truth vector in the same index space as ``prediction`` for Ki.

    Identifiers (K0/K1) predict INTERMEDIATE_CLASS_NAMES.
    Globals + specialists + detector predict GLOBAL_CLASS_NAMES.

    What if you compared K0 against global labels?  Precision would look
    artificially terrible (3-way vs 5-way mismatch) and independent
    calibration would push thresholds to almost-never-accept.
    """
    labels = payload["labels"].sort_values("sample_id")
    kind = str(payload["candidates"].set_index("id").loc[candidate_id, "kind"])
    if kind == "identifier":
        lookup = {name: i for i, name in enumerate(INTERMEDIATE_CLASS_NAMES)}
        return labels["true_intermediate_label"].map(lookup).to_numpy(dtype=int)
    lookup = {name: i for i, name in enumerate(GLOBAL_CLASS_NAMES)}
    return labels["true_global_label"].map(lookup).to_numpy(dtype=int)


def _eligible_mask(payload: dict, candidate_id: str) -> np.ndarray:
    """Samples used for independent stats.

    Specialists are only scored on their intermediate group — a coupe model
    accepting on an SUV sample is outside its job description.  If you scored
    specialists on *all* samples, independent precision targets would be
    dominated by out-of-group errors and thresholds would become tiny.
    """
    labels = payload["labels"].sort_values("sample_id")
    n = len(labels)
    kind = str(payload["candidates"].set_index("id").loc[candidate_id, "kind"])
    if kind != "specialized":
        return np.ones(n, dtype=bool)
    group = payload["candidates"].set_index("id").loc[candidate_id, "group"]
    return (labels["true_intermediate_label"].to_numpy() == group).to_numpy(dtype=bool)


def _confidence_and_pred(payload: dict, candidate_id: str) -> tuple[np.ndarray, np.ndarray]:
    rows = (
        payload["outcomes"][payload["outcomes"]["candidate_id"] == candidate_id]
        .sort_values("sample_id")
    )
    return (
        rows["confidence"].to_numpy(dtype=float),
        rows["prediction"].to_numpy(dtype=int),
    )


def precision_at_threshold(
    confidence: np.ndarray,
    prediction: np.ndarray,
    true_label: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
) -> tuple[float, float, int]:
    """Return (precision, p_idk, n_accept) on eligible samples.

    precision = P(correct | accept).  p_idk = P(reject) = 1 - P(accept).
    If nothing is accepted, precision is defined as 1.0 (vacuously safe) so
    a search that raises the threshold forever still looks "precise" — we
    break ties by preferring *lower* thresholds (more accepts) elsewhere.
    """
    elig_conf = confidence[eligible]
    elig_pred = prediction[eligible]
    elig_true = true_label[eligible]
    if len(elig_conf) == 0:
        return 1.0, 1.0, 0
    accepted = elig_conf >= threshold
    n_accept = int(accepted.sum())
    p_idk = 1.0 - (n_accept / len(elig_conf))
    if n_accept == 0:
        return 1.0, p_idk, 0
    precision = float(np.mean(elig_pred[accepted] == elig_true[accepted]))
    return precision, p_idk, n_accept


def calibrate_precision_threshold(
    confidence: np.ndarray,
    prediction: np.ndarray,
    true_label: np.ndarray,
    eligible: np.ndarray,
    target_precision: float,
) -> float:
    """Lowest threshold on the empirical confidence grid with precision >= target.

    Scan unique confidences from low → high (loose → strict) and take the
    *first* cut that hits the precision floor with at least one accept.
    That is the most permissive (fastest) independently-safe threshold.

    What if you took the *highest* such threshold instead?  You would over-
    defer (huge P(IDK), cascade always falls to Kdet) and look artificially
    precise while destroying speedup.
    """
    elig_conf = confidence[eligible]
    if len(elig_conf) == 0:
        return 1.0
    # Candidate cutpoints: every observed confidence, plus 0 and a value
    # just above max (reject all).
    grid = np.unique(
        np.concatenate(
            [
                np.array([0.0]),
                elig_conf,
                np.array([np.nextafter(float(elig_conf.max()), np.inf)]),
            ]
        )
    )
    for threshold in grid:
        prec, _, n_accept = precision_at_threshold(
            confidence, prediction, true_label, eligible, float(threshold)
        )
        if prec + 1e-12 >= target_precision and n_accept > 0:
            return float(threshold)
    # Impossible to hit target with any accepts — reject everything.
    return float(grid[-1])


def calibrate_p_idk_threshold(
    confidence: np.ndarray,
    eligible: np.ndarray,
    target_p_idk: float,
) -> float:
    """Threshold so that roughly ``target_p_idk`` fraction of eligible samples IDK.

    Syntax: ``np.quantile(confidence, target_p_idk)`` is the cut where the
    bottom ``target_p_idk`` fraction of confidences lie below it — those
    samples IDK.  Clip to [0, 1] for safety.

    What if target_p_idk=0?  Threshold → min confidence ⇒ accept everyone.
    What if target_p_idk=1?  Threshold → above max ⇒ accept nobody.
    """
    elig_conf = confidence[eligible]
    if len(elig_conf) == 0:
        return 1.0
    p = float(np.clip(target_p_idk, 0.0, 1.0))
    return float(np.quantile(elig_conf, p))


def collection_precision_and_p_idk(
    payload: dict,
    candidate_id: str,
) -> tuple[float, float, float]:
    """Stats under the payload's current (collection) threshold for one Ki."""
    candidates = payload["candidates"].set_index("id")
    threshold = float(candidates.loc[candidate_id, "threshold"])
    confidence, prediction = _confidence_and_pred(payload, candidate_id)
    true_label = _true_label_vector(payload, candidate_id)
    eligible = _eligible_mask(payload, candidate_id)
    prec, p_idk, _ = precision_at_threshold(
        confidence, prediction, true_label, eligible, threshold
    )
    return threshold, prec, p_idk


def paper_precision_target(candidate_id: str) -> float:
    """Paper-style precision floor from the Ki's hierarchy level."""
    level = KI_REGISTRY[candidate_id].level
    # PAPER_THRESHOLD_HI_BY_LEVEL stores H_i, not precision — but paper V-A
    # chose those H_i to keep cumulative error ≤ ~10%.  We use the same
    # numbers as a precision *target* for independent calibration:
    # global → 0.90, intermediate/specialized → 0.95.
    return float(PAPER_THRESHOLD_HI_BY_LEVEL[level])


def independent_thresholds(
    validation_payload: dict,
    tunable_ids: tuple[str, ...],
    mode: str,
) -> dict[str, float]:
    """Calibrate each tunable Ki alone (no cascade cost in the objective)."""
    out: dict[str, float] = {}
    for candidate_id in tunable_ids:
        confidence, prediction = _confidence_and_pred(validation_payload, candidate_id)
        true_label = _true_label_vector(validation_payload, candidate_id)
        eligible = _eligible_mask(validation_payload, candidate_id)
        coll_t, coll_prec, coll_p_idk = collection_precision_and_p_idk(
            validation_payload, candidate_id
        )

        if mode == "indep_precision":
            target = paper_precision_target(candidate_id)
            out[candidate_id] = calibrate_precision_threshold(
                confidence, prediction, true_label, eligible, target
            )
        elif mode == "indep_precision_match":
            out[candidate_id] = calibrate_precision_threshold(
                confidence, prediction, true_label, eligible, coll_prec
            )
        elif mode == "indep_p_idk_match":
            out[candidate_id] = calibrate_p_idk_threshold(
                confidence, eligible, coll_p_idk
            )
        elif mode == "indep_p_idk_fixed":
            out[candidate_id] = calibrate_p_idk_threshold(
                confidence, eligible, FIXED_P_IDK
            )
        else:
            raise ValueError(f"Unknown independent mode: {mode}")
    return out


def _speedup(baseline_cost: float | None, opt_cost: float | None) -> float | None:
    if baseline_cost is None or opt_cost is None or float(opt_cost) <= 0:
        return None
    return float(baseline_cost) / float(opt_cost)


def evaluate_threshold_vector(
    val_eval: FixedLayoutThresholdEvaluator,
    hold_eval: FixedLayoutThresholdEvaluator,
    thresholds: dict[str, float],
    target_accuracy: float,
) -> dict:
    val_metrics = val_eval.evaluate(thresholds)
    hold_metrics = hold_eval.evaluate(thresholds)
    return {
        "thresholds": {str(k): float(v) for k, v in thresholds.items()},
        "validation": {
            "accuracy": float(val_metrics["accuracy"]),
            "expected_cost": float(val_metrics["expected_cost"]),
        },
        "holdout": {
            "accuracy": float(hold_metrics["accuracy"]),
            "expected_cost": float(hold_metrics["expected_cost"]),
            "macro_accuracy": float(hold_metrics.get("macro_accuracy", float("nan"))),
            "worst_class_accuracy": float(
                hold_metrics.get("worst_class_accuracy", float("nan"))
            ),
            "route_counts": hold_metrics.get("route_counts"),
        },
        "holdout_feasible": bool(float(hold_metrics["accuracy"]) >= target_accuracy),
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
) -> dict:
    payload = load_empirical_outcomes(outcomes_path)
    validation_payload, holdout_payload, split_meta = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=random_seed,
    )

    # Freeze DP layout from validation only.
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
    tunable = val_eval.tunable_ids

    collection_thresholds = dict(val_eval.default_thresholds)
    collection_val = val_eval.evaluate()
    target_accuracy = float(collection_val["accuracy"])
    collection_hold = hold_eval.evaluate()

    methods_out: dict[str, Any] = {}

    # --- collection baseline -------------------------------------------------
    methods_out["collection"] = {
        **evaluate_threshold_vector(
            val_eval, hold_eval, collection_thresholds, target_accuracy
        ),
        "method": "collection",
        "holdout_speedup_vs_collection": 1.0,
    }

    # --- independent calibrations -------------------------------------------
    for mode in (
        "indep_precision",
        "indep_precision_match",
        "indep_p_idk_match",
        "indep_p_idk_fixed",
    ):
        print(f"  [{mode}] ...", flush=True)
        thresholds = independent_thresholds(validation_payload, tunable, mode)
        # Only keep keys the evaluator expects (tunable subset).
        thresholds = {k: float(thresholds[k]) for k in tunable}
        block = evaluate_threshold_vector(
            val_eval, hold_eval, thresholds, target_accuracy
        )
        block["method"] = mode
        block["holdout_speedup_vs_collection"] = _speedup(
            float(collection_hold["expected_cost"]),
            float(block["holdout"]["expected_cost"]),
        )
        methods_out[mode] = block
        print(
            f"    holdout acc={block['holdout']['accuracy']:.4f}  "
            f"cost={block['holdout']['expected_cost']:.2f}ms  "
            f"feasible={block['holdout_feasible']}",
            flush=True,
        )

    # --- joint end-to-end anneal ---------------------------------------------
    print("  [joint_anneal] ...", flush=True)
    annealed = optimize_fixed_layout_thresholds_simulated_annealing(
        val_eval,
        target_accuracy,
        quantile_points=quantile_points,
        n_iterations=annealing_iterations,
        random_seed=random_seed,
    )
    joint_thresholds = {str(k): float(v) for k, v in annealed["thresholds"].items()}
    joint_block = evaluate_threshold_vector(
        val_eval, hold_eval, joint_thresholds, target_accuracy
    )
    joint_block["method"] = "joint_anneal"
    joint_block["holdout_speedup_vs_collection"] = _speedup(
        float(collection_hold["expected_cost"]),
        float(joint_block["holdout"]["expected_cost"]),
    )
    joint_block["anneal_validation_feasible"] = bool(annealed["feasible"])
    methods_out["joint_anneal"] = joint_block
    print(
        f"    holdout acc={joint_block['holdout']['accuracy']:.4f}  "
        f"cost={joint_block['holdout']['expected_cost']:.2f}ms  "
        f"feasible={joint_block['holdout_feasible']}",
        flush=True,
    )

    # Deltas vs joint for the table.
    joint_cost = float(joint_block["holdout"]["expected_cost"])
    joint_acc = float(joint_block["holdout"]["accuracy"])
    for name, block in methods_out.items():
        block["delta_cost_vs_joint_ms"] = float(block["holdout"]["expected_cost"]) - joint_cost
        block["delta_acc_vs_joint"] = float(block["holdout"]["accuracy"]) - joint_acc

    return {
        "scene": scene,
        "detector_mode": detector_mode,
        "outcomes_path": str(outcomes_path),
        "split": split_meta,
        "layout": cascade_to_dict(cascade),
        "tunable_ids": list(tunable),
        "target_accuracy": target_accuracy,
        "target_accuracy_source": "collection_validation",
        "methods": methods_out,
    }


def write_comparison_md(summary: dict, output_dir: Path) -> Path:
    rows: list[dict] = []
    for key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        joint = report["methods"]["joint_anneal"]
        for method_name in METHODS:
            block = report["methods"][method_name]
            rows.append(
                {
                    "scene": report["scene"],
                    "detector_mode": report["detector_mode"],
                    "method": method_name,
                    "holdout_acc": block["holdout"]["accuracy"],
                    "holdout_cost_ms": block["holdout"]["expected_cost"],
                    "speedup_vs_collection": block.get("holdout_speedup_vs_collection"),
                    "feasible": block["holdout_feasible"],
                    "delta_cost_vs_joint_ms": block.get("delta_cost_vs_joint_ms"),
                    "delta_acc_vs_joint": block.get("delta_acc_vs_joint"),
                    "thresholds": block.get("thresholds"),
                }
            )

    md = [
        "# Independent vs Joint Thresholds — Comparison",
        "",
        "Question: does **joint** end-to-end annealing beat calibrating each Ki "
        "**independently** (precision / P(IDK)), on the same DP layout + holdout?",
        "",
        "| scene | detector | method | holdout acc | cost (ms) | speedup vs collect | "
        "feasible | Δcost vs joint | Δacc vs joint |",
        "|---|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        md.append(
            "| {scene} | {det} | {method} | {acc} | {cost} | {spd} | {feas} | "
            "{dc} | {da} |".format(
                scene=row["scene"],
                det=row["detector_mode"],
                method=row["method"],
                acc=_fmt(row["holdout_acc"]),
                cost=_fmt(row["holdout_cost_ms"]),
                spd=_fmt(row["speedup_vs_collection"]),
                feas=row["feasible"],
                dc=_fmt(row["delta_cost_vs_joint_ms"]),
                da=_fmt(row["delta_acc_vs_joint"]),
            )
        )

    # Count how often joint is the unique cheapest feasible method.
    joint_wins = 0
    indep_beats_joint = 0
    compared = 0
    for key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        compared += 1
        joint = report["methods"]["joint_anneal"]
        joint_cost = float(joint["holdout"]["expected_cost"])
        best_indep_cost = min(
            float(report["methods"][m]["holdout"]["expected_cost"])
            for m in METHODS
            if m.startswith("indep_")
        )
        if joint_cost < best_indep_cost - 1e-9:
            joint_wins += 1
        if best_indep_cost < joint_cost - 1e-9:
            indep_beats_joint += 1

    md.extend(
        [
            "",
            "## Verdict",
            "",
            f"Across {compared} (scene, detector) settings:",
            f"- Joint cheaper than every independent method in **{joint_wins}** settings",
            f"- Some independent method cheaper than joint in **{indep_beats_joint}** settings",
            "",
            "Δcost vs joint < 0 means that method is cheaper than joint anneal. "
            "Independent methods never see cascade cost while choosing thresholds.",
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
    colors = {
        "collection": "#6B7280",
        "indep_precision": "#8B3A3A",
        "indep_precision_match": "#C45C26",
        "indep_p_idk_match": "#B08968",
        "indep_p_idk_fixed": "#4C6A80",
        "joint_anneal": "#2F5D50",
    }
    written: list[Path] = []

    for detector_mode in ("paper", "trained"):
        scenes = [
            s
            for s in ALL_SCENES
            if summary.get("runs", {}).get(f"{s}__{detector_mode}", {}).get("status")
            == "ok"
        ]
        if not scenes:
            continue

        # Grouped cost bars
        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        x = np.arange(len(scenes))
        n_m = len(METHODS)
        width = 0.8 / n_m
        for i, method in enumerate(METHODS):
            costs = [
                float(
                    summary["runs"][f"{scene}__{detector_mode}"]["methods"][method][
                        "holdout"
                    ]["expected_cost"]
                )
                for scene in scenes
            ]
            ax.bar(
                x + (i - (n_m - 1) / 2) * width,
                costs,
                width=width,
                color=colors[method],
                label=METHOD_LABELS[method],
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.set_ylabel("Holdout expected cost (ms)")
        ax.set_title(f"Independent vs joint thresholds — cost ({detector_mode})")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        path = figures_dir / f"fig_indep_vs_joint_cost_{detector_mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

        # Δcost vs joint for independent methods only
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        indep_methods = [m for m in METHODS if m.startswith("indep_")]
        n_m = len(indep_methods)
        width = 0.8 / n_m
        for i, method in enumerate(indep_methods):
            deltas = [
                float(
                    summary["runs"][f"{scene}__{detector_mode}"]["methods"][method][
                        "delta_cost_vs_joint_ms"
                    ]
                )
                for scene in scenes
            ]
            ax.bar(
                x + (i - (n_m - 1) / 2) * width,
                deltas,
                width=width,
                color=colors[method],
                label=METHOD_LABELS[method],
                edgecolor="white",
                linewidth=0.3,
            )
        ax.axhline(0.0, color="#6B7280", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.set_ylabel("Δ holdout cost vs joint (ms)")
        ax.set_title(f"Independent − joint cost (neg. = indep cheaper) — {detector_mode}")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        path = figures_dir / f"fig_indep_vs_joint_delta_{detector_mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

        # Accuracy bars: collection vs best-indep vs joint
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        show = ("collection", "indep_precision", "indep_p_idk_match", "joint_anneal")
        n_m = len(show)
        width = 0.8 / n_m
        for i, method in enumerate(show):
            accs = [
                float(
                    summary["runs"][f"{scene}__{detector_mode}"]["methods"][method][
                        "holdout"
                    ]["accuracy"]
                )
                for scene in scenes
            ]
            ax.bar(
                x + (i - (n_m - 1) / 2) * width,
                accs,
                width=width,
                color=colors[method],
                label=METHOD_LABELS[method],
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(scenes)
        ax.set_ylabel("Holdout accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Independent vs joint — accuracy ({detector_mode})")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        path = figures_dir / f"fig_indep_vs_joint_acc_{detector_mode}.png"
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
        "experiment": "independent_vs_joint_thresholds",
        "question": (
            "Does joint end-to-end annealing beat independent per-Ki "
            "precision / P(IDK) calibration on the same cascade + holdout?"
        ),
        "methods": list(METHODS),
        "fixed_p_idk": FIXED_P_IDK,
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
