"""Kdet × accuracy-budget interaction (factorial grid).

Research question
-----------------
Is the huge paper speedup from “stacked lower-Kdet + 2pp” mostly **Kdet
alone**, mostly **budget alone**, or a real **interaction / synergy**?

Why this matters
----------------
Stacked recipes confounded the two levers. Multi-seed showed 2pp is stable
and 3pp is fragile. This grid separates the levers on a shared split per
seed and marks unsafe cells.

Usage
-----
    python experiment_kdet_budget_interaction.py
    python experiment_kdet_budget_interaction.py --scenes h24 --detector-modes paper
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


ALL_SCENES = ("h24", "h08")
DEFAULT_SEEDS = (0, 1, 2)
# Paper factorial axes (ms and accuracy shortfall).
PAPER_KDETS: tuple[float, ...] = (10_000.0, 5_000.0, 2_000.0, 1_000.0)
BUDGETS: tuple[float, ...] = (0.00, 0.01, 0.02, 0.03)
PROTECT_KDET = 10_000.0
PROTECT_BUDGET = 0.0

DEFAULT_OUTCOMES_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/kdet_budget_interaction")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")
PAPER_KDET_DEFAULT = float(PAPER_DETECTOR_COST_MS)


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


def _budget_label(budget: float) -> str:
    pp = budget * 100.0
    if abs(pp - round(pp)) < 1e-9:
        return f"{int(round(pp))}pp"
    return f"{str(pp).replace('.', 'p')}pp"


def _cell_key(kdet: float, budget: float) -> str:
    return f"kdet_{int(kdet)}__budget_{_budget_label(budget)}"


def _mean_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def anneal_cell(
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
    """Anneal one fixed cascade at floor = collection_micro − budget."""
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
        "detector_cost_ms": float(detector_cost_ms),
        "floor": float(floor),
        "baseline_micro_validation": baseline_micro,
        "layout": cascade_to_dict(cascade),
        "thresholds": thresholds,
        "collection_holdout": {
            "accuracy": float(collection_hold["accuracy"]),
            "expected_cost": float(collection_hold["expected_cost"]),
        },
        "validation": {
            "accuracy": float(opt_val["accuracy"]),
            "macro_accuracy": float(opt_val["macro_accuracy"]),
            "worst_class_accuracy": float(opt_val["worst_class_accuracy"]),
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


def run_one_seed_paper(
    scene: str,
    outcomes_path: Path,
    *,
    seed: int,
    split_strategy: str,
    kdets: tuple[float, ...],
    budgets: tuple[float, ...],
    annealing_iterations: int,
    quantile_points: int,
    holdout_fraction: float,
) -> dict:
    """One seed: shared split; DP per Kdet; anneal every (Kdet, budget) cell."""
    payload = load_empirical_outcomes(outcomes_path)
    validation_payload, holdout_payload, split_meta = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=seed,
    )

    cells: dict[str, Any] = {}
    layouts_by_kdet: dict[str, Any] = {}

    # Re-synthesize DP when Kdet changes; freeze that layout across budgets.
    for ki, kdet in enumerate(kdets):
        dp = HierarchyOptimizer(
            validation_payload,
            detector_mode="paper",
            detector_cost_ms=float(kdet),
        ).synthesize()
        cascade = make_cascade(
            _strip_detector(dp.initial),
            specialized={k: list(v) for k, v in dp.specialized.items()},
        )
        layouts_by_kdet[str(int(kdet))] = cascade_to_dict(cascade)

        for bi, budget in enumerate(budgets):
            key = _cell_key(kdet, budget)
            print(
                f"    [{key}] ...",
                flush=True,
            )
            # Distinct anneal seed per cell, but same data split for the seed.
            block = anneal_cell(
                validation_payload,
                holdout_payload,
                cascade=cascade,
                detector_mode="paper",
                detector_cost_ms=float(kdet),
                budget=float(budget),
                annealing_iterations=annealing_iterations,
                quantile_points=quantile_points,
                random_seed=seed * 10_000 + ki * 100 + bi,
            )
            block["cell"] = key
            cells[key] = block
            print(
                f"      holdout acc={block['holdout']['accuracy']:.4f}  "
                f"cost={block['holdout']['expected_cost']:.2f}ms",
                flush=True,
            )

    protect_key = _cell_key(PROTECT_KDET, PROTECT_BUDGET)
    protect = cells[protect_key]
    p_acc = float(protect["holdout"]["accuracy"])
    p_cost = float(protect["holdout"]["expected_cost"])

    for block in cells.values():
        h = block["holdout"]
        block["delta_vs_protect"] = {
            "accuracy": float(h["accuracy"]) - p_acc,
            "cost_ms": float(h["expected_cost"]) - p_cost,
            "speedup": _speedup(p_cost, float(h["expected_cost"])),
        }

    # Interaction residuals on the speedup scale for this seed.
    # pred(k,b) ≈ speedup(k,0) * speedup(10000,b)  (independent multiplicative model)
    interaction: dict[str, Any] = {}
    for kdet in kdets:
        for budget in budgets:
            key = _cell_key(kdet, budget)
            sp = cells[key]["delta_vs_protect"]["speedup"]
            sp_k0 = cells[_cell_key(kdet, 0.0)]["delta_vs_protect"]["speedup"]
            sp_b_only = cells[_cell_key(PROTECT_KDET, budget)]["delta_vs_protect"][
                "speedup"
            ]
            pred = None
            residual = None
            if sp_k0 is not None and sp_b_only is not None:
                pred = float(sp_k0) * float(sp_b_only)
                if sp is not None:
                    residual = float(sp) - pred
            interaction[key] = {
                "speedup": sp,
                "speedup_kdet_only_budget0": sp_k0,
                "speedup_budget_only_kdet10000": sp_b_only,
                "predicted_independent_speedup": pred,
                "residual_speedup": residual,
                # Synergy if actual beats independent product.
                "synergy": bool(residual is not None and residual > 0.05),
            }

    return {
        "scene": scene,
        "detector_mode": "paper",
        "seed": int(seed),
        "split": split_meta,
        "layouts_by_kdet": layouts_by_kdet,
        "cells": cells,
        "protect_cell": protect_key,
        "interaction": interaction,
    }


def run_one_seed_trained(
    scene: str,
    outcomes_path: Path,
    *,
    seed: int,
    split_strategy: str,
    budgets: tuple[float, ...],
    annealing_iterations: int,
    quantile_points: int,
    holdout_fraction: float,
) -> dict:
    """Trained reference: measured Kdet only (no synthetic Kdet grid)."""
    payload = load_empirical_outcomes(outcomes_path)
    measured = float(payload["detector"]["cost"])
    validation_payload, holdout_payload, split_meta = split_empirical_outcomes(
        payload,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        random_seed=seed,
    )
    dp = HierarchyOptimizer(
        validation_payload,
        detector_mode="trained",
        detector_cost_ms=measured,
    ).synthesize()
    cascade = make_cascade(
        _strip_detector(dp.initial),
        specialized={k: list(v) for k, v in dp.specialized.items()},
    )

    cells: dict[str, Any] = {}
    for bi, budget in enumerate(budgets):
        key = f"measured_kdet_{measured:g}__budget_{_budget_label(budget)}"
        print(f"    [{key}] ...", flush=True)
        block = anneal_cell(
            validation_payload,
            holdout_payload,
            cascade=cascade,
            detector_mode="trained",
            detector_cost_ms=measured,
            budget=float(budget),
            annealing_iterations=annealing_iterations,
            quantile_points=quantile_points,
            random_seed=seed * 10_000 + bi,
        )
        block["cell"] = key
        cells[key] = block
        print(
            f"      holdout acc={block['holdout']['accuracy']:.4f}  "
            f"cost={block['holdout']['expected_cost']:.2f}ms",
            flush=True,
        )

    protect_key = f"measured_kdet_{measured:g}__budget_{_budget_label(0.0)}"
    protect = cells[protect_key]
    p_acc = float(protect["holdout"]["accuracy"])
    p_cost = float(protect["holdout"]["expected_cost"])
    for block in cells.values():
        h = block["holdout"]
        block["delta_vs_protect"] = {
            "accuracy": float(h["accuracy"]) - p_acc,
            "cost_ms": float(h["expected_cost"]) - p_cost,
            "speedup": _speedup(p_cost, float(h["expected_cost"])),
        }

    return {
        "scene": scene,
        "detector_mode": "trained",
        "seed": int(seed),
        "split": split_meta,
        "measured_detector_cost_ms": measured,
        "layout": cascade_to_dict(cascade),
        "cells": cells,
        "protect_cell": protect_key,
        "note": "No synthetic Kdet grid in trained mode — budget sweep only.",
    }


def aggregate_paper_seeds(
    seed_reports: list[dict],
    *,
    kdets: tuple[float, ...],
    budgets: tuple[float, ...],
) -> dict:
    cells_agg: dict[str, Any] = {}
    for kdet in kdets:
        for budget in budgets:
            key = _cell_key(kdet, budget)
            accs, costs, speedups, d_accs, residuals = [], [], [], [], []
            within_3pp = 0
            n_ok = 0
            for report in seed_reports:
                block = report["cells"][key]
                d = block["delta_vs_protect"]
                n_ok += 1
                accs.append(float(block["holdout"]["accuracy"]))
                costs.append(float(block["holdout"]["expected_cost"]))
                if d.get("speedup") is not None:
                    speedups.append(float(d["speedup"]))
                if d.get("accuracy") is not None:
                    d_accs.append(float(d["accuracy"]))
                    if float(d["accuracy"]) >= -0.03:
                        within_3pp += 1
                inter = report.get("interaction", {}).get(key, {})
                if inter.get("residual_speedup") is not None:
                    residuals.append(float(inter["residual_speedup"]))

            mean_dacc = _mean_std(d_accs)["mean"]
            frac3 = float(within_3pp) / n_ok if n_ok else None
            safe = bool(
                n_ok > 0
                and mean_dacc is not None
                and float(mean_dacc) >= -0.03
                and frac3 is not None
                and frac3 >= 0.8
            )
            cells_agg[key] = {
                "kdet_ms": float(kdet),
                "budget": float(budget),
                "n_ok": n_ok,
                "holdout_accuracy": _mean_std(accs),
                "holdout_expected_cost": _mean_std(costs),
                "speedup_vs_protect": _mean_std(speedups),
                "delta_acc_vs_protect": _mean_std(d_accs),
                "residual_speedup_vs_independent": _mean_std(residuals),
                "fraction_seeds_delta_acc_ge_neg3pp": frac3,
                "safe": safe,
            }

    # Headline cells for the paper question.
    k1000_b0 = cells_agg[_cell_key(1_000.0, 0.0)]
    k1000_b2 = cells_agg[_cell_key(1_000.0, 0.02)]
    k10k_b2 = cells_agg[_cell_key(10_000.0, 0.02)]
    headline = {
        "kdet1000_budget0pp": {
            "speedup_mean": k1000_b0["speedup_vs_protect"]["mean"],
            "delta_acc_mean": k1000_b0["delta_acc_vs_protect"]["mean"],
            "safe": k1000_b0["safe"],
        },
        "kdet10000_budget2pp": {
            "speedup_mean": k10k_b2["speedup_vs_protect"]["mean"],
            "delta_acc_mean": k10k_b2["delta_acc_vs_protect"]["mean"],
            "safe": k10k_b2["safe"],
        },
        "kdet1000_budget2pp": {
            "speedup_mean": k1000_b2["speedup_vs_protect"]["mean"],
            "delta_acc_mean": k1000_b2["delta_acc_vs_protect"]["mean"],
            "safe": k1000_b2["safe"],
            "residual_vs_independent_mean": k1000_b2[
                "residual_speedup_vs_independent"
            ]["mean"],
        },
    }
    # Classify the previously winning cell.
    sp_combo = headline["kdet1000_budget2pp"]["speedup_mean"]
    sp_k = headline["kdet1000_budget0pp"]["speedup_mean"]
    sp_b = headline["kdet10000_budget2pp"]["speedup_mean"]
    resid = headline["kdet1000_budget2pp"]["residual_vs_independent_mean"]
    if sp_combo is None or sp_k is None or sp_b is None:
        verdict = "insufficient_data"
    elif resid is not None and float(resid) > 0.5:
        verdict = "synergy"
    elif float(sp_k) >= float(sp_b) and float(sp_k) >= 0.8 * float(sp_combo):
        verdict = "mostly_kdet_alone"
    elif float(sp_b) >= float(sp_k) and float(sp_b) >= 0.8 * float(sp_combo):
        verdict = "mostly_budget_alone"
    else:
        verdict = "both_contribute_near_independent"
    headline["verdict"] = verdict
    headline["verdict_note"] = (
        "synergy = residual > 0.5 vs speedup(k,0)*speedup(10k,b); "
        "mostly_kdet_alone = Kdet-only already ≈ combo; "
        "mostly_budget_alone = budget-only already ≈ combo."
    )
    return {"cells": cells_agg, "headline": headline}


def run_scene_mode(
    scene: str,
    outcomes_path: Path,
    *,
    detector_mode: str,
    seeds: tuple[int, ...],
    kdets: tuple[float, ...],
    budgets: tuple[float, ...],
    annealing_iterations: int,
    quantile_points: int,
    holdout_fraction: float,
) -> dict:
    seed_reports: list[dict] = []
    for seed in seeds:
        print(f"  --- seed={seed} (random_per_run) ---", flush=True)
        if detector_mode == "paper":
            seed_reports.append(
                run_one_seed_paper(
                    scene,
                    outcomes_path,
                    seed=seed,
                    split_strategy="random_per_run",
                    kdets=kdets,
                    budgets=budgets,
                    annealing_iterations=annealing_iterations,
                    quantile_points=quantile_points,
                    holdout_fraction=holdout_fraction,
                )
            )
        else:
            seed_reports.append(
                run_one_seed_trained(
                    scene,
                    outcomes_path,
                    seed=seed,
                    split_strategy="random_per_run",
                    budgets=budgets,
                    annealing_iterations=annealing_iterations,
                    quantile_points=quantile_points,
                    holdout_fraction=holdout_fraction,
                )
            )

    print("  --- blocked_per_run reference (seed=0) ---", flush=True)
    if detector_mode == "paper":
        blocked_ref = run_one_seed_paper(
            scene,
            outcomes_path,
            seed=0,
            split_strategy="blocked_per_run",
            kdets=kdets,
            budgets=budgets,
            annealing_iterations=annealing_iterations,
            quantile_points=quantile_points,
            holdout_fraction=holdout_fraction,
        )
        aggregate = aggregate_paper_seeds(
            seed_reports, kdets=kdets, budgets=budgets
        )
    else:
        blocked_ref = run_one_seed_trained(
            scene,
            outcomes_path,
            seed=0,
            split_strategy="blocked_per_run",
            budgets=budgets,
            annealing_iterations=annealing_iterations,
            quantile_points=quantile_points,
            holdout_fraction=holdout_fraction,
        )
        # Trained: simple budget aggregates only.
        cells_agg: dict[str, Any] = {}
        keys = list(seed_reports[0]["cells"].keys())
        for key in keys:
            speedups, d_accs = [], []
            within_3pp = 0
            for report in seed_reports:
                d = report["cells"][key]["delta_vs_protect"]
                if d.get("speedup") is not None:
                    speedups.append(float(d["speedup"]))
                if d.get("accuracy") is not None:
                    d_accs.append(float(d["accuracy"]))
                    if float(d["accuracy"]) >= -0.03:
                        within_3pp += 1
            n_ok = len(seed_reports)
            mean_dacc = _mean_std(d_accs)["mean"]
            frac3 = float(within_3pp) / n_ok if n_ok else None
            cells_agg[key] = {
                "speedup_vs_protect": _mean_std(speedups),
                "delta_acc_vs_protect": _mean_std(d_accs),
                "fraction_seeds_delta_acc_ge_neg3pp": frac3,
                "safe": bool(
                    mean_dacc is not None
                    and float(mean_dacc) >= -0.03
                    and frac3 is not None
                    and frac3 >= 0.8
                ),
            }
        aggregate = {"cells": cells_agg, "headline": {"verdict": "trained_budget_sweep_only"}}

    return {
        "scene": scene,
        "detector_mode": detector_mode,
        "split_strategy_primary": "random_per_run",
        "seeds": list(seeds),
        "kdets_ms": list(kdets) if detector_mode == "paper" else "measured_only",
        "budgets": list(budgets),
        "per_seed": seed_reports,
        "aggregate": aggregate,
        "blocked_per_run_seed0_reference": blocked_ref,
    }


def write_comparison_md(summary: dict, output_dir: Path) -> Path:
    md = [
        "# Kdet × Accuracy-Budget Interaction — Comparison",
        "",
        "Question: is `(Kdet=1000, budget=2pp)` mostly Kdet alone, mostly "
        "budget alone, or synergy?",
        "",
        "Protocol: seeds `{0,1,2}`, `random_per_run` 80/20, anneal 8000. "
        "Protect cell = `(Kdet=10000, budget=0pp)`.",
        "",
    ]

    for mode_scene_key in ("h24__paper", "h08__paper"):
        report = summary.get("runs", {}).get(mode_scene_key)
        if not report or report.get("status") != "ok":
            continue
        scene = report["scene"]
        md.append(f"## {scene}/paper — mean speedup heatmap")
        md.append("")
        md.append("| Kdet \\ budget | 0pp | 1pp | 2pp | 3pp |")
        md.append("|---|---:|---:|---:|---:|")
        for kdet in PAPER_KDETS:
            row = [f"{int(kdet)}"]
            for budget in BUDGETS:
                cell = report["aggregate"]["cells"][_cell_key(kdet, budget)]
                sp = cell["speedup_vs_protect"]["mean"]
                safe = "\\*" if cell["safe"] else ""
                row.append(f"{sp:.2f}{safe}" if sp is not None else "")
            md.append("| " + " | ".join(row) + " |")
        md.append("")
        md.append("\\* = safe (mean Δacc ≥ −3pp and ≥80% seeds pass).")
        md.append("")
        md.append(f"## {scene}/paper — mean Δacc (pp)")
        md.append("")
        md.append("| Kdet \\ budget | 0pp | 1pp | 2pp | 3pp |")
        md.append("|---|---:|---:|---:|---:|")
        for kdet in PAPER_KDETS:
            row = [f"{int(kdet)}"]
            for budget in BUDGETS:
                cell = report["aggregate"]["cells"][_cell_key(kdet, budget)]
                da = cell["delta_acc_vs_protect"]["mean"]
                row.append(f"{da * 100:.2f}" if da is not None else "")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

        h = report["aggregate"]["headline"]
        md.append(f"### Headline ({scene}/paper)")
        md.append("")
        md.append(
            f"- `(1000, 0pp)` speedup={_fmt(h['kdet1000_budget0pp']['speedup_mean'])} "
            f"safe={h['kdet1000_budget0pp']['safe']}"
        )
        md.append(
            f"- `(10000, 2pp)` speedup={_fmt(h['kdet10000_budget2pp']['speedup_mean'])} "
            f"safe={h['kdet10000_budget2pp']['safe']}"
        )
        md.append(
            f"- `(1000, 2pp)` speedup={_fmt(h['kdet1000_budget2pp']['speedup_mean'])} "
            f"residual_vs_independent={_fmt(h['kdet1000_budget2pp']['residual_vs_independent_mean'])} "
            f"safe={h['kdet1000_budget2pp']['safe']}"
        )
        md.append(f"- **Verdict:** `{h['verdict']}`")
        md.append("")

    md.extend(["## Safe cells", ""])
    for key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        safe = [
            name
            for name, cell in report["aggregate"]["cells"].items()
            if cell.get("safe")
        ]
        md.append(
            f"- **{report['scene']}/{report['detector_mode']}**: "
            + (", ".join(f'`{n}`' for n in safe) if safe else "_none_")
        )
    md.append("")

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
    report = summary.get("runs", {}).get("h24__paper")
    if not report or report.get("status") != "ok":
        return written

    kdets = list(PAPER_KDETS)
    budgets = list(BUDGETS)
    speed = np.full((len(kdets), len(budgets)), np.nan)
    dacc = np.full((len(kdets), len(budgets)), np.nan)
    for i, kdet in enumerate(kdets):
        for j, budget in enumerate(budgets):
            cell = report["aggregate"]["cells"][_cell_key(kdet, budget)]
            sp = cell["speedup_vs_protect"]["mean"]
            da = cell["delta_acc_vs_protect"]["mean"]
            if sp is not None:
                speed[i, j] = sp
            if da is not None:
                dacc[i, j] = da * 100.0

    def _heatmap(mat: np.ndarray, title: str, cbar: str, fname: str, cmap: str) -> Path:
        fig, ax = plt.subplots(figsize=(5.8, 4.2))
        im = ax.imshow(mat, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([_budget_label(b) for b in budgets])
        ax.set_yticks(range(len(kdets)))
        ax.set_yticklabels([str(int(k)) for k in kdets])
        ax.set_xlabel("accuracy budget")
        ax.set_ylabel("Kdet (ms)")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar)
        fig.tight_layout()
        path = figures_dir / fname
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        return path

    written.append(
        _heatmap(
            speed,
            "h24/paper mean speedup vs protect",
            "speedup",
            "fig_kdet_budget_h24_speedup_heatmap.png",
            "YlOrBr",
        )
    )
    written.append(
        _heatmap(
            dacc,
            "h24/paper mean Δacc vs protect (pp)",
            "Δacc (pp)",
            "fig_kdet_budget_h24_delta_acc_heatmap.png",
            "RdYlGn",
        )
    )

    # Line plot: speedup vs budget for each Kdet curve.
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = {
        10_000.0: "#6B7280",
        5_000.0: "#B08968",
        2_000.0: "#C45C26",
        1_000.0: "#2F5D50",
    }
    x = np.array([b * 100 for b in budgets])
    for kdet in kdets:
        ys = [
            report["aggregate"]["cells"][_cell_key(kdet, b)]["speedup_vs_protect"][
                "mean"
            ]
            for b in budgets
        ]
        ax.plot(
            x,
            ys,
            marker="o",
            color=colors.get(kdet, "#333"),
            label=f"Kdet={int(kdet)}",
            linewidth=1.6,
        )
    ax.set_xlabel("accuracy budget (pp)")
    ax.set_ylabel("mean speedup vs protect")
    ax.set_title("h24/paper — speedup vs budget by Kdet")
    ax.legend(frameon=False, fontsize=8)
    ax.axhline(1.0, color="#6B7280", linewidth=0.8)
    fig.tight_layout()
    path = figures_dir / "fig_kdet_budget_h24_speedup_curves.png"
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
        "experiment": "kdet_budget_interaction",
        "question": (
            "Is (Kdet=1000, budget=2pp) mostly Kdet alone, mostly budget alone, "
            "or synergy?"
        ),
        "annealing_iterations": args.iterations,
        "seeds": list(args.seeds),
        "paper_kdets_ms": list(PAPER_KDETS),
        "budgets": list(BUDGETS),
        "prompt": "prompts/kdet_budget_interaction.md",
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
                    kdets=PAPER_KDETS,
                    budgets=BUDGETS,
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
                if detector_mode == "paper":
                    print(
                        f"  verdict: {report['aggregate']['headline']['verdict']}",
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
    # Drop bulky nested duplication before writing summary: keep paths + aggregates.
    summary_for_disk = dict(summary)
    slim_runs = {}
    for key, report in summary["runs"].items():
        if report.get("status") != "ok":
            slim_runs[key] = report
            continue
        slim_runs[key] = {
            "status": "ok",
            "scene": report["scene"],
            "detector_mode": report["detector_mode"],
            "report_path": report.get("report_path"),
            "aggregate": report["aggregate"],
            "seeds": report["seeds"],
        }
    summary_for_disk["runs"] = slim_runs
    summary_path.write_text(
        json.dumps(summary_for_disk, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(f"\nWrote {summary_path}")
    write_comparison_md(summary, args.output_dir)
    if not args.skip_plots:
        plot_figures(summary, args.figures_dir)


if __name__ == "__main__":
    main()
