"""Stacked recipe: combine order + accuracy budget + optional lower Kdet.

Research question
-----------------
If we COMBINE the best speed levers (sequence order, accuracy budget, optional
lower paper-Kdet cost), do we beat any single lever alone without a major
accuracy drop vs protecting baseline?

Why stacking might help
-----------------------
- **Order** changes who accepts early (short-circuit path cost).
- **Accuracy budget** lets thresholds be more aggressive (accept more, skip
  expensive later stages) while allowing a small micro shortfall.
- **Lower Kdet cost** changes the DP structure itself (often shorter chains)
  and the accounting cost of falling through to the detector.

Single levers touch one of those; stacked recipes touch several at once.

Usage
-----
    python experiment_stacked_recipe.py
    python experiment_stacked_recipe.py --scenes h24 --detector-modes paper
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
from experiment_sequence_order_ablations import (
    build_order_candidates,
    specialized_for_order,
)
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
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/stacked_recipe")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")
PAPER_KDET = float(PAPER_DETECTOR_COST_MS)  # 10_000
LOW_KDET = 1_000.0
# Same capping spirit as experiment_sequence_order_ablations:
#   - if chain length <= 4, try all permutations (4! = 24, affordable)
#   - if longer, fall back to structured + a few randoms (never 5! = 120+)
# We still DROP ref_* orders so stacking does not invent alternate memberships.
ORDER_MAX_EXHAUSTIVE = 4
ORDER_MAX_RANDOM = 8


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


def compact_order_candidates(
    dp_initial: list[str],
    *,
    random_seed: int,
) -> list[tuple[str, list[str]]]:
    """Capped order set for stacked recipes (validation-only selection).

    What this returns
    -----------------
    A list of ``(label, order)`` pairs.  ``label`` is a human-readable name
    like ``"dp_order"`` or ``"perm_K3_K0_…"``; ``order`` is the classifier
    sequence *without* the trailing detector.

    Why we cap
    ----------
    ``n!`` grows fast (5! = 120).  Matching the sequence-order experiment,
    we allow exhaustive permutations only when ``n <= ORDER_MAX_EXHAUSTIVE``
    (default 4).  We also drop ``ref_*`` candidates so we only reorder the
    DP membership set — stacking is about order/budget/Kdet, not inventing
    a new classifier set.

    What if you raised ORDER_MAX_EXHAUSTIVE to 5?
        You would anneal ~120 layouts during order search.  That is ~5×
        slower and usually not worth it for a stacking ablation.
    """
    # build_order_candidates already de-duplicates and puts DP first.
    raw = build_order_candidates(
        dp_initial,
        random_seed=random_seed,
        max_exhaustive_len=ORDER_MAX_EXHAUSTIVE,
        max_random_perms=ORDER_MAX_RANDOM,
    )
    # List comprehension with a filter: keep everything except reference
    # membership variants (those change *who* is on the trunk, not just order).
    keep: list[tuple[str, list[str]]] = [
        (label, order)
        for label, order in raw
        if not label.startswith("ref_")
    ]
    # Defensive: if somehow DP vanished, put it back at index 0.
    if not keep or keep[0][0] != "dp_order":
        dp = _strip_detector(dp_initial)
        keep = [("dp_order", dp)] + [x for x in keep if x[0] != "dp_order"]
    return keep


def anneal_layout(
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
    """Anneal one fixed cascade at floor = collection_micro - budget; score holdout."""
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
        "floor": float(floor),
        "baseline_micro_validation": baseline_micro,
        "layout": cascade_to_dict(cascade),
        "thresholds": thresholds,
        "collection_validation": {
            "accuracy": float(collection_val["accuracy"]),
            "expected_cost": float(collection_val["expected_cost"]),
        },
        "collection_holdout": {
            "accuracy": float(collection_hold["accuracy"]),
            "macro_accuracy": float(collection_hold["macro_accuracy"]),
            "worst_class_accuracy": float(collection_hold["worst_class_accuracy"]),
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
            "route_counts": opt_hold.get("route_counts"),
        },
        "holdout_feasible": bool(float(opt_hold["accuracy"]) >= floor),
        "anneal_feasible": bool(annealed.get("feasible")),
        "holdout_speedup_vs_collection": _speedup(
            float(collection_hold["expected_cost"]),
            float(opt_hold["expected_cost"]),
        ),
    }


def select_best_order(
    validation_payload: dict,
    holdout_payload: dict,
    *,
    dp_cascade: Cascade,
    detector_mode: str,
    detector_cost_ms: float,
    annealing_iterations: int,
    quantile_points: int,
    random_seed: int,
    cache: dict[tuple, dict],
) -> tuple[list[str], dict[tuple[str, str], list[str]], dict]:
    """Validation-only order search at budget=0pp; return (order, specialized, log).

    Holdout is passed through for anneal_layout scoring but MUST NOT be used
    to pick the winner — we rank by validation metrics only.
    """
    candidates = compact_order_candidates(
        dp_cascade.initial, random_seed=random_seed
    )
    dp_specialized = {
        key: list(chain) for key, chain in dp_cascade.specialized.items()
    }
    order_log: list[dict] = []
    best: dict | None = None
    best_order: list[str] = _strip_detector(dp_cascade.initial)
    best_spec = specialized_for_order(dp_specialized, best_order)

    for i, (label, order) in enumerate(candidates):
        spec = specialized_for_order(dp_specialized, order)
        cascade = make_cascade(list(order), specialized=spec)
        cache_key = (
            float(detector_cost_ms),
            tuple(order),
            0.0,
            "order_search",
        )
        if cache_key in cache:
            result = cache[cache_key]
        else:
            result = anneal_layout(
                validation_payload,
                holdout_payload,
                cascade=cascade,
                detector_mode=detector_mode,
                detector_cost_ms=detector_cost_ms,
                budget=0.0,
                annealing_iterations=annealing_iterations,
                quantile_points=quantile_points,
                random_seed=random_seed + 11 * (i + 1),
            )
            cache[cache_key] = result
        entry = {
            "label": label,
            "order": list(order),
            "validation_accuracy": result["validation"]["accuracy"],
            "validation_cost": result["validation"]["expected_cost"],
            "validation_feasible": result["validation"]["feasible"],
        }
        order_log.append(entry)
        # Rank on VALIDATION only: feasible first, then lower cost, then higher acc.
        # This tuple is compared lexicographically (left-to-right), like sorting
        # rows in a spreadsheet by column A, then B, then C.
        #   key[0] = 0 if feasible else 1   → every feasible beats every infeasible
        #   key[1] = expected cost          → among feasible, cheaper wins
        #   key[2] = -accuracy              → tie-break: higher acc (more negative -acc)
        # What if you ranked by holdout cost here instead?
        #   That would leak the test set into model selection = optimistic bias.
        key = (
            0 if result["validation"]["feasible"] else 1,
            float(result["validation"]["expected_cost"]),
            -float(result["validation"]["accuracy"]),
        )
        if best is None or key < best["key"]:
            best = {"key": key, "label": label, "result": result}
            best_order = list(order)
            best_spec = spec

    assert best is not None
    return best_order, best_spec, {
        "selected_label": best["label"],
        "selected_order": best_order,
        "candidates": order_log,
    }


def run_recipe(
    name: str,
    validation_payload: dict,
    holdout_payload: dict,
    *,
    detector_mode: str,
    detector_cost_ms: float,
    order: list[str] | None,
    specialized: dict[tuple[str, str], list[str]] | None,
    budget: float,
    annealing_iterations: int,
    quantile_points: int,
    random_seed: int,
    cache: dict[tuple, dict],
    synthesize_dp: bool = True,
) -> dict:
    """Run one named recipe; reuse cache when (cost, order, budget) matches."""
    if synthesize_dp or order is None:
        dp_opt = HierarchyOptimizer(
            validation_payload,
            detector_mode=detector_mode,
            detector_cost_ms=detector_cost_ms,
        )
        dp_cascade = dp_opt.synthesize()
        if order is None:
            order = _strip_detector(dp_cascade.initial)
            specialized = {
                k: list(v) for k, v in dp_cascade.specialized.items()
            }
        elif specialized is None:
            specialized = specialized_for_order(
                {k: list(v) for k, v in dp_cascade.specialized.items()},
                order,
            )
    assert order is not None and specialized is not None
    cascade = make_cascade(list(order), specialized=specialized)
    cache_key = (float(detector_cost_ms), tuple(order), float(budget), "recipe")
    if cache_key in cache:
        result = dict(cache[cache_key])
    else:
        result = anneal_layout(
            validation_payload,
            holdout_payload,
            cascade=cascade,
            detector_mode=detector_mode,
            detector_cost_ms=detector_cost_ms,
            budget=budget,
            annealing_iterations=annealing_iterations,
            quantile_points=quantile_points,
            random_seed=random_seed,
        )
        cache[cache_key] = result
    out = dict(result)
    out["recipe"] = name
    out["detector_mode"] = detector_mode
    out["requested_detector_cost_ms"] = float(detector_cost_ms)
    return out


def run_scene_mode(
    scene: str,
    outcomes_path: Path,
    *,
    detector_mode: str,
    annealing_iterations: int,
    order_search_iterations: int,
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
    cache: dict[tuple, dict] = {}
    recipes_out: dict[str, Any] = {}

    # DP at paper default cost (or trained measured cost).
    default_cost = PAPER_KDET if detector_mode == "paper" else float(
        payload["detector"]["cost"]
    )
    dp_opt = HierarchyOptimizer(
        validation_payload,
        detector_mode=detector_mode,
        detector_cost_ms=default_cost,
    )
    dp_cascade = dp_opt.synthesize()
    dp_order = _strip_detector(dp_cascade.initial)
    dp_spec = {k: list(v) for k, v in dp_cascade.specialized.items()}

    def go(
        name: str,
        *,
        cost: float,
        order: list[str] | None,
        spec: dict | None,
        budget: float,
        seed_off: int,
    ) -> dict:
        print(
            f"  [{name}] cost={cost:g} budget={budget} "
            f"order={order if order is not None else 'DP'} ...",
            flush=True,
        )
        block = run_recipe(
            name,
            validation_payload,
            holdout_payload,
            detector_mode=detector_mode,
            detector_cost_ms=cost,
            order=order,
            specialized=spec,
            budget=budget,
            annealing_iterations=annealing_iterations,
            quantile_points=quantile_points,
            random_seed=random_seed + seed_off,
            cache=cache,
            synthesize_dp=(order is None),
        )
        print(
            f"    holdout acc={block['holdout']['accuracy']:.4f}  "
            f"cost={block['holdout']['expected_cost']:.2f}ms  "
            f"feas={block['holdout_feasible']}",
            flush=True,
        )
        return block

    # A) baseline_protect
    recipes_out["baseline_protect"] = go(
        "baseline_protect",
        cost=default_cost,
        order=dp_order,
        spec=dp_spec,
        budget=0.0,
        seed_off=1,
    )
    # Collection baseline on that layout (already inside result).
    recipes_out["collection"] = {
        "recipe": "collection",
        "annealed": False,
        "layout": recipes_out["baseline_protect"]["layout"],
        "holdout": recipes_out["baseline_protect"]["collection_holdout"],
        "validation": recipes_out["baseline_protect"]["collection_validation"],
        "thresholds": None,
        "budget": None,
        "floor": None,
        "holdout_feasible": True,
        "holdout_speedup_vs_collection": 1.0,
    }

    # B / C) budget only
    recipes_out["budget_only_2pp"] = go(
        "budget_only_2pp",
        cost=default_cost,
        order=dp_order,
        spec=dp_spec,
        budget=0.02,
        seed_off=2,
    )
    recipes_out["budget_only_3pp"] = go(
        "budget_only_3pp",
        cost=default_cost,
        order=dp_order,
        spec=dp_spec,
        budget=0.03,
        seed_off=3,
    )

    # D) order search at budget 0 (validation-only pick)
    print("  [order_only] searching orders at budget=0pp (validation rank)...", flush=True)
    best_order, best_spec, order_search = select_best_order(
        validation_payload,
        holdout_payload,
        dp_cascade=dp_cascade,
        detector_mode=detector_mode,
        detector_cost_ms=default_cost,
        annealing_iterations=order_search_iterations,
        quantile_points=quantile_points,
        random_seed=random_seed + 100,
        cache=cache,
    )
    # Final order_only result: reuse cache entry if present, else anneal at full iters.
    recipes_out["order_only"] = go(
        "order_only",
        cost=default_cost,
        order=best_order,
        spec=best_spec,
        budget=0.0,
        seed_off=4,
    )
    recipes_out["order_only"]["order_search"] = order_search

    # E / F) stacked order + budget
    recipes_out["stacked_order_budget_2pp"] = go(
        "stacked_order_budget_2pp",
        cost=default_cost,
        order=best_order,
        spec=best_spec,
        budget=0.02,
        seed_off=5,
    )
    recipes_out["stacked_order_budget_3pp"] = go(
        "stacked_order_budget_3pp",
        cost=default_cost,
        order=best_order,
        spec=best_spec,
        budget=0.03,
        seed_off=6,
    )

    # G / H) paper-only lower Kdet stacks
    if detector_mode == "paper":
        # G: DP at 1000 + budget 2pp
        dp_low = HierarchyOptimizer(
            validation_payload,
            detector_mode="paper",
            detector_cost_ms=LOW_KDET,
        ).synthesize()
        low_order = _strip_detector(dp_low.initial)
        low_spec = {k: list(v) for k, v in dp_low.specialized.items()}
        recipes_out["stacked_kdet1000_budget_2pp"] = go(
            "stacked_kdet1000_budget_2pp",
            cost=LOW_KDET,
            order=low_order,
            spec=low_spec,
            budget=0.02,
            seed_off=7,
        )

        # H: order search at 1000, then budget 2pp
        print(
            "  [stacked_full_kdet1000...] order search at Kdet=1000 ...",
            flush=True,
        )
        best_low_order, best_low_spec, low_order_search = select_best_order(
            validation_payload,
            holdout_payload,
            dp_cascade=dp_low,
            detector_mode="paper",
            detector_cost_ms=LOW_KDET,
            annealing_iterations=order_search_iterations,
            quantile_points=quantile_points,
            random_seed=random_seed + 200,
            cache=cache,
        )
        recipes_out["stacked_full_kdet1000_order_budget_2pp"] = go(
            "stacked_full_kdet1000_order_budget_2pp",
            cost=LOW_KDET,
            order=best_low_order,
            spec=best_low_spec,
            budget=0.02,
            seed_off=8,
        )
        recipes_out["stacked_full_kdet1000_order_budget_2pp"][
            "order_search"
        ] = low_order_search
    else:
        recipes_out["stacked_kdet1000_budget_2pp"] = {
            "status": "skipped",
            "reason": "paper-only recipe (trained Kdet cost is measured, not synthetic)",
        }
        recipes_out["stacked_full_kdet1000_order_budget_2pp"] = {
            "status": "skipped",
            "reason": "paper-only recipe (trained Kdet cost is measured, not synthetic)",
        }

    # Deltas vs baseline_protect
    base = recipes_out["baseline_protect"]
    base_acc = float(base["holdout"]["accuracy"])
    base_cost = float(base["holdout"]["expected_cost"])
    for name, block in recipes_out.items():
        if name == "collection" or block.get("status") == "skipped":
            continue
        if "holdout" not in block or "accuracy" not in block["holdout"]:
            continue
        h = block["holdout"]
        block["delta_vs_baseline_protect"] = {
            "accuracy": float(h["accuracy"]) - base_acc,
            "cost_ms": float(h["expected_cost"]) - base_cost,
            "speedup": _speedup(base_cost, float(h["expected_cost"])),
        }

    # Winner among recipes with Δacc >= -3pp vs baseline_protect
    eligible = []
    for name, block in recipes_out.items():
        if name in {"collection"} or block.get("status") == "skipped":
            continue
        d = block.get("delta_vs_baseline_protect")
        if not d:
            continue
        if float(d["accuracy"]) >= -0.03:
            eligible.append((name, block))
    if eligible:
        winner_name, winner_block = min(
            eligible,
            key=lambda pair: float(pair[1]["holdout"]["expected_cost"]),
        )
    else:
        winner_name, winner_block = "baseline_protect", base

    # Explicit stacking vs single-lever comparison (same holdout split).
    # Single levers = B/C/D; stacked = E/F/G/H.  "Beats" means lower holdout
    # cost among recipes that stay within -3pp of baseline_protect accuracy.
    single_names = ("budget_only_2pp", "budget_only_3pp", "order_only")
    stacked_names = (
        "stacked_order_budget_2pp",
        "stacked_order_budget_3pp",
        "stacked_kdet1000_budget_2pp",
        "stacked_full_kdet1000_order_budget_2pp",
    )

    def _best_in(names: tuple[str, ...]) -> dict | None:
        pool = []
        for name in names:
            block = recipes_out.get(name)
            if not block or block.get("status") == "skipped":
                continue
            d = block.get("delta_vs_baseline_protect") or {}
            if float(d.get("accuracy", -1.0)) < -0.03:
                continue
            pool.append((name, block))
        if not pool:
            return None
        name, block = min(pool, key=lambda p: float(p[1]["holdout"]["expected_cost"]))
        return {
            "recipe": name,
            "holdout_accuracy": block["holdout"]["accuracy"],
            "holdout_expected_cost": block["holdout"]["expected_cost"],
            "speedup_vs_baseline_protect": (block.get("delta_vs_baseline_protect") or {}).get(
                "speedup"
            ),
        }

    best_single = _best_in(single_names)
    best_stacked = _best_in(stacked_names)
    stacking_helps = None
    if best_single and best_stacked:
        stacking_helps = float(best_stacked["holdout_expected_cost"]) < float(
            best_single["holdout_expected_cost"]
        )

    return {
        "scene": scene,
        "detector_mode": detector_mode,
        "split": split_meta,
        "default_detector_cost_ms": default_cost,
        "dp_layout_default_cost": cascade_to_dict(dp_cascade),
        "recipes": recipes_out,
        "winner_within_3pp_acc": {
            "recipe": winner_name,
            "holdout_accuracy": winner_block["holdout"]["accuracy"],
            "holdout_expected_cost": winner_block["holdout"]["expected_cost"],
            "delta_vs_baseline_protect": winner_block.get(
                "delta_vs_baseline_protect"
            ),
        },
        "stacking_vs_single_lever": {
            "best_single_within_3pp": best_single,
            "best_stacked_within_3pp": best_stacked,
            "stacking_beats_best_single": stacking_helps,
            "note": (
                "Compared on the SAME holdout split. Negative result "
                "(stacking_beats_best_single=false) is acceptable."
            ),
        },
    }


def write_comparison_md(summary: dict, output_dir: Path) -> Path:
    rows: list[dict] = []
    for _key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        for name, block in report["recipes"].items():
            if name == "collection":
                continue
            if block.get("status") == "skipped":
                rows.append(
                    {
                        "scene": report["scene"],
                        "detector_mode": report["detector_mode"],
                        "recipe": name,
                        "status": "skipped",
                    }
                )
                continue
            d = block.get("delta_vs_baseline_protect") or {}
            rows.append(
                {
                    "scene": report["scene"],
                    "detector_mode": report["detector_mode"],
                    "recipe": name,
                    "status": "ok",
                    "layout": block.get("layout", {}).get("initial"),
                    "holdout_acc": block["holdout"]["accuracy"],
                    "holdout_cost": block["holdout"]["expected_cost"],
                    "feas": block.get("holdout_feasible"),
                    "d_acc": d.get("accuracy"),
                    "d_cost": d.get("cost_ms"),
                    "speedup_vs_base": d.get("speedup"),
                    "winner": report.get("winner_within_3pp_acc", {}).get("recipe")
                    == name,
                }
            )

    md = [
        "# Stacked Recipe — Comparison",
        "",
        "Question: does combining **order + accuracy budget (+ optional lower "
        "Kdet)** beat any single lever alone, without a major accuracy drop?",
        "",
        "Winner column: cheapest recipe with Δacc vs `baseline_protect` ≥ −3pp.",
        "",
        "| scene | detector | recipe | holdout acc | cost (ms) | Δacc vs base | "
        "Δcost vs base | speedup vs base | feas | winner? |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        if row.get("status") == "skipped":
            md.append(
                f"| {row['scene']} | {row['detector_mode']} | {row['recipe']} | "
                f"| skipped | | | | | |"
            )
            continue
        md.append(
            "| {scene} | {det} | {rec} | {acc} | {cost} | {da} | {dc} | {sp} | "
            "{feas} | {win} |".format(
                scene=row["scene"],
                det=row["detector_mode"],
                rec=row["recipe"],
                acc=_fmt(row["holdout_acc"]),
                cost=_fmt(row["holdout_cost"]),
                da=_fmt(row["d_acc"]),
                dc=_fmt(row["d_cost"]),
                sp=_fmt(row["speedup_vs_base"]),
                feas=row["feas"],
                win="YES" if row["winner"] else "",
            )
        )

    # Verdict: does a stacked recipe win?
    md.extend(["", "## Verdict", ""])
    stacked_names = {
        "stacked_order_budget_2pp",
        "stacked_order_budget_3pp",
        "stacked_kdet1000_budget_2pp",
        "stacked_full_kdet1000_order_budget_2pp",
    }
    n = 0
    stacked_wins = 0
    stacking_helps_n = 0
    stacking_compared_n = 0
    for _key, report in summary.get("runs", {}).items():
        if report.get("status") != "ok":
            continue
        n += 1
        w = report.get("winner_within_3pp_acc", {}).get("recipe")
        if w in stacked_names:
            stacked_wins += 1
        svs = report.get("stacking_vs_single_lever") or {}
        helps = svs.get("stacking_beats_best_single")
        best_s = (svs.get("best_single_within_3pp") or {}).get("recipe")
        best_st = (svs.get("best_stacked_within_3pp") or {}).get("recipe")
        if helps is not None:
            stacking_compared_n += 1
            if helps:
                stacking_helps_n += 1
        md.append(
            f"- **{report['scene']}/{report['detector_mode']}** winner: `{w}` "
            f"(best single `{best_s}`, best stacked `{best_st}`, "
            f"stacking_helps={helps})"
        )
    md.append("")
    md.append(
        f"Stacked recipe won (within −3pp acc) in **{stacked_wins}/{n}** "
        f"(scene, detector) settings."
    )
    md.append(
        f"Stacking beat the best single lever (same −3pp gate) in "
        f"**{stacking_helps_n}/{stacking_compared_n}** comparable settings. "
        f"If that fraction is low, stacking did **not** help beyond the best "
        f"individual lever on that split — a valid negative result."
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
    recipe_order = [
        "baseline_protect",
        "budget_only_2pp",
        "budget_only_3pp",
        "order_only",
        "stacked_order_budget_2pp",
        "stacked_order_budget_3pp",
        "stacked_kdet1000_budget_2pp",
        "stacked_full_kdet1000_order_budget_2pp",
    ]
    colors = {
        "baseline_protect": "#6B7280",
        "budget_only_2pp": "#B08968",
        "budget_only_3pp": "#C45C26",
        "order_only": "#4C6A80",
        "stacked_order_budget_2pp": "#2F5D50",
        "stacked_order_budget_3pp": "#1B4332",
        "stacked_kdet1000_budget_2pp": "#8B3A3A",
        "stacked_full_kdet1000_order_budget_2pp": "#5C1A1A",
    }
    written: list[Path] = []

    # h24 paper cost bars
    for mode in ("paper", "trained"):
        key = f"h24__{mode}"
        report = summary.get("runs", {}).get(key)
        if not report or report.get("status") != "ok":
            continue
        names, costs, cols = [], [], []
        for name in recipe_order:
            block = report["recipes"].get(name)
            if not block or block.get("status") == "skipped" or "holdout" not in block:
                continue
            if "expected_cost" not in block["holdout"]:
                continue
            names.append(name.replace("stacked_", "S:").replace("budget_only_", "B:"))
            costs.append(float(block["holdout"]["expected_cost"]))
            cols.append(colors.get(name, "#333"))
        if not names:
            continue
        fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(names) + 2), 3.8))
        ax.bar(np.arange(len(names)), costs, color=cols, edgecolor="white", linewidth=0.3)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Holdout expected cost (ms)")
        ax.set_title(f"h24 stacked recipes — cost ({mode})")
        fig.tight_layout()
        path = figures_dir / f"fig_stacked_h24_cost_{mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

        # Pareto scatter
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        for name in recipe_order:
            block = report["recipes"].get(name)
            if not block or block.get("status") == "skipped" or "holdout" not in block:
                continue
            if "accuracy" not in block["holdout"]:
                continue
            ax.scatter(
                float(block["holdout"]["expected_cost"]),
                float(block["holdout"]["accuracy"]),
                s=55,
                c=colors.get(name, "#333"),
                label=name,
                zorder=3,
            )
        ax.set_xlabel("Holdout expected cost (ms)")
        ax.set_ylabel("Holdout accuracy")
        ax.set_title(f"h24 recipe Pareto ({mode})")
        ax.legend(frameon=False, fontsize=6, loc="best")
        fig.tight_layout()
        path = figures_dir / f"fig_stacked_h24_pareto_{mode}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(path)

    # All scenes: speedup vs baseline_protect for key recipes
    for mode in ("paper", "trained"):
        key_recipes = [
            "budget_only_3pp",
            "order_only",
            "stacked_order_budget_3pp",
            "stacked_kdet1000_budget_2pp",
            "stacked_full_kdet1000_order_budget_2pp",
        ]
        scenes = []
        for scene in ALL_SCENES:
            report = summary.get("runs", {}).get(f"{scene}__{mode}")
            if report and report.get("status") == "ok":
                scenes.append(scene)
        if not scenes:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        x = np.arange(len(scenes))
        width = 0.15
        drawn = 0
        for i, rname in enumerate(key_recipes):
            ys = []
            ok = False
            for scene in scenes:
                block = summary["runs"][f"{scene}__{mode}"]["recipes"].get(rname)
                if not block or block.get("status") == "skipped":
                    ys.append(np.nan)
                else:
                    sp = (block.get("delta_vs_baseline_protect") or {}).get("speedup")
                    ys.append(float(sp) if sp is not None else np.nan)
                    ok = True
            if not ok:
                continue
            ax.bar(
                x + (i - 2) * width,
                ys,
                width=width,
                color=colors.get(rname, "#333"),
                label=rname.replace("stacked_", "S:"),
                edgecolor="white",
                linewidth=0.3,
            )
            drawn += 1
        if drawn:
            ax.axhline(1.0, color="#6B7280", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(scenes)
            ax.set_ylabel("Speedup vs baseline_protect")
            ax.set_title(f"Stacked vs single levers — speedup ({mode})")
            ax.legend(frameon=False, fontsize=6, ncol=2)
            fig.tight_layout()
            path = figures_dir / f"fig_stacked_speedup_{mode}.png"
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
    parser.add_argument("--iterations", type=int, default=8_000)
    parser.add_argument(
        "--order-search-iterations",
        type=int,
        default=2_000,
        help="Cheaper anneal budget for validation-only order screening.",
    )
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
    summary: dict[str, Any] = {
        "experiment": "stacked_recipe",
        "question": (
            "Does combining order + accuracy budget (+ optional lower Kdet) "
            "beat any single lever alone without a major accuracy drop?"
        ),
        "annealing_iterations": args.iterations,
        "order_search_iterations": args.order_search_iterations,
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
                    annealing_iterations=args.iterations,
                    order_search_iterations=args.order_search_iterations,
                    quantile_points=args.quantile_points,
                    holdout_fraction=args.holdout_fraction,
                    split_strategy=args.split_strategy,
                    random_seed=args.seed,
                )
                report["status"] = "ok"
                path = args.output_dir / f"{key}.json"
                path.write_text(
                    json.dumps(report, indent=2, sort_keys=True, default=float) + "\n"
                )
                report["report_path"] = str(path)
                summary["runs"][key] = report
                w = report["winner_within_3pp_acc"]
                print(
                    f"  winner(≤3pp acc drop): {w['recipe']}  "
                    f"cost={w['holdout_expected_cost']:.2f}ms  "
                    f"acc={w['holdout_accuracy']:.4f}"
                )
                print(f"  Wrote {path}")
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
