"""A4 nested joint approximator: layout family × SA thresholds.

Outer loop tries a small set of cascade layouts S.
Inner loop = SA approximator for thresholds H (reuse threshold_optimizer).

Objective F (locked):
  minimize E[cost_ms]_val  s.t.  Acc_val >= 0.98337
  report holdout Acc/cost separately.

Complexity: O(L * I * N * D) with L=#layouts, I=SA iters, N=samples, D=depth.

Run (from repo root):
  python joint_a4_nested.py --outcomes path/to/empirical_outcomes.pkl --save
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hierarchy_optimizer import (
    PAPER_DETECTOR_COST_MS,
    Cascade,
    HierarchyOptimizer,
    filter_disabled_cascade_kis,
)
from threshold_optimizer import (
    SLOT_SEP,
    optimize_and_evaluate_holdout,
    split_empirical_outcomes,
)
from empirical_outcomes import load_empirical_outcomes

TARGET_ACCURACY = 0.98337
QUANTILE_POINTS = 50
ANNEAL_ITERS = 8000
HOLDOUT_FRACTION = 0.20
DETECTOR_MODE = "paper"
DETECTOR_COST_MS = float(PAPER_DETECTOR_COST_MS)
SEED = 0
SPLIT_STRATEGY = "blocked_per_run"


@dataclass
class Bank:
    thresholds: dict[str, float] = field(default_factory=dict)
    initial: list[str] = field(default_factory=list)
    specialized: dict[str, list[str]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


def _specialized_to_json(specialized: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, chain in specialized.items():
        if isinstance(key, tuple):
            router, group = key
            out[f"{router}|{group}"] = list(chain)
        else:
            out[str(key)] = list(chain)
    return out


def _policy_key(val_acc: float | None, val_cost: float | None) -> tuple:
    feasible = val_acc is not None and float(val_acc) >= TARGET_ACCURACY - 1e-12
    cost = float(val_cost) if val_cost is not None else float("inf")
    return (0 if feasible else 1, cost)


def synthesize_dp_layout(outcomes_path: Path) -> Cascade:
    payload = filter_disabled_cascade_kis(load_empirical_outcomes(outcomes_path))
    validation_payload, _, _ = split_empirical_outcomes(
        payload,
        holdout_fraction=HOLDOUT_FRACTION,
        split_strategy=SPLIT_STRATEGY,
        random_seed=SEED,
    )
    opt = HierarchyOptimizer(
        validation_payload,
        detector_mode=DETECTOR_MODE,
        detector_cost_ms=DETECTOR_COST_MS,
    )
    return opt.synthesize()


def a4_layout_families(outcomes_path: Path) -> list[tuple[str, Cascade]]:
    """Discrete layout set for nested joint search (K1 never included)."""
    dp = synthesize_dp_layout(outcomes_path)
    teammate = Cascade(
        expected_cost=0.0,
        initial=["K0", "K3", "detector"],
        specialized={
            ("K0", "coupe"): ["K3", "K5", "K6", "K2", "detector"],
            ("K0", "suv"): ["K3", "K2", "K4", "detector"],
        },
        detector="detector",
    )
    short_global = Cascade(
        expected_cost=0.0,
        initial=["K0", "K2", "detector"],
        specialized={
            ("K0", "coupe"): ["K6", "detector"],
            ("K0", "suv"): ["K4", "detector"],
        },
        detector="detector",
    )
    k0_only = Cascade(
        expected_cost=0.0,
        initial=["K0", "detector"],
        specialized={
            ("K0", "coupe"): ["K5", "K6", "detector"],
            ("K0", "suv"): ["K4", "detector"],
        },
        detector="detector",
    )
    return [
        ("dp_expand", dp),
        ("teammate_short", teammate),
        ("short_global", short_global),
        ("k0_branchy", k0_only),
    ]


def run_sa_on_cascade(outcomes_path: Path, cascade: Cascade) -> dict:
    return optimize_and_evaluate_holdout(
        outcomes_path,
        TARGET_ACCURACY,
        method="anneal",
        detector_mode=DETECTOR_MODE,
        detector_cost_ms=DETECTOR_COST_MS,
        holdout_fraction=HOLDOUT_FRACTION,
        split_strategy=SPLIT_STRATEGY,
        quantile_points=QUANTILE_POINTS,
        annealing_iterations=ANNEAL_ITERS,
        random_seed=SEED,
        cascade=cascade,
    )


def bank_from_result(result: dict, cascade: Cascade, *, family: str) -> Bank:
    anneal = result.get("annealing") or {}
    validation = anneal.get("validation") or {}
    holdout = anneal.get("holdout") or {}
    thresholds = validation.get("thresholds") or holdout.get("thresholds") or {}
    val_acc = validation.get("accuracy")
    val_cost = validation.get("expected_cost")
    hol_acc = holdout.get("accuracy")
    hol_cost = holdout.get("expected_cost")
    return Bank(
        thresholds={str(k): float(v) for k, v in thresholds.items()},
        initial=list(cascade.initial),
        specialized=_specialized_to_json(cascade.specialized),
        metrics={
            "method": "a4",
            "layout_family": family,
            "target_accuracy": TARGET_ACCURACY,
            "annealing_iterations": ANNEAL_ITERS,
            "quantile_points": QUANTILE_POINTS,
            "k1_removed": True,
            "position_thresholds": bool(thresholds)
            and all(SLOT_SEP in str(k) for k in thresholds),
            "validation_accuracy": val_acc,
            "validation_expected_cost_ms": val_cost,
            "validation_feasible": (
                float(val_acc) >= TARGET_ACCURACY if val_acc is not None else None
            ),
            "holdout_accuracy": hol_acc,
            "holdout_expected_cost_ms": hol_cost,
            "holdout_feasible": (
                float(hol_acc) >= TARGET_ACCURACY if hol_acc is not None else None
            ),
        },
    )


def run_a4_nested(outcomes_path: Path) -> Bank:
    """A4: for each layout, SA H; pick best by F."""
    families = a4_layout_families(outcomes_path)
    best: Bank | None = None
    rows: list[dict[str, Any]] = []
    for name, cascade in families:
        print(f"A4 family={name} S={cascade.initial}", flush=True)
        result = run_sa_on_cascade(outcomes_path, cascade)
        bank = bank_from_result(result, cascade, family=name)
        rows.append(
            {
                "layout_family": name,
                "initial": list(cascade.initial),
                "val_acc": bank.metrics.get("validation_accuracy"),
                "val_cost": bank.metrics.get("validation_expected_cost_ms"),
                "holdout_acc": bank.metrics.get("holdout_accuracy"),
                "holdout_cost": bank.metrics.get("holdout_expected_cost_ms"),
                "feasible": bank.metrics.get("validation_feasible"),
            }
        )
        if best is None or _policy_key(
            bank.metrics.get("validation_accuracy"),
            bank.metrics.get("validation_expected_cost_ms"),
        ) < _policy_key(
            best.metrics.get("validation_accuracy"),
            best.metrics.get("validation_expected_cost_ms"),
        ):
            best = bank
    assert best is not None
    best.metrics["family_results"] = rows
    return best


def bank_to_report(bank: Bank, scene: str = "h24") -> dict:
    return {
        "method": "a4",
        "scene": scene,
        "target_accuracy": TARGET_ACCURACY,
        "detector_mode": DETECTOR_MODE,
        "detector_cost_ms": DETECTOR_COST_MS,
        "quantile_points": QUANTILE_POINTS,
        "annealing_iterations": ANNEAL_ITERS,
        "holdout_fraction": HOLDOUT_FRACTION,
        "k1_removed": True,
        "position_thresholds": bank.metrics.get("position_thresholds"),
        "layout": {"initial": bank.initial, "specialized": bank.specialized},
        "thresholds": bank.thresholds,
        "validation": {
            "accuracy": bank.metrics.get("validation_accuracy"),
            "expected_cost_ms": bank.metrics.get("validation_expected_cost_ms"),
            "feasible": bank.metrics.get("validation_feasible"),
        },
        "holdout": {
            "accuracy": bank.metrics.get("holdout_accuracy"),
            "expected_cost_ms": bank.metrics.get("holdout_expected_cost_ms"),
            "feasible": bank.metrics.get("holdout_feasible"),
        },
        "metrics": bank.metrics,
        "algorithm": {
            "name": "A4_nested_joint_approximator",
            "outer": "enumerate discrete layout family",
            "inner": "SA threshold approximator + CD polish",
            "complexity": "O(L * I * N * D)",
            "L_layouts": 4,
            "I_sa_iters": ANNEAL_ITERS,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A4 nested joint layout×SA approximator")
    parser.add_argument(
        "--outcomes",
        type=Path,
        required=True,
        help="empirical_outcomes.pkl path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints/joint_opt/h24"),
    )
    parser.add_argument("--scene", default="h24")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)

    bank = run_a4_nested(args.outcomes)
    report = bank_to_report(bank, scene=args.scene)
    print(
        f"A4 winner family={bank.metrics.get('layout_family')} "
        f"val_acc={bank.metrics.get('validation_accuracy')} "
        f"val_cost={bank.metrics.get('validation_expected_cost_ms')} "
        f"holdout_cost={bank.metrics.get('holdout_expected_cost_ms')}"
    )
    if args.save:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        path = args.out_dir / "a4_nested.json"
        path.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
