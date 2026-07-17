"""Per-scene threshold bank experiment.

Goal
----
For each M3N-VC scene with cached empirical outcomes, learn a threshold vector
``{K0..K6: H_i}`` that improves efficiency while protecting accuracy.  This is
the *oracle* bank a future scene detector would look up — we are NOT doing
scene switching here, only building the per-scene thresholds.

Two complementary modes
-----------------------
1. ``per_scene_structure`` — each scene gets its own DP-optimal cascade
   layout *and* its own thresholds (max flexibility).
2. ``shared_h24_structure`` — freeze h24's DP layout for every scene, and
   only retune thresholds (asks: "is threshold adaptation alone enough if
   the wiring stays fixed?").

Outputs
-------
* ``checkpoints/threshold_experiments/per_scene_thresholds/``
  - per-scene JSON reports
  - ``summary.json``
  - ``scene_threshold_bank_paper.json`` / ``scene_threshold_bank_trained.json``
    (schema compatible with IDKCascades' scene_threshold_bank)

Usage
-----
    python experiment_per_scene_thresholds.py
    python experiment_per_scene_thresholds.py --detector-modes paper --iterations 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from traceback import format_exc

from empirical_outcomes import load_empirical_outcomes
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    optimize_and_evaluate_holdout,
)
from utils.labels import KI_REGISTRY, threshold_hi_for_ki


ALL_SCENES = ("h24", "h08", "s31", "a06", "i29")
# i22 is intentionally omitted: no single-vehicle empirical outcomes yet.
ALL_KI = ("K0", "K1", "K2", "K3", "K4", "K5", "K6")
DEFAULT_OUTCOMES_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments/per_scene_thresholds")


def outcome_path_for_scene(outcomes_dir: Path, scene: str) -> Path:
    if scene == "h24":
        return outcomes_dir / "empirical_outcomes.pkl"
    return outcomes_dir / f"empirical_outcomes_{scene}.pkl"


def _default_threshold_bank() -> dict[str, float]:
    """Registry / metrics defaults for every Ki (used to fill unused slots)."""
    return {ki: float(threshold_hi_for_ki(ki)) for ki in ALL_KI if ki in KI_REGISTRY}


def _complete_threshold_vector(optimized: dict[str, float]) -> dict[str, float]:
    """Ensure the bank always has K0..K6, even if a cascade never used K5.

    Why: a scene detector later may switch banks mid-stream; missing keys
    would crash. Filling unused Kis with their original calibration threshold
    is a safe no-op default (that classifier simply keeps old behavior if
    somehow invoked).
    """
    bank = _default_threshold_bank()
    for ki, value in optimized.items():
        bank[ki] = float(value)
    return {ki: bank[ki] for ki in ALL_KI}


def _cascade_from_split(split: dict) -> Cascade:
    initial = list(split["initial_layout"])
    specialized = {
        tuple(key.split(":", 1)): list(chain)
        for key, chain in split["specialized_layout"].items()
    }
    return Cascade(
        expected_cost=0.0,
        initial=initial,
        specialized=specialized,  # type: ignore[arg-type]
        detector="detector",
    )


def _compact(result: dict) -> dict:
    anneal = result.get("annealing") or {}
    baseline = result.get("baseline") or {}
    a_hold = anneal.get("holdout") or {}
    a_val = anneal.get("validation") or {}
    b_hold = baseline.get("holdout") or {}
    base_cost = b_hold.get("expected_cost")
    opt_cost = a_hold.get("expected_cost")
    speedup = (
        float(base_cost) / float(opt_cost)
        if base_cost and opt_cost and float(opt_cost) > 0
        else None
    )
    return {
        "target_accuracy": result.get("target_accuracy"),
        "detector_mode": result.get("detector_mode"),
        "layout": result.get("split", {}).get("initial_layout"),
        "specialized_layout": result.get("split", {}).get("specialized_layout"),
        "baseline_holdout_acc": b_hold.get("accuracy"),
        "baseline_holdout_cost_ms": b_hold.get("expected_cost"),
        "opt_validation_acc": a_val.get("accuracy"),
        "opt_holdout_acc": a_hold.get("accuracy"),
        "opt_holdout_cost_ms": a_hold.get("expected_cost"),
        "opt_holdout_macro_acc": a_hold.get("macro_accuracy"),
        "opt_holdout_worst_class_acc": a_hold.get("worst_class_accuracy"),
        "holdout_feasible": anneal.get("holdout_feasible"),
        "holdout_speedup_vs_baseline": speedup,
        "thresholds": a_val.get("thresholds") or a_hold.get("thresholds"),
        "holdout_route_counts": a_hold.get("route_counts"),
    }


def optimize_scene(
    scene: str,
    outcomes: Path,
    *,
    detector_mode: str,
    cascade: Cascade | None,
    iterations: int,
    quantile_points: int,
    seed: int,
) -> dict:
    return optimize_and_evaluate_holdout(
        outcomes,
        target_accuracy=None,  # protect each scene's own baseline accuracy
        method="anneal",
        detector_mode=detector_mode,
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
        holdout_fraction=0.20,
        split_strategy="blocked_per_run",
        quantile_points=quantile_points,
        annealing_iterations=iterations,
        random_seed=seed,
        cascade=cascade,
    )


def build_h24_frozen_cascade(
    outcomes_dir: Path,
    detector_mode: str,
    iterations: int,
    quantile_points: int,
    seed: int,
) -> tuple[Cascade, dict]:
    """Learn h24's DP layout once; reuse it as the shared structure."""
    h24_path = outcome_path_for_scene(outcomes_dir, "h24")
    result = optimize_scene(
        "h24",
        h24_path,
        detector_mode=detector_mode,
        cascade=None,
        iterations=iterations,
        quantile_points=quantile_points,
        seed=seed,
    )
    return _cascade_from_split(result["split"]), result


def run_mode(
    mode: str,
    *,
    scenes: list[str],
    outcomes_dir: Path,
    output_dir: Path,
    detector_mode: str,
    iterations: int,
    quantile_points: int,
    seed: int,
) -> dict:
    mode_dir = output_dir / f"{mode}_{detector_mode}"
    mode_dir.mkdir(parents=True, exist_ok=True)

    frozen_cascade: Cascade | None = None
    h24_source: dict | None = None
    if mode == "shared_h24_structure":
        print(f"\n[{mode}/{detector_mode}] synthesizing frozen h24 layout...")
        frozen_cascade, h24_source = build_h24_frozen_cascade(
            outcomes_dir, detector_mode, iterations, quantile_points, seed
        )
        (mode_dir / "source_h24.json").write_text(
            json.dumps(h24_source, indent=2, sort_keys=True, default=float) + "\n"
        )
        print(f"  frozen layout: {h24_source['split']['initial_layout']}")

    summary: dict = {
        "mode": mode,
        "detector_mode": detector_mode,
        "target_accuracy_source": "baseline_validation",
        "scenes": {},
    }
    threshold_bank: dict[str, dict[str, float]] = {}

    for scene in scenes:
        outcomes = outcome_path_for_scene(outcomes_dir, scene)
        print(f"\n[{mode}/{detector_mode}] {scene}")
        if not outcomes.is_file():
            summary["scenes"][scene] = {
                "status": "skipped",
                "reason": f"missing {outcomes}",
            }
            print(f"  skipped: {outcomes}")
            continue

        # shared mode: every scene uses h24's wiring; per_scene mode: None => DP.
        cascade = frozen_cascade if mode == "shared_h24_structure" else None
        try:
            # For shared mode on h24 itself, reuse the source result (same split).
            if (
                mode == "shared_h24_structure"
                and scene == "h24"
                and h24_source is not None
            ):
                result = h24_source
            else:
                result = optimize_scene(
                    scene,
                    outcomes,
                    detector_mode=detector_mode,
                    cascade=cascade,
                    iterations=iterations,
                    quantile_points=quantile_points,
                    seed=seed,
                )
            result["scene"] = scene
            result["mode"] = mode
            path = mode_dir / f"{scene}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")

            compact = _compact(result)
            thresholds = compact.get("thresholds") or {}
            bank_row = _complete_threshold_vector(thresholds)
            threshold_bank[scene] = bank_row

            summary["scenes"][scene] = {
                "status": "ok",
                "report": str(path),
                "threshold_bank_row": bank_row,
                **compact,
            }
            print(
                f"  holdout acc={compact['opt_holdout_acc']:.4f}  "
                f"cost={compact['opt_holdout_cost_ms']:.2f}ms  "
                f"speedup={compact['holdout_speedup_vs_baseline'] or float('nan'):.2f}x  "
                f"layout={compact['layout']}"
            )
            print(f"  thresholds={ {k: round(v, 4) for k, v in bank_row.items()} }")
        except Exception as error:
            summary["scenes"][scene] = {
                "status": "failed",
                "error": str(error),
                "traceback": format_exc(),
            }
            print(f"  FAILED: {error}")

    summary_path = mode_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")

    bank_payload = {
        "description": (
            "Per-scene optimized IDK thresholds for M3N-VC. "
            f"mode={mode}, detector_mode={detector_mode}, "
            "target=each scene's baseline validation accuracy."
        ),
        "scene_ids": list(ALL_SCENES) + ["i22"],
        "threshold_bank": threshold_bank,
        # i22 placeholder: no single-vehicle outcomes yet (multi-target scene).
        "notes": (
            "i22 omitted from optimization (no usable single-label outcomes). "
            "Unused Ki slots filled with registry default thresholds. "
            "Keys are scene_id, not sensor_id."
        ),
        "mode": mode,
        "detector_mode": detector_mode,
    }
    # Keep i22 key present so downstream scene-detector code does not KeyError.
    bank_payload["threshold_bank"]["i22"] = {ki: 0.0 for ki in ALL_KI}

    bank_name = f"scene_threshold_bank_{mode}_{detector_mode}.json"
    bank_path = output_dir / bank_name
    # Also write the "primary" paper/per-scene bank under the familiar name.
    bank_path.write_text(json.dumps(bank_payload, indent=2, sort_keys=True) + "\n")
    if mode == "per_scene_structure":
        canonical = output_dir / f"scene_threshold_bank_{detector_mode}.json"
        canonical.write_text(json.dumps(bank_payload, indent=2, sort_keys=True) + "\n")
        # And a copy at checkpoints/ for the schema consumers.
        root_bank = Path("checkpoints") / f"scene_threshold_bank_{detector_mode}.json"
        root_bank.write_text(json.dumps(bank_payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {canonical}")
        print(f"Wrote {root_bank}")

    print(f"Wrote {summary_path}")
    print(f"Wrote {bank_path}")
    return summary


def write_comparison_table(output_dir: Path, summaries: dict[str, dict]) -> Path:
    """Side-by-side table: per-scene structure vs shared h24 structure."""
    rows = []
    for key, summary in summaries.items():
        for scene, run in summary.get("scenes", {}).items():
            if run.get("status") != "ok":
                continue
            rows.append(
                {
                    "experiment": key,
                    "mode": summary.get("mode"),
                    "detector_mode": summary.get("detector_mode"),
                    "scene": scene,
                    "layout": run.get("layout"),
                    "baseline_holdout_acc": run.get("baseline_holdout_acc"),
                    "opt_holdout_acc": run.get("opt_holdout_acc"),
                    "opt_holdout_cost_ms": run.get("opt_holdout_cost_ms"),
                    "holdout_speedup_vs_baseline": run.get("holdout_speedup_vs_baseline"),
                    "holdout_feasible": run.get("holdout_feasible"),
                    "thresholds": run.get("threshold_bank_row"),
                }
            )

    payload = {"table": rows}
    path = output_dir / "COMPARISON.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")

    md = [
        "# Per-Scene Threshold Bank Results",
        "",
        "Oracle thresholds for each scene. No scene-switching was run.",
        "",
        "| experiment | scene | holdout acc | cost (ms) | speedup | feasible | layout |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        layout = "→".join(row["layout"] or [])
        md.append(
            "| {exp} | {scene} | {acc} | {cost} | {spd} | {feas} | `{layout}` |".format(
                exp=row["experiment"],
                scene=row["scene"],
                acc=_fmt(row["opt_holdout_acc"]),
                cost=_fmt(row["opt_holdout_cost_ms"]),
                spd=_fmt(row["holdout_speedup_vs_baseline"]),
                feas=row["holdout_feasible"],
                layout=layout,
            )
        )
    md_path = output_dir / "COMPARISON.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {path}")
    print(f"Wrote {md_path}")
    return path


def _fmt(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", nargs="+", default=list(ALL_SCENES))
    parser.add_argument("--outcomes-dir", type=Path, default=DEFAULT_OUTCOMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("per_scene_structure", "shared_h24_structure"),
        default=("per_scene_structure", "shared_h24_structure"),
    )
    parser.add_argument(
        "--detector-modes",
        nargs="+",
        choices=("paper", "trained"),
        default=("paper", "trained"),
    )
    parser.add_argument("--iterations", type=int, default=8_000)
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    for detector_mode in args.detector_modes:
        for mode in args.modes:
            key = f"{mode}__{detector_mode}"
            summaries[key] = run_mode(
                mode,
                scenes=list(args.scenes),
                outcomes_dir=args.outcomes_dir,
                output_dir=args.output_dir,
                detector_mode=detector_mode,
                iterations=args.iterations,
                quantile_points=args.quantile_points,
                seed=args.seed,
            )

    write_comparison_table(args.output_dir, summaries)


if __name__ == "__main__":
    main()
