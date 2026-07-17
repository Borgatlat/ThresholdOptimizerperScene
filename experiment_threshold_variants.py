"""Threshold-optimizer experiments across layouts, targets, detectors, scenes.

This intentionally does NOT train per-scene classifiers or run scene-switching.
It only varies how ``threshold_optimizer.py`` is used:

  1. Cascade structure (DP-optimal vs hand-built layouts)
  2. Target accuracy (baseline / fixed targets)
  3. Detector mode (paper perfect-Kdet vs trained Kdet)
  4. Dataset / scene (cached empirical_outcomes_*.pkl files)
  5. Threshold transfer (tune on h24, evaluate frozen on other scenes)
  6. Search settings (quantile grid size, holdout split strategy)

Usage
-----
    # Full suite (writes under checkpoints/threshold_experiments/)
    python experiment_threshold_variants.py

    # One suite only
    python experiment_threshold_variants.py --suites layouts targets transfer
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from traceback import format_exc
from typing import Callable

from empirical_outcomes import load_empirical_outcomes
from hierarchy_optimizer import Cascade, HierarchyOptimizer, PAPER_DETECTOR_COST_MS
from threshold_optimizer import (
    DEFAULT_QUANTILE_POINTS,
    FixedLayoutThresholdEvaluator,
    optimize_and_evaluate_holdout,
)


ALL_SCENES = ("h24", "h08", "s31", "a06", "i29")
DEFAULT_OUTCOMES_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/threshold_experiments")
DETECTOR = "detector"


def outcome_path_for_scene(outcomes_dir: Path, scene: str) -> Path:
    if scene == "h24":
        return outcomes_dir / "empirical_outcomes.pkl"
    return outcomes_dir / f"empirical_outcomes_{scene}.pkl"


def make_cascade(
    initial: list[str],
    specialized: dict[tuple[str, str], list[str]] | None = None,
) -> Cascade:
    """Build a Cascade object for replay (expected_cost is unused by the tuner)."""
    if initial[-1] != DETECTOR:
        initial = [*initial, DETECTOR]
    specialized = specialized or {}
    normalized_specialized: dict[tuple[str, str], list[str]] = {}
    for key, chain in specialized.items():
        chain_list = list(chain)
        if not chain_list or chain_list[-1] != DETECTOR:
            chain_list = [*chain_list, DETECTOR]
        normalized_specialized[key] = chain_list
    return Cascade(
        expected_cost=0.0,
        initial=initial,
        specialized=normalized_specialized,
        detector=DETECTOR,
    )


# Named layouts for the structure sweep. Think of each as a different
# "circuit diagram" for the same cached classifier confidences.
LAYOUT_BUILDERS: dict[str, Callable[[], Cascade | None]] = {
    # None => let HierarchyOptimizer.synthesize() choose on the validation split.
    "dp_optimal": lambda: None,
    # Global classifiers only (paper baseline family).
    "global_only": lambda: make_cascade(["K2", "K3"]),
    # Single strong global -> detector (minimal 2-stage).
    "single_global": lambda: make_cascade(["K3"]),
    # Classic hierarchy: intermediate router then specialists (no globals).
    "hierarchy_classic": lambda: make_cascade(
        ["K0"],
        {
            ("K0", "suv"): ["K4"],
            ("K0", "coupe"): ["K6"],
        },
    ),
    # 3-stage linear cascade (similar spirit to a basic VisualNet/ImageNet
    # cascade: cheap router -> stronger model -> detector).
    "three_linear": lambda: make_cascade(["K0", "K3"]),
    # Cheap global then expensive global then detector.
    "three_global": lambda: make_cascade(["K2", "K3"]),
    # Intermediate + globals on the trunk, specialists after K0.
    "k0_k2_k3_hierarchy": lambda: make_cascade(
        ["K0", "K2", "K3"],
        {
            ("K0", "suv"): ["K4", "K3"],
            ("K0", "coupe"): ["K6", "K3"],
        },
    ),
    # Deeper initial chain using both identifiers.
    "both_identifiers": lambda: make_cascade(
        ["K0", "K1", "K2", "K3"],
        {
            ("K0", "suv"): ["K4"],
            ("K0", "coupe"): ["K6"],
            ("K1", "suv"): ["K4"],
            ("K1", "coupe"): ["K5", "K6"],
        },
    ),
}


def _compact_holdout(result: dict) -> dict:
    """Keep the fields we usually stare at in a results table."""
    out: dict = {
        "target_accuracy": result.get("target_accuracy"),
        "target_accuracy_source": result.get("target_accuracy_source"),
        "detector_mode": result.get("detector_mode"),
        "split": {
            "strategy": result.get("split", {}).get("strategy"),
            "layout_source": result.get("split", {}).get("layout_source"),
            "initial_layout": result.get("split", {}).get("initial_layout"),
            "specialized_layout": result.get("split", {}).get("specialized_layout"),
            "validation_samples": result.get("split", {}).get("validation_samples"),
            "holdout_samples": result.get("split", {}).get("holdout_samples"),
        },
    }
    for key in ("baseline", "annealing", "exhaustive"):
        block = result.get(key)
        if not isinstance(block, dict):
            continue
        out[key] = {
            "validation_accuracy": block.get("validation", {}).get("accuracy"),
            "validation_cost_ms": block.get("validation", {}).get("expected_cost"),
            "holdout_accuracy": block.get("holdout", {}).get("accuracy"),
            "holdout_cost_ms": block.get("holdout", {}).get("expected_cost"),
            "holdout_feasible": block.get("holdout_feasible"),
            "accuracy_gap": block.get("accuracy_gap"),
            "cost_gap_ms": block.get("cost_gap_ms"),
            "thresholds": block.get("validation", {}).get("thresholds")
            or block.get("holdout", {}).get("thresholds"),
            "holdout_route_counts": block.get("holdout", {}).get("route_counts"),
            "holdout_worst_class_accuracy": block.get("holdout", {}).get(
                "worst_class_accuracy"
            ),
            "holdout_macro_accuracy": block.get("holdout", {}).get("macro_accuracy"),
        }
        # Speedup vs baseline on holdout (cost only meaningful when feasible-ish).
        base_cost = out.get("baseline", {}).get("holdout_cost_ms")
        opt_cost = out[key].get("holdout_cost_ms")
        if base_cost and opt_cost and opt_cost > 0:
            out[key]["holdout_speedup_vs_baseline"] = float(base_cost) / float(opt_cost)
    return out


def _run_holdout(
    outcomes: Path,
    *,
    cascade: Cascade | None,
    detector_mode: str,
    target_accuracy: float | None,
    method: str,
    iterations: int,
    quantile_points: int | None,
    holdout_fraction: float,
    split_strategy: str,
    seed: int,
) -> dict:
    return optimize_and_evaluate_holdout(
        outcomes,
        target_accuracy=target_accuracy,
        method=method,
        detector_mode=detector_mode,
        detector_cost_ms=PAPER_DETECTOR_COST_MS,
        holdout_fraction=holdout_fraction,
        split_strategy=split_strategy,
        quantile_points=quantile_points,
        annealing_iterations=iterations,
        random_seed=seed,
        cascade=cascade,
    )


def suite_layouts(
    outcomes_dir: Path,
    output_dir: Path,
    *,
    iterations: int,
    quantile_points: int | None,
    seed: int,
) -> dict:
    """Same scene (h24), same detector, different cascade topologies."""
    scene = "h24"
    outcomes = outcome_path_for_scene(outcomes_dir, scene)
    suite_dir = output_dir / "layouts_h24_paper"
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"suite": "layouts", "scene": scene, "runs": {}}

    for name, builder in LAYOUT_BUILDERS.items():
        cascade = builder()
        print(f"\n[layouts] {name}")
        try:
            result = _run_holdout(
                outcomes,
                cascade=cascade,
                detector_mode="paper",
                target_accuracy=None,  # baseline target
                method="anneal",
                iterations=iterations,
                quantile_points=quantile_points,
                holdout_fraction=0.20,
                split_strategy="blocked_per_run",
                seed=seed,
            )
            result["layout_name"] = name
            path = suite_dir / f"{name}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
            summary["runs"][name] = {"status": "ok", "report": str(path), **_compact_holdout(result)}  # type: ignore[index]
            anneal = summary["runs"][name].get("annealing", {})  # type: ignore[union-attr]
            print(
                f"  holdout acc={anneal.get('holdout_accuracy'):.4f}  "
                f"cost={anneal.get('holdout_cost_ms'):.2f}ms  "
                f"speedup={anneal.get('holdout_speedup_vs_baseline', float('nan')):.2f}x"
            )
        except Exception as error:
            summary["runs"][name] = {"status": "failed", "error": str(error), "traceback": format_exc()}  # type: ignore[index]
            print(f"  FAILED: {error}")

    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    print(f"Wrote {summary_path}")
    return summary


def suite_targets(
    outcomes_dir: Path,
    output_dir: Path,
    *,
    iterations: int,
    quantile_points: int | None,
    seed: int,
) -> dict:
    """DP layout on h24, sweep accuracy targets under both detector modes."""
    outcomes = outcome_path_for_scene(outcomes_dir, "h24")
    suite_dir = output_dir / "targets_h24"
    suite_dir.mkdir(parents=True, exist_ok=True)
    targets: list[tuple[str, float | None]] = [
        ("baseline", None),
        ("acc_0.90", 0.90),
        ("acc_0.95", 0.95),
        ("acc_0.98", 0.98),
    ]
    summary: dict[str, object] = {"suite": "targets", "runs": {}}

    for detector_mode in ("paper", "trained"):
        for label, target in targets:
            name = f"{detector_mode}_{label}"
            print(f"\n[targets] {name}")
            try:
                result = _run_holdout(
                    outcomes,
                    cascade=None,
                    detector_mode=detector_mode,
                    target_accuracy=target,
                    method="anneal",
                    iterations=iterations,
                    quantile_points=quantile_points,
                    holdout_fraction=0.20,
                    split_strategy="blocked_per_run",
                    seed=seed,
                )
                path = suite_dir / f"{name}.json"
                path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
                summary["runs"][name] = {"status": "ok", "report": str(path), **_compact_holdout(result)}  # type: ignore[index]
                anneal = summary["runs"][name].get("annealing", {})  # type: ignore[union-attr]
                print(
                    f"  target={result['target_accuracy']:.4f}  "
                    f"holdout acc={anneal.get('holdout_accuracy'):.4f}  "
                    f"cost={anneal.get('holdout_cost_ms'):.2f}ms"
                )
            except Exception as error:
                summary["runs"][name] = {
                    "status": "failed",
                    "error": str(error),
                    "traceback": format_exc(),
                }  # type: ignore[index]
                print(f"  FAILED: {error}")

    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    print(f"Wrote {summary_path}")
    return summary


def suite_scenes_trained(
    outcomes_dir: Path,
    output_dir: Path,
    *,
    iterations: int,
    quantile_points: int | None,
    seed: int,
) -> dict:
    """Per-scene threshold tuning with the *trained* Kdet (paper suite already exists)."""
    suite_dir = output_dir / "scenes_trained_baseline_target"
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"suite": "scenes_trained", "runs": {}}

    for scene in ALL_SCENES:
        outcomes = outcome_path_for_scene(outcomes_dir, scene)
        print(f"\n[scenes_trained] {scene}")
        if not outcomes.is_file():
            summary["runs"][scene] = {"status": "skipped", "reason": f"missing {outcomes}"}  # type: ignore[index]
            print(f"  skipped: {outcomes}")
            continue
        try:
            result = _run_holdout(
                outcomes,
                cascade=None,
                detector_mode="trained",
                target_accuracy=None,
                method="anneal",
                iterations=iterations,
                quantile_points=quantile_points,
                holdout_fraction=0.20,
                split_strategy="blocked_per_run",
                seed=seed,
            )
            path = suite_dir / f"{scene}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
            summary["runs"][scene] = {"status": "ok", "report": str(path), **_compact_holdout(result)}  # type: ignore[index]
            anneal = summary["runs"][scene].get("annealing", {})  # type: ignore[union-attr]
            print(
                f"  holdout acc={anneal.get('holdout_accuracy'):.4f}  "
                f"cost={anneal.get('holdout_cost_ms'):.2f}ms  "
                f"speedup={anneal.get('holdout_speedup_vs_baseline', float('nan')):.2f}x"
            )
        except Exception as error:
            summary["runs"][scene] = {
                "status": "failed",
                "error": str(error),
                "traceback": format_exc(),
            }  # type: ignore[index]
            print(f"  FAILED: {error}")

    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    print(f"Wrote {summary_path}")
    return summary


def suite_transfer(
    outcomes_dir: Path,
    output_dir: Path,
    *,
    iterations: int,
    quantile_points: int | None,
    seed: int,
) -> dict:
    """Tune thresholds on h24, freeze layout+thresholds, evaluate on other scenes.

    Also retunes thresholds on each scene while freezing the *h24 DP layout*
    (structure transfer + local threshold retune). That is NOT scene-switching;
    it asks whether one shared topology can be adapted with thresholds alone.
    """
    suite_dir = output_dir / "transfer_h24_layout"
    suite_dir.mkdir(parents=True, exist_ok=True)
    h24_path = outcome_path_for_scene(outcomes_dir, "h24")

    print("\n[transfer] optimizing source policy on h24 (paper, baseline target)")
    source = _run_holdout(
        h24_path,
        cascade=None,
        detector_mode="paper",
        target_accuracy=None,
        method="anneal",
        iterations=iterations,
        quantile_points=quantile_points,
        holdout_fraction=0.20,
        split_strategy="blocked_per_run",
        seed=seed,
    )
    source_path = suite_dir / "source_h24.json"
    source_path.write_text(json.dumps(source, indent=2, sort_keys=True, default=float) + "\n")

    # Freeze the validation DP layout chosen on h24.
    initial = list(source["split"]["initial_layout"])
    specialized = {
        tuple(key.split(":", 1)): list(chain)
        for key, chain in source["split"]["specialized_layout"].items()
    }
    frozen_cascade = Cascade(
        expected_cost=0.0,
        initial=initial,
        specialized=specialized,  # type: ignore[arg-type]
        detector=DETECTOR,
    )
    frozen_thresholds = source["annealing"]["validation"]["thresholds"]

    summary: dict[str, object] = {
        "suite": "transfer",
        "source_report": str(source_path),
        "frozen_layout": {
            "initial": initial,
            "specialized": source["split"]["specialized_layout"],
        },
        "frozen_thresholds": frozen_thresholds,
        "zero_shot": {},
        "retune_thresholds_on_frozen_layout": {},
    }

    for scene in ALL_SCENES:
        outcomes = outcome_path_for_scene(outcomes_dir, scene)
        if not outcomes.is_file():
            print(f"[transfer] {scene}: skipped (missing outcomes)")
            continue

        print(f"\n[transfer] {scene}")
        # Zero-shot: h24 thresholds + h24 layout, no retuning.
        evaluator = FixedLayoutThresholdEvaluator(
            HierarchyOptimizer(
                load_empirical_outcomes(outcomes),
                detector_mode="paper",
                detector_cost_ms=PAPER_DETECTOR_COST_MS,
            ),
            frozen_cascade,
        )
        # Only keep thresholds for models that exist in this layout.
        usable = {
            cid: float(frozen_thresholds[cid])
            for cid in evaluator.tunable_ids
            if cid in frozen_thresholds
        }
        zero_shot = evaluator.evaluate(usable)
        summary["zero_shot"][scene] = {  # type: ignore[index]
            "accuracy": zero_shot["accuracy"],
            "expected_cost_ms": zero_shot["expected_cost"],
            "macro_accuracy": zero_shot.get("macro_accuracy"),
            "worst_class_accuracy": zero_shot.get("worst_class_accuracy"),
            "route_counts": zero_shot.get("route_counts"),
        }
        print(
            f"  zero-shot acc={zero_shot['accuracy']:.4f}  "
            f"cost={zero_shot['expected_cost']:.2f}ms"
        )

        # Retune thresholds on this scene, but keep the h24 topology frozen.
        try:
            retune = _run_holdout(
                outcomes,
                cascade=frozen_cascade,
                detector_mode="paper",
                target_accuracy=None,
                method="anneal",
                iterations=iterations,
                quantile_points=quantile_points,
                holdout_fraction=0.20,
                split_strategy="blocked_per_run",
                seed=seed,
            )
            path = suite_dir / f"retune_{scene}.json"
            path.write_text(json.dumps(retune, indent=2, sort_keys=True, default=float) + "\n")
            summary["retune_thresholds_on_frozen_layout"][scene] = {  # type: ignore[index]
                "status": "ok",
                "report": str(path),
                **_compact_holdout(retune),
            }
            anneal = summary["retune_thresholds_on_frozen_layout"][scene].get("annealing", {})  # type: ignore[index]
            print(
                f"  retuned holdout acc={anneal.get('holdout_accuracy'):.4f}  "
                f"cost={anneal.get('holdout_cost_ms'):.2f}ms  "
                f"speedup={anneal.get('holdout_speedup_vs_baseline', float('nan')):.2f}x"
            )
        except Exception as error:
            summary["retune_thresholds_on_frozen_layout"][scene] = {  # type: ignore[index]
                "status": "failed",
                "error": str(error),
                "traceback": format_exc(),
            }
            print(f"  retune FAILED: {error}")

    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    print(f"Wrote {summary_path}")
    return summary


def suite_search_settings(
    outcomes_dir: Path,
    output_dir: Path,
    *,
    iterations: int,
    seed: int,
) -> dict:
    """Same layout/target; vary quantile grid density and holdout split style."""
    outcomes = outcome_path_for_scene(outcomes_dir, "h24")
    suite_dir = output_dir / "search_settings_h24"
    suite_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("q10_blocked", 10, "blocked_per_run"),
        ("q25_blocked", 25, "blocked_per_run"),
        ("q50_blocked", 50, "blocked_per_run"),
        ("q100_blocked", 100, "blocked_per_run"),
        ("q50_random", 50, "random_per_run"),
    ]
    summary: dict[str, object] = {"suite": "search_settings", "runs": {}}

    for name, q, split in configs:
        print(f"\n[search_settings] {name}")
        try:
            result = _run_holdout(
                outcomes,
                cascade=None,
                detector_mode="paper",
                target_accuracy=None,
                method="anneal",
                iterations=iterations,
                quantile_points=q,
                holdout_fraction=0.20,
                split_strategy=split,
                seed=seed,
            )
            path = suite_dir / f"{name}.json"
            path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
            summary["runs"][name] = {
                "status": "ok",
                "quantile_points": q,
                "split_strategy": split,
                "report": str(path),
                **_compact_holdout(result),
            }  # type: ignore[index]
            anneal = summary["runs"][name].get("annealing", {})  # type: ignore[union-attr]
            print(
                f"  holdout acc={anneal.get('holdout_accuracy'):.4f}  "
                f"cost={anneal.get('holdout_cost_ms'):.2f}ms"
            )
        except Exception as error:
            summary["runs"][name] = {
                "status": "failed",
                "error": str(error),
                "traceback": format_exc(),
            }  # type: ignore[index]
            print(f"  FAILED: {error}")

    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    print(f"Wrote {summary_path}")
    return summary


def suite_layout_by_scene(
    outcomes_dir: Path,
    output_dir: Path,
    *,
    iterations: int,
    quantile_points: int | None,
    seed: int,
) -> dict:
    """Cross layouts × scenes for a compact subset of interesting topologies."""
    layouts = ("dp_optimal", "global_only", "hierarchy_classic", "three_linear")
    suite_dir = output_dir / "layouts_by_scene_paper"
    suite_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"suite": "layouts_by_scene", "runs": {}}

    for scene in ALL_SCENES:
        outcomes = outcome_path_for_scene(outcomes_dir, scene)
        if not outcomes.is_file():
            continue
        for layout_name in layouts:
            key = f"{scene}__{layout_name}"
            print(f"\n[layouts_by_scene] {key}")
            try:
                result = _run_holdout(
                    outcomes,
                    cascade=LAYOUT_BUILDERS[layout_name](),
                    detector_mode="paper",
                    target_accuracy=None,
                    method="anneal",
                    iterations=iterations,
                    quantile_points=quantile_points,
                    holdout_fraction=0.20,
                    split_strategy="blocked_per_run",
                    seed=seed,
                )
                path = suite_dir / f"{key}.json"
                path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
                summary["runs"][key] = {"status": "ok", "report": str(path), **_compact_holdout(result)}  # type: ignore[index]
                anneal = summary["runs"][key].get("annealing", {})  # type: ignore[union-attr]
                print(
                    f"  holdout acc={anneal.get('holdout_accuracy'):.4f}  "
                    f"cost={anneal.get('holdout_cost_ms'):.2f}ms  "
                    f"speedup={anneal.get('holdout_speedup_vs_baseline', float('nan')):.2f}x"
                )
            except Exception as error:
                summary["runs"][key] = {
                    "status": "failed",
                    "error": str(error),
                    "traceback": format_exc(),
                }  # type: ignore[index]
                print(f"  FAILED: {error}")

    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n")
    print(f"Wrote {summary_path}")
    return summary


def write_master_summary(output_dir: Path, suite_summaries: dict[str, dict]) -> Path:
    rows: list[dict] = []
    for suite_name, summary in suite_summaries.items():
        runs = summary.get("runs")
        if isinstance(runs, dict):
            for run_name, run in runs.items():
                if not isinstance(run, dict) or run.get("status") != "ok":
                    continue
                anneal = run.get("annealing") or {}
                baseline = run.get("baseline") or {}
                rows.append(
                    {
                        "suite": suite_name,
                        "run": run_name,
                        "detector_mode": run.get("detector_mode"),
                        "target_accuracy": run.get("target_accuracy"),
                        "baseline_holdout_acc": baseline.get("holdout_accuracy"),
                        "baseline_holdout_cost_ms": baseline.get("holdout_cost_ms"),
                        "opt_holdout_acc": anneal.get("holdout_accuracy"),
                        "opt_holdout_cost_ms": anneal.get("holdout_cost_ms"),
                        "holdout_speedup_vs_baseline": anneal.get(
                            "holdout_speedup_vs_baseline"
                        ),
                        "holdout_feasible": anneal.get("holdout_feasible"),
                        "layout": run.get("split", {}).get("initial_layout"),
                    }
                )
        # Transfer suite stores nested dicts instead of runs.
        if suite_name == "transfer":
            for scene, zs in summary.get("zero_shot", {}).items():
                rows.append(
                    {
                        "suite": "transfer_zero_shot",
                        "run": scene,
                        "opt_holdout_acc": zs.get("accuracy"),
                        "opt_holdout_cost_ms": zs.get("expected_cost_ms"),
                    }
                )
            for scene, run in summary.get("retune_thresholds_on_frozen_layout", {}).items():
                if not isinstance(run, dict) or run.get("status") != "ok":
                    continue
                anneal = run.get("annealing") or {}
                rows.append(
                    {
                        "suite": "transfer_retune",
                        "run": scene,
                        "opt_holdout_acc": anneal.get("holdout_accuracy"),
                        "opt_holdout_cost_ms": anneal.get("holdout_cost_ms"),
                        "holdout_speedup_vs_baseline": anneal.get(
                            "holdout_speedup_vs_baseline"
                        ),
                    }
                )

    master = {"table": rows, "suites": list(suite_summaries.keys())}
    path = output_dir / "MASTER_SUMMARY.json"
    path.write_text(json.dumps(master, indent=2, sort_keys=True, default=float) + "\n")

    # Also a human-readable markdown table.
    md_lines = [
        "# Threshold Optimizer Experiment Summary",
        "",
        "Scene-switching / per-scene classifier training was **not** run.",
        "",
        "| suite | run | holdout acc | holdout cost (ms) | speedup vs baseline | feasible |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {suite} | {run} | {acc} | {cost} | {speedup} | {feas} |".format(
                suite=row.get("suite"),
                run=row.get("run"),
                acc=_fmt(row.get("opt_holdout_acc")),
                cost=_fmt(row.get("opt_holdout_cost_ms")),
                speedup=_fmt(row.get("holdout_speedup_vs_baseline")),
                feas=row.get("holdout_feasible"),
            )
        )
    md_path = output_dir / "MASTER_SUMMARY.md"
    md_path.write_text("\n".join(md_lines) + "\n")
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
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=(
            "layouts",
            "targets",
            "scenes_trained",
            "transfer",
            "search_settings",
            "layouts_by_scene",
            "all",
        ),
        default=["all"],
    )
    parser.add_argument("--outcomes-dir", type=Path, default=DEFAULT_OUTCOMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iterations", type=int, default=8_000)
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.suites)
    if "all" in selected:
        selected = {
            "layouts",
            "targets",
            "scenes_trained",
            "transfer",
            "search_settings",
            "layouts_by_scene",
        }

    common = dict(
        iterations=args.iterations,
        quantile_points=args.quantile_points,
        seed=args.seed,
    )
    summaries: dict[str, dict] = {}

    if "layouts" in selected:
        summaries["layouts"] = suite_layouts(args.outcomes_dir, args.output_dir, **common)
    if "targets" in selected:
        summaries["targets"] = suite_targets(args.outcomes_dir, args.output_dir, **common)
    if "scenes_trained" in selected:
        summaries["scenes_trained"] = suite_scenes_trained(
            args.outcomes_dir, args.output_dir, **common
        )
    if "transfer" in selected:
        summaries["transfer"] = suite_transfer(args.outcomes_dir, args.output_dir, **common)
    if "search_settings" in selected:
        summaries["search_settings"] = suite_search_settings(
            args.outcomes_dir,
            args.output_dir,
            iterations=args.iterations,
            seed=args.seed,
        )
    if "layouts_by_scene" in selected:
        summaries["layouts_by_scene"] = suite_layout_by_scene(
            args.outcomes_dir, args.output_dir, **common
        )

    write_master_summary(args.output_dir, summaries)


# Re-export for type checkers / accidental imports from experiments.
__all__ = [
    "LAYOUT_BUILDERS",
    "main",
    "make_cascade",
]


if __name__ == "__main__":
    main()
