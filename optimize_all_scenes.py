"""Optimize every available scene with the paper Kdet and a baseline target.

Each scene's target is its frozen baseline validation accuracy. This makes the
experiment answer: "can threshold tuning reduce expected cost without doing
worse than the baseline on the data used to select the policy?"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from traceback import format_exc

from threshold_optimizer import (
    DEFAULT_MAX_EXHAUSTIVE_COMBINATIONS,
    DEFAULT_QUANTILE_POINTS,
    optimize_and_evaluate_holdout,
)


ALL_SCENES = ("h24", "h08", "s31", "a06", "i29", "i22")
DEFAULT_OUTCOMES_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/paper_kdet_baseline_target")


def outcome_path_for_scene(outcomes_dir: Path, scene: str) -> Path:
    if scene == "h24":
        return outcomes_dir / "empirical_outcomes.pkl"
    return outcomes_dir / f"empirical_outcomes_{scene}.pkl"


def optimize_all_scenes(
    scenes: tuple[str, ...] | list[str] = ALL_SCENES,
    *,
    outcomes_dir: Path = DEFAULT_OUTCOMES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    method: str = "anneal",
    holdout_fraction: float = 0.20,
    split_strategy: str = "blocked_per_run",
    quantile_points: int | None = DEFAULT_QUANTILE_POINTS,
    max_combinations: int = DEFAULT_MAX_EXHAUSTIVE_COMBINATIONS,
    iterations: int = 10_000,
    random_seed: int = 0,
) -> dict:
    """Run independent paper-Kdet, baseline-target experiments per scene.

    Missing outcome files are recorded as ``skipped`` so a missing scene (for
    example i22) does not discard completed reports for the other scenes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    existing_scenes: dict[str, object] = {}
    if summary_path.is_file():
        existing_summary = json.loads(summary_path.read_text())
        if isinstance(existing_summary, dict) and isinstance(existing_summary.get("scenes"), dict):
            existing_scenes = dict(existing_summary["scenes"])

    summary: dict[str, object] = {
        "detector_mode": "paper",
        "target_accuracy_source": "baseline_validation",
        "method": method,
        "scenes": existing_scenes,
    }

    for scene in scenes:
        input_path = outcome_path_for_scene(outcomes_dir, scene)
        scene_summary: dict[str, object] = {"outcomes": str(input_path)}
        if not input_path.is_file():
            scene_summary["status"] = "skipped"
            scene_summary["reason"] = "empirical outcomes file not found"
            summary["scenes"][scene] = scene_summary  # type: ignore[index]
            print(f"[{scene}] skipped: {input_path} does not exist")
            continue

        output_path = output_dir / f"{scene}.json"
        print(f"[{scene}] optimizing {input_path} -> {output_path}")
        try:
            result = optimize_and_evaluate_holdout(
                input_path,
                target_accuracy=None,
                method=method,
                detector_mode="paper",
                holdout_fraction=holdout_fraction,
                split_strategy=split_strategy,
                quantile_points=quantile_points,
                max_combinations=max_combinations,
                annealing_iterations=iterations,
                random_seed=random_seed,
            )
            result["scene"] = scene
            result["outcomes_path"] = str(input_path)
            output_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=float) + "\n")
            scene_summary.update(
                {
                    "status": "ok",
                    "report": str(output_path),
                    "target_accuracy": result["target_accuracy"],
                }
            )
        except Exception as error:
            scene_summary.update(
                {
                    "status": "failed",
                    "error": str(error),
                    "traceback": format_exc(),
                }
            )
            print(f"[{scene}] failed: {error}")
        summary["scenes"][scene] = scene_summary  # type: ignore[index]

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize cached empirical outcomes for all scenes with the paper Kdet."
    )
    parser.add_argument("--scenes", nargs="+", default=ALL_SCENES)
    parser.add_argument("--outcomes-dir", type=Path, default=DEFAULT_OUTCOMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--method",
        choices=("anneal", "exhaustive", "benchmark"),
        default="anneal",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument(
        "--split-strategy",
        choices=("blocked_per_run", "random_per_run"),
        default="blocked_per_run",
    )
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--all-observed-thresholds", action="store_true")
    parser.add_argument("--max-combinations", type=int, default=DEFAULT_MAX_EXHAUSTIVE_COMBINATIONS)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    optimize_all_scenes(
        list(args.scenes),
        outcomes_dir=args.outcomes_dir,
        output_dir=args.output_dir,
        method=args.method,
        holdout_fraction=args.holdout_fraction,
        split_strategy=args.split_strategy,
        quantile_points=None if args.all_observed_thresholds else args.quantile_points,
        max_combinations=args.max_combinations,
        iterations=args.iterations,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
