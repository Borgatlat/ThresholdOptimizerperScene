"""Optimize and live-benchmark the five agreed h24 Jetson cascade policies.

Run this after ``profile_models_jetson.py`` and
``collect_empirical_outcomes.py --paper-detector``.  Every optimizer uses the
same blocked-per-run 80/20 split.  Layout/threshold selection sees only the
first 80%; live inference runs once over every sample in the final 20%.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

import pandas as pd

from experiments.m3n_vc.checkpoint_paths import (
    file_fingerprint,
    resolve_registry_checkpoint,
)
from experiments.m3n_vc.utils.classifier_registry import ClassifierRegistry


TARGET_97 = 0.97
TARGET_95 = 0.95
DEFAULT_OUTCOMES = Path("checkpoints/empirical_outcomes_h24_jetson_nano.pkl")
DEFAULT_REGISTRY = Path("checkpoints/classifier_registry_jetson_nano.json")
DEFAULT_PROCESSED_DIR = Path("datasets/processed")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_OUTPUT_DIR = Path("checkpoints/jetson_nano_h24_cascade_experiments")
DEFAULT_ITERATIONS = 1_000
DEFAULT_RESTARTS = 10
DEFAULT_QUANTILE_POINTS = 50
DEFAULT_POPULATION_SIZE = 32
DEFAULT_GENERATIONS = 24
DEFAULT_EVALUATION_BUDGET = 512
EXPECTED_RUNS = {"run1", "run3", "run5", "run7", "run9"}
EXPECTED_CANDIDATES = {f"K{index}" for index in range(7)} | {"Kdet"}
EXPECTED_THRESHOLDS = {
    "K0": 0.95,
    "K1": 0.95,
    "K2": 0.90,
    "K3": 0.90,
    "K4": 0.95,
    "K5": 0.95,
    "K6": 0.95,
}


@dataclass(frozen=True)
class LiveExperiment:
    experiment_id: str
    label: str
    target_accuracy: float | None
    optimizer_summary: Path
    saved_policy: str
    output: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _scene_dir(processed_dir: Path) -> Path:
    if (processed_dir / "h24_metadata.parquet").is_file():
        return processed_dir
    nested = processed_dir / "h24"
    if (nested / "h24_metadata.parquet").is_file():
        return nested
    return processed_dir


def validate_jetson_inputs(
    outcomes_path: Path,
    registry_path: Path,
    processed_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, object]:
    """Reject stale/mismatched inputs before a long optimization starts."""
    for path in (outcomes_path, registry_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    scene_dir = _scene_dir(processed_dir)
    for name in (
        "h24_paired_mic_norm.npy",
        "h24_paired_geo_norm.npy",
        "h24_metadata.parquet",
    ):
        path = scene_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = pd.read_pickle(outcomes_path)
    if not isinstance(payload, Mapping):
        raise ValueError("Empirical outcomes must contain a mapping packet.")
    collection = payload.get("collection")
    candidates = payload.get("candidates")
    labels = payload.get("labels")
    if not isinstance(collection, Mapping):
        raise ValueError("Empirical outcomes have no collection provenance.")
    if not isinstance(candidates, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        raise ValueError("Empirical outcomes are missing candidates or labels tables.")
    if collection.get("paper_detector") is not True:
        raise ValueError("Jetson experiments require --paper-detector outcomes.")
    saved_checkpoints = collection.get("model_checkpoints")
    if not isinstance(saved_checkpoints, Mapping):
        raise ValueError(
            "Empirical outcomes have no model checkpoint fingerprints. "
            "Re-run collect_empirical_outcomes.py with the current code."
        )

    registry_sha256 = _sha256(registry_path)
    if collection.get("registry_sha256") != registry_sha256:
        raise ValueError(
            "The empirical outcomes were not collected with the current Jetson registry. "
            "Re-run collect_empirical_outcomes.py after profiling."
        )
    registry_payload = _read_json(registry_path)
    runtime_profile = registry_payload.get("runtime_profile")
    profiled_checkpoints = (
        runtime_profile.get("model_checkpoints")
        if isinstance(runtime_profile, Mapping)
        else None
    )
    if not isinstance(profiled_checkpoints, Mapping):
        raise ValueError(
            "Jetson registry has no profiled checkpoint fingerprints. "
            "Re-run profile_models_jetson.py with the current code."
        )
    run_ids = set(labels["run_id"].astype(str))
    if run_ids != EXPECTED_RUNS:
        raise ValueError(
            f"Expected evaluation runs {sorted(EXPECTED_RUNS)}, "
            f"found {sorted(run_ids)}."
        )
    candidate_ids = set(candidates["id"].astype(str))
    if candidate_ids != EXPECTED_CANDIDATES:
        raise ValueError(
            f"Expected candidates {sorted(EXPECTED_CANDIDATES)}, found {sorted(candidate_ids)}."
        )

    indexed = candidates.set_index("id")
    for model_id, expected in EXPECTED_THRESHOLDS.items():
        actual = float(indexed.loc[model_id, "threshold"])
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{model_id} threshold is {actual}, expected {expected}.")
        cost = float(indexed.loc[model_id, "cost"])
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"{model_id} has invalid profiled cost {cost}.")
    detector_cost = float(indexed.loc["Kdet", "cost"])
    if not math.isclose(detector_cost, 10_000.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Kdet cost is {detector_cost}, expected 10000 ms.")

    registry = ClassifierRegistry.load(registry_path)
    current_checkpoints: dict[str, dict[str, str | int]] = {}
    for model_id in sorted(EXPECTED_THRESHOLDS):
        record = registry.get(model_id)
        if record is None:
            raise ValueError(f"Jetson registry has no record for {model_id}.")
        checkpoint_path = resolve_registry_checkpoint(
            record.checkpoint,
            model_id,
            checkpoint_dir,
            registry_path=registry_path,
        )
        current = file_fingerprint(checkpoint_path)
        saved = saved_checkpoints.get(model_id)
        if not isinstance(saved, Mapping):
            raise ValueError(f"Empirical outcomes have no {model_id} checkpoint hash.")
        profiled = profiled_checkpoints.get(model_id)
        if not isinstance(profiled, Mapping):
            raise ValueError(f"Jetson profile has no {model_id} checkpoint hash.")
        if (
            saved.get("sha256") != current["sha256"]
            or int(saved.get("size_bytes", -1)) != current["size_bytes"]
        ):
            raise ValueError(
                f"{model_id} checkpoint differs from empirical collection. "
                "Restore the original weights or recollect outcomes."
            )
        if (
            profiled.get("sha256") != current["sha256"]
            or int(profiled.get("size_bytes", -1)) != current["size_bytes"]
        ):
            raise ValueError(
                f"{model_id} checkpoint differs from the model latency profile. "
                "Re-run profiling and empirical collection."
            )
        current_checkpoints[model_id] = current

    return {
        "outcomes": str(outcomes_path.resolve()),
        "outcomes_sha256": _sha256(outcomes_path),
        "registry": str(registry_path.resolve()),
        "registry_sha256": registry_sha256,
        "processed_dir": str(scene_dir.resolve()),
        "samples": int(len(labels)),
        "runs": sorted(run_ids),
        "model_checkpoints": current_checkpoints,
        "candidate_cost_ms": {
            model_id: float(indexed.loc[model_id, "cost"])
            for model_id in sorted(candidate_ids)
        },
    }


def _python_module(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in arguments)]


def build_commands(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, list[str]]], list[LiveExperiment]]:
    output_dir = args.output_dir
    dp_97 = output_dir / "dp_target_097.json"
    dp_95 = output_dir / "dp_target_095.json"
    ga_97 = output_dir / "ga_target_097"
    ga_95 = output_dir / "ga_target_095"

    optimization_commands: list[tuple[str, list[str]]] = []
    for suffix, target, dp_output, ga_output in (
        ("097", TARGET_97, dp_97, ga_97),
        ("095", TARGET_95, dp_95, ga_95),
    ):
        optimization_commands.append(
            (
                f"dp_threshold_target_{suffix}",
                _python_module(
                    "experiments.m3n_vc.benchmark_full_candidate_dp_sa",
                    "--outcomes",
                    args.outcomes,
                    "--output",
                    dp_output,
                    "--target-accuracy",
                    target,
                    "--iterations",
                    args.iterations,
                    "--restarts",
                    args.restarts,
                    "--seed",
                    args.seed,
                    "--skip-linear",
                ),
            )
        )
        ga_command = _python_module(
            "experiments.m3n_vc.joint_optimize_hierarchy_ga_with_k1",
            "--outcomes",
            args.outcomes,
            "--output-dir",
            ga_output,
            "--target-accuracy",
            target,
            "--iterations",
            args.iterations,
            "--restarts",
            args.restarts,
            "--quantile-points",
            args.quantile_points,
            "--inner-seed",
            args.seed,
            "--split-seed",
            args.seed,
            "--outer-seed",
            args.seed,
            "--population-size",
            args.population_size,
            "--generations",
            args.generations,
            "--evaluation-budget",
            args.evaluation_budget,
            "--workers",
            1,
        )
        if args.overwrite:
            ga_command.append("--overwrite")
        optimization_commands.append((f"ga_joint_target_{suffix}", ga_command))

    live_dir = output_dir / "live"
    live_experiments = [
        LiveExperiment(
            "fixed_threshold_dp",
            "Fixed Threshold Layout",
            None,
            dp_97,
            "dp_fixed_thresholds",
            live_dir / "fixed_threshold_dp.json",
        ),
        LiveExperiment(
            "dp_threshold_target_097",
            "Fixed Threshold Layout (Optimized Thresholds, 97%)",
            TARGET_97,
            dp_97,
            "sa_on_dp_layout",
            live_dir / "dp_threshold_target_097.json",
        ),
        LiveExperiment(
            "ga_joint_target_097",
            "Genetic Joint Optimizer Layout (97%)",
            TARGET_97,
            ga_97 / "summary.json",
            "winner",
            live_dir / "ga_joint_target_097.json",
        ),
        LiveExperiment(
            "dp_threshold_target_095",
            "Fixed Threshold Layout (Optimized Thresholds, 95%)",
            TARGET_95,
            dp_95,
            "sa_on_dp_layout",
            live_dir / "dp_threshold_target_095.json",
        ),
        LiveExperiment(
            "ga_joint_target_095",
            "Genetic Joint Optimizer Layout (95%)",
            TARGET_95,
            ga_95 / "summary.json",
            "winner",
            live_dir / "ga_joint_target_095.json",
        ),
    ]
    return optimization_commands, live_experiments


def _live_command(args: argparse.Namespace, experiment: LiveExperiment) -> list[str]:
    return _python_module(
        "experiments.m3n_vc.live_cascade_benchmark",
        "--summary",
        experiment.optimizer_summary,
        "--saved-policy",
        experiment.saved_policy,
        "--outcomes",
        args.outcomes,
        "--processed-dir",
        args.processed_dir,
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--registry",
        args.registry,
        "--device",
        args.device,
        "--partition",
        "holdout",
        "--warmup-iterations",
        args.warmup_iterations,
        "--max-samples",
        0,
        "--seed",
        args.seed,
        "--detector-cost-ms",
        10_000,
        "--output",
        experiment.output,
    )


def build_live_commands(
    args: argparse.Namespace,
    experiments: Sequence[LiveExperiment],
) -> list[tuple[str, list[str]]]:
    return [
        (f"live_{experiment.experiment_id}", _live_command(args, experiment))
        for experiment in experiments
    ]


def _run_command(name: str, command: Sequence[str], repo_root: Path) -> float:
    print(f"\n[{name}] {shlex.join(command)}", flush=True)
    started = perf_counter()
    subprocess.run(list(command), cwd=repo_root, check=True)
    return perf_counter() - started


def _select_optimizer_packet(
    summary: Mapping[str, object], saved_policy: str
) -> tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    if saved_policy == "winner":
        container = summary["winner"]
        if not isinstance(container, Mapping):
            raise ValueError("GA summary winner is malformed.")
        layout = container["layout"]
    else:
        methods = summary["methods"]
        if not isinstance(methods, Mapping):
            raise ValueError("DP summary methods are malformed.")
        container = methods[saved_policy]
        if not isinstance(container, Mapping):
            raise ValueError(f"DP method {saved_policy!r} is malformed.")
        layout = container.get("layout", summary["layout"])
    validation = container["validation"]
    testing = container["holdout"]
    if not all(isinstance(value, Mapping) for value in (layout, validation, testing)):
        raise ValueError("Optimizer layout/validation/testing packet is malformed.")
    return layout, validation, testing


def _compact_live(report: Mapping[str, object]) -> dict[str, object]:
    benchmark = report["benchmark"]
    if not isinstance(benchmark, Mapping):
        raise ValueError("Live benchmark report is malformed.")
    return {
        "samples": report["loaded_samples"],
        "accuracy": benchmark["accuracy"],
        "correct": benchmark["correct"],
        "costs": benchmark["costs"],
        "measured_wall_latency": benchmark["measured_wall_latency"],
        "measured_model_latency": benchmark["measured_model_latency"],
        "routing": benchmark["routing"],
        "per_class_accuracy": benchmark["per_class_accuracy"],
    }


def _build_summary(
    args: argparse.Namespace,
    inputs: Mapping[str, object],
    live_experiments: Sequence[LiveExperiment],
    durations: Mapping[str, float],
) -> dict[str, object]:
    experiments: list[dict[str, object]] = []
    for experiment in live_experiments:
        optimizer_summary = _read_json(experiment.optimizer_summary)
        live_report = _read_json(experiment.output)
        layout, validation, testing = _select_optimizer_packet(
            optimizer_summary, experiment.saved_policy
        )
        experiments.append(
            {
                "id": experiment.experiment_id,
                "label": experiment.label,
                "target_accuracy": experiment.target_accuracy,
                "optimizer_summary": str(experiment.optimizer_summary.resolve()),
                "optimizer_summary_sha256": _sha256(experiment.optimizer_summary),
                "saved_policy": experiment.saved_policy,
                "layout": layout,
                "thresholds": validation.get("thresholds"),
                "active_slots": validation.get("active_slots"),
                "empirical_validation": validation,
                "empirical_testing": testing,
                "live_report": str(experiment.output.resolve()),
                "live_report_sha256": _sha256(experiment.output),
                "live_testing": _compact_live(live_report),
            }
        )
    return {
        "schema_version": "jetson-five-cascade-experiments/v1",
        "status": "complete",
        "conditions": {
            "targets": [TARGET_97, TARGET_95],
            "classifiers": [f"K{index}" for index in range(7)],
            "k1_included": True,
            "paper_detector_cost_ms": 10_000.0,
            "split": "first 80% validation / final 20% testing within run1,3,5,7,9",
            "threshold_optimizer": f"best_of_{args.restarts}_chellapilla_continuous_gaussian_sa",
            "iterations_per_restart": args.iterations,
            "restarts": args.restarts,
            "quantile_points_compatibility_argument": args.quantile_points,
            "ga_population_size": args.population_size,
            "ga_generations": args.generations,
            "ga_evaluation_budget": args.evaluation_budget,
            "ga_workers": 1,
            "live_batch_size": 1,
            "live_testing_repetitions": 1,
            "live_warmup_iterations_per_reachable_model": args.warmup_iterations,
            "live_device_request": args.device,
            "canonical_requested_conditions": bool(
                args.iterations == DEFAULT_ITERATIONS
                and args.restarts == DEFAULT_RESTARTS
                and args.quantile_points == DEFAULT_QUANTILE_POINTS
                and args.population_size == DEFAULT_POPULATION_SIZE
                and args.generations == DEFAULT_GENERATIONS
                and args.evaluation_budget == DEFAULT_EVALUATION_BUDGET
                and args.warmup_iterations == 25
                and args.seed == 0
                and args.device == "auto"
            ),
        },
        "inputs": dict(inputs),
        "stage_completion_seconds_this_invocation": dict(durations),
        "experiments": experiments,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--restarts", type=int, default=DEFAULT_RESTARTS)
    parser.add_argument("--quantile-points", type=int, default=DEFAULT_QUANTILE_POINTS)
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--evaluation-budget", type=int, default=DEFAULT_EVALUATION_BUDGET)
    parser.add_argument("--warmup-iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact subprocess plan without requiring Jetson artifacts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    positive = (
        args.iterations,
        args.restarts,
        args.quantile_points,
        args.population_size,
        args.generations,
        args.evaluation_budget,
    )
    if any(value < 1 for value in positive) or args.warmup_iterations < 0:
        raise ValueError("Optimizer sizes must be positive and warmup non-negative.")
    if not args.population_size <= args.evaluation_budget:
        raise ValueError("evaluation-budget must be at least population-size.")

    repo_root = Path(__file__).resolve().parents[2]
    optimization_commands, live_experiments = build_commands(args)
    live_commands = build_live_commands(args, live_experiments)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "optimization": [
                        {"name": name, "command": command}
                        for name, command in optimization_commands
                    ],
                    "live": [
                        {"name": name, "command": command}
                        for name, command in live_commands
                    ],
                },
                indent=2,
            )
        )
        return

    inputs = validate_jetson_inputs(
        args.outcomes,
        args.registry,
        args.processed_dir,
        args.checkpoint_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    durations: dict[str, float] = {}
    state_path = args.output_dir / "pipeline_state.json"
    for name, command in (*optimization_commands, *live_commands):
        _write_json_atomic(
            state_path,
            {
                "status": "running",
                "current_stage": name,
                "completed_stages": list(durations),
                "stage_completion_seconds_this_invocation": durations,
            },
        )
        try:
            durations[name] = _run_command(name, command, repo_root)
        except subprocess.CalledProcessError as error:
            _write_json_atomic(
                state_path,
                {
                    "status": "failed",
                    "failed_stage": name,
                    "return_code": error.returncode,
                    "completed_stages": list(durations),
                    "stage_completion_seconds_this_invocation": durations,
                },
            )
            raise

    summary = _build_summary(args, inputs, live_experiments, durations)
    summary_path = args.output_dir / "summary.json"
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(
        state_path,
        {
            "status": "complete",
            "completed_stages": list(durations),
            "stage_completion_seconds_this_invocation": durations,
            "summary": str(summary_path.resolve()),
        },
    )
    print(json.dumps({"status": "complete", "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
