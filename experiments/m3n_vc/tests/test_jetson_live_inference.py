from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import torch

from experiments.m3n_vc.checkpoint_paths import file_fingerprint
from experiments.m3n_vc.live_cascade_benchmark import (
    FrozenLayout,
    LiveCascade,
    load_saved_policy,
    load_winner_policy,
)
from experiments.m3n_vc.run_jetson_cascade_experiments import (
    build_commands,
    build_live_commands,
    build_parser,
    validate_jetson_inputs,
)
from experiments.m3n_vc.utils.classifier_registry import ClassifierRegistry


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ClassifierRegistry.load(ROOT / "checkpoints/classifier_registry.json")


class ConstantLogits(torch.nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("result", torch.tensor([logits], dtype=torch.float32))

    def forward(self, *_inputs: torch.Tensor) -> torch.Tensor:
        return self.result


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros((1, 1, 4, 4)), torch.zeros((1, 1, 4, 4))


def test_loads_current_ga_winner_with_position_specific_policy() -> None:
    path = (
        ROOT
        / "checkpoints/k1_including_h24_with_run9_target_095_paper_sa/ga/summary.json"
    )
    summary = json.loads(path.read_text())
    layout, thresholds, active_slots, policy = load_winner_policy(summary, "holdout")

    assert layout.initial == ("K3", "K0", "K1", "detector")
    assert layout.specialized[("K1", "suv")] == ("K2", "K4", "detector")
    assert thresholds["K2@specialized[K1:suv][0]"] == policy["thresholds"][
        "K2@specialized[K1:suv][0]"
    ]
    assert "K4" in active_slots


def test_loads_dp_fixed_and_optimized_policies() -> None:
    path = (
        ROOT
        / "checkpoints/k1_including_h24_with_run9_dp_target_paper_sa/"
        "dp_layout_threshold_optimization.json"
    )
    summary = json.loads(path.read_text())

    for saved_policy in ("dp_fixed_thresholds", "sa_on_dp_layout"):
        layout, thresholds, active_slots, packet = load_saved_policy(
            summary, saved_policy, "holdout"
        )
        validation = summary["methods"][saved_policy]["validation"]
        assert list(layout.initial) == summary["layout"]["initial"]
        assert thresholds == validation["thresholds"]
        assert active_slots == tuple(validation["active_slots"])
        assert packet == summary["methods"][saved_policy]["holdout"]


def test_explicit_empty_active_slots_remain_empty() -> None:
    summary = {
        "layout": {"initial": ["K3", "detector"], "specialized": {}},
        "methods": {
            "dp_fixed_thresholds": {
                "validation": {
                    "thresholds": {"K3@initial[0]": 0.95},
                    "active_slots": [],
                },
                "holdout": {
                    "thresholds": {"K3@initial[0]": 0.95},
                    "active_slots": ["K3@initial[0]"],
                }
            }
        },
    }
    _, _, active_slots, _ = load_saved_policy(
        summary, "dp_fixed_thresholds", "holdout"
    )
    assert active_slots == ()


def test_five_experiment_pipeline_defaults_are_frozen() -> None:
    args = build_parser().parse_args([])
    optimization, live = build_commands(args)

    assert [name for name, _ in optimization] == [
        "dp_threshold_target_097",
        "ga_joint_target_097",
        "dp_threshold_target_095",
        "ga_joint_target_095",
    ]
    assert [experiment.saved_policy for experiment in live] == [
        "dp_fixed_thresholds",
        "sa_on_dp_layout",
        "winner",
        "sa_on_dp_layout",
        "winner",
    ]
    assert [experiment.target_accuracy for experiment in live] == [
        None,
        0.97,
        0.97,
        0.95,
        0.95,
    ]
    ga_commands = [command for name, command in optimization if name.startswith("ga_")]
    assert all(command[command.index("--workers") + 1] == "1" for command in ga_commands)
    assert all(command[command.index("--iterations") + 1] == "1000" for command in ga_commands)
    assert all(command[command.index("--restarts") + 1] == "10" for command in ga_commands)
    stage_names = [name for name, _ in optimization] + [
        name for name, _ in build_live_commands(args, live)
    ]
    assert len(stage_names) == len(set(stage_names)) == 9


def test_pipeline_validates_registry_collection_and_checkpoint_identity() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        processed = root / "processed"
        processed.mkdir()
        for name in (
            "h24_paired_mic_norm.npy",
            "h24_paired_geo_norm.npy",
            "h24_metadata.parquet",
        ):
            (processed / name).write_bytes(b"test-placeholder")

        registry_payload = json.loads((ROOT / "checkpoints/classifier_registry.json").read_text())
        checkpoints = {
            f"K{index}": file_fingerprint(ROOT / f"checkpoints/K{index}.pt")
            for index in range(7)
        }
        registry_payload["runtime_profile"] = {
            "model_checkpoints": checkpoints,
        }
        registry_path = root / "classifier_registry_jetson_nano.json"
        registry_path.write_text(json.dumps(registry_payload))

        candidates = pd.DataFrame(
            [
                {
                    "id": model_id,
                    "threshold": 0.90 if model_id in {"K2", "K3"} else 0.95,
                    "cost": float(index + 1),
                }
                for index, model_id in enumerate(sorted(checkpoints))
            ]
            + [{"id": "Kdet", "threshold": float("nan"), "cost": 10_000.0}]
        )
        outcomes_path = root / "empirical.pkl"
        pd.to_pickle(
            {
                "collection": {
                    "paper_detector": True,
                    "registry_sha256": file_fingerprint(registry_path)["sha256"],
                    "model_checkpoints": checkpoints,
                },
                "candidates": candidates,
                "labels": pd.DataFrame(
                    {"run_id": ["run1", "run3", "run5", "run7", "run9"]}
                ),
            },
            outcomes_path,
        )

        validated = validate_jetson_inputs(
            outcomes_path,
            registry_path,
            processed,
            ROOT / "checkpoints",
        )
        assert set(validated["model_checkpoints"]) == set(checkpoints)


def test_repeated_model_slots_use_independent_thresholds() -> None:
    layout = FrozenLayout(
        initial=("K3", "K3", "detector"),
        specialized={},
    )
    cascade = LiveCascade(
        layout,
        {
            "K3@initial[0]": 0.9999,
            "K3@initial[1]": 0.10,
        },
        {"K3": ConstantLogits([8.0, 0.0, 0.0, 0.0, 0.0])},
        REGISTRY,
        torch.device("cpu"),
    )
    mic, geo = _inputs()
    result = cascade.run(mic, geo, "miata")

    assert result.prediction == "gle350"
    assert result.terminal_route == "K3@initial[1]"
    assert result.route_path == ("K3@initial[0]", "K3@initial[1]")
    assert len(result.invocations) == 2


def test_active_slots_skip_pruned_occurrences() -> None:
    layout = FrozenLayout(
        initial=("K3", "K3", "detector"),
        specialized={},
    )
    cascade = LiveCascade(
        layout,
        {
            "K3@initial[0]": 0.9999,
            "K3@initial[1]": 0.10,
        },
        {"K3": ConstantLogits([8.0, 0.0, 0.0, 0.0, 0.0])},
        REGISTRY,
        torch.device("cpu"),
        active_slots=("K3@initial[1]",),
    )
    mic, geo = _inputs()
    result = cascade.run(mic, geo, "miata")

    assert result.route_path == ("K3@initial[1]",)
    assert len(result.invocations) == 1


def test_router_branch_and_non_sleeping_oracle_detector() -> None:
    layout = FrozenLayout(
        initial=("K0", "detector"),
        specialized={("K0", "coupe"): ("K5", "detector")},
    )
    cascade = LiveCascade(
        layout,
        {"K0": 0.10, "K5": 0.9999},
        {
            "K0": ConstantLogits([0.0, 8.0, 0.0]),
            "K5": ConstantLogits([8.0, 0.0]),
        },
        REGISTRY,
        torch.device("cpu"),
        detector_cost_ms=10_000.0,
    )
    mic, geo = _inputs()
    result = cascade.run(mic, geo, "miata")

    assert result.prediction == "miata"
    assert result.terminal_route == "detector"
    assert result.route_path == ("K0", "K5", "detector")
    assert result.synthetic_detector_ms == 10_000.0
    assert result.measured_wall_ms < result.synthetic_detector_ms
