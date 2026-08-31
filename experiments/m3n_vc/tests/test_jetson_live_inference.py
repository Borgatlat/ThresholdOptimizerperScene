from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.m3n_vc.live_cascade_benchmark import (
    FrozenLayout,
    LiveCascade,
    load_winner_policy,
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
