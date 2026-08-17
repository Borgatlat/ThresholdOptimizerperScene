from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cascade_profile import HierarchyProfile
from empirical_outcomes import (
    EvaluationSplit,
    PredictionBatch,
    PretrainedClassifier,
    ensure_empirical_outcomes,
)
from hierarchy_optimizer import HierarchyOptimizer
from joint_optimize_hierarchy_ga import MemeticSearchConfig, run_memetic_search
from layout_search import (
    LayoutSpace,
    TopologyGenome,
    cascade_from_genome,
    repair_genome,
)
from optimization_pipeline import optimize_joint_from_outcomes
from result_packets import create_result_packet, load_result_packet, write_result_packet
from threshold_optimizer import FixedLayoutThresholdEvaluator, split_empirical_outcomes


def cifar_profile() -> HierarchyProfile:
    groups = {
        f"superclass_{group}": tuple(
            f"class_{group}_{leaf}" for leaf in range(5)
        )
        for group in range(20)
    }
    return HierarchyProfile(
        dataset_id="cifar100/test",
        global_classes=tuple(
            class_name for class_names in groups.values() for class_name in class_names
        ),
        groups=groups,
        router_outputs=tuple(groups),
    )


def predictor(values, confidence=0.99):
    return lambda split: PredictionBatch(
        predictions=values[split.name],
        confidence=[confidence] * len(split.true_labels),
    )


class GenericCifarArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = cifar_profile()
        cls.splits = [
            EvaluationSplit(
                "validation",
                inputs=None,
                true_labels=("class_0_0", "class_1_0"),
            ),
            EvaluationSplit(
                "test",
                inputs=None,
                true_labels=("class_0_1", "class_1_1"),
            ),
        ]
        true_values = {
            split.name: list(split.true_labels) for split in cls.splits
        }
        router_values = {
            "validation": ["superclass_0", "superclass_1"],
            "test": ["superclass_0", "superclass_1"],
        }
        classifiers = [
            PretrainedClassifier(
                id="router_a",
                kind="identifier",
                predict=predictor(router_values),
                output_labels=cls.profile.router_outputs,
                expected_cost_ms=1.0,
                threshold=0.8,
            ),
            PretrainedClassifier(
                id="router_b",
                kind="identifier",
                predict=predictor(router_values),
                output_labels=cls.profile.router_outputs,
                expected_cost_ms=1.2,
                threshold=0.8,
            ),
            PretrainedClassifier(
                id="global",
                kind="global",
                predict=predictor(true_values),
                output_labels=cls.profile.global_classes,
                expected_cost_ms=3.0,
                threshold=0.8,
            ),
        ]
        for group, class_names in cls.profile.groups.items():
            specialist_values = {
                split.name: [
                    true_label if true_label in class_names else class_names[0]
                    for true_label in split.true_labels
                ]
                for split in cls.splits
            }
            classifiers.append(
                PretrainedClassifier(
                    id=f"specialist_{group}",
                    kind="specialized",
                    group=group,
                    predict=predictor(specialist_values),
                    output_labels=class_names,
                    expected_cost_ms=2.0,
                    threshold=0.8,
                )
            )
        classifiers.append(
            PretrainedClassifier(
                id="detector_model",
                kind="detector",
                predict=predictor(true_values),
                output_labels=cls.profile.global_classes,
                expected_cost_ms=10.0,
            )
        )
        cls.classifiers = classifiers

    def test_collection_cache_and_predefined_split_are_dataset_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "outcomes.pkl"
            payload = ensure_empirical_outcomes(
                path, self.profile, self.splits, self.classifiers
            )
            cached = ensure_empirical_outcomes(
                path, self.profile, self.splits, self.classifiers
            )
            validation, test, split = split_empirical_outcomes(cached)

            self.assertEqual(payload["schema_version"], "empirical-outcomes/v2")
            self.assertTrue(path.with_suffix(".profile.json").is_file())
            self.assertEqual(split["strategy"], "predefined")
            self.assertEqual(len(validation["labels"]), 2)
            self.assertEqual(len(test["labels"]), 2)

    def test_twenty_superclasses_create_dynamic_branch_modules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = ensure_empirical_outcomes(
                Path(directory) / "outcomes.pkl",
                self.profile,
                self.splits,
                self.classifiers,
            )
            optimizer = HierarchyOptimizer(payload, detector_mode="paper")
            space = LayoutSpace.from_candidates(
                self.profile, payload["candidates"], optimizer.detector_id
            )
            genome = repair_genome(
                TopologyGenome(initial=("router_a", "router_b", "global")),
                space,
            )

            self.assertEqual(len(genome.branches), 2 * 20)
            self.assertEqual(
                {group for _, group, _ in genome.branches},
                set(self.profile.groups),
            )

            routed_branches = tuple(
                (
                    router_id,
                    group,
                    (f"specialist_{group}",),
                )
                for router_id in ("router_a", "router_b")
                for group in self.profile.group_ids
            )
            cascade = cascade_from_genome(
                TopologyGenome(("router_a", "router_b", "global"), routed_branches),
                space,
            )
            evaluator = FixedLayoutThresholdEvaluator(optimizer, cascade)
            self.assertEqual(len(evaluator.threshold_slots), 2 + 1 + 40)

    def test_generic_ga_does_not_assume_coupe_or_suv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = ensure_empirical_outcomes(
                Path(directory) / "outcomes.pkl",
                self.profile,
                self.splits,
                self.classifiers,
            )
            optimizer = HierarchyOptimizer(payload, detector_mode="paper")
            space = LayoutSpace.from_candidates(
                self.profile, payload["candidates"], optimizer.detector_id
            )

            def evaluate(genome):
                size = len(genome.initial) + sum(
                    len(chain) for _, _, chain in genome.branches
                )
                return {
                    "validation": {
                        "accuracy": 1.0,
                        "expected_cost": float(size + 1),
                    }
                }

            result = run_memetic_search(
                space,
                evaluate,
                target_accuracy=0.9,
                config=MemeticSearchConfig(
                    population_size=6,
                    generations=2,
                    evaluation_budget=10,
                    random_seed=3,
                ),
            )
            self.assertGreaterEqual(len(result.records), 6)

    def test_standard_result_packet_contains_required_figure_data(self) -> None:
        metrics = {
            "accuracy": 0.95,
            "expected_cost": 12.5,
            "total": 4,
            "route_counts": {"global": 3, "detector": 1},
            "thresholds": {"global": 0.8},
        }
        packet = create_result_packet(
            profile=self.profile,
            method_id="test",
            method_label="Test",
            target_accuracy=0.9,
            layout={"initial": ["global"], "branches": {}},
            validation=metrics,
            test=metrics,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_result_packet(packet, Path(directory) / "result.json")
            loaded = load_result_packet(path)
        test_metrics = loaded["partitions"]["test"]
        self.assertEqual(test_metrics["routes"], {"detector": 1, "global": 3})
        self.assertEqual(test_metrics["expected_cost_ms"], 12.5)

    def test_end_to_end_pipeline_writes_standard_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            outcomes_path = directory_path / "outcomes.pkl"
            result_path = directory_path / "result.json"
            ensure_empirical_outcomes(
                outcomes_path,
                self.profile,
                self.splits,
                self.classifiers,
            )
            _, packet = optimize_joint_from_outcomes(
                outcomes_path,
                target_accuracy=0.9,
                result_path=result_path,
                search_config=MemeticSearchConfig(
                    population_size=1,
                    generations=1,
                    evaluation_budget=1,
                    random_seed=0,
                ),
                seeds=(TopologyGenome(()),),
            )

            self.assertTrue(result_path.is_file())
            self.assertEqual(packet["schema_version"], "cascade-result/v1")
            self.assertEqual(packet["partitions"]["test"]["routes"], {"detector": 2})


if __name__ == "__main__":
    unittest.main()
