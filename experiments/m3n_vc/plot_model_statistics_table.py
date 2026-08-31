"""Render a paper-style table of the M3N-VC classifier statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt

from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc.utils.labels import (
    GLOBAL_CLASS_NAMES,
    INTERMEDIATE_CLASS_NAMES,
    KI_REGISTRY,
    threshold_hi_for_ki,
)


DEFAULT_REGISTRY = Path("checkpoints/classifier_registry.json")
DEFAULT_OUTCOMES = Path("checkpoints/empirical_outcomes_h24_with_run9.pkl")
DEFAULT_OUTPUT = Path("checkpoints/model_statistics_table_h24_with_run9.png")
DEFAULT_STATISTICS_OUTPUT = Path("checkpoints/model_statistics_h24_with_run9.json")
DEFAULT_PROFILE_REPORT = Path("checkpoints/jetson_nano_model_profile.json")
PAPER_KDET_COST_MS = 10_000.0
CLASSIFIER_IDS = tuple([f"K{i}" for i in range(7)] + ["Kdet"])


def _classifier_statistics(
    registry_path: Path,
    outcomes_path: Path,
) -> dict[str, dict[str, object]]:
    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    registry = {row["name"]: row for row in registry_payload["classifiers"]}
    payload = load_empirical_outcomes(outcomes_path)
    labels = payload["labels"].set_index("sample_id")
    outcomes = payload["outcomes"]
    statistics: dict[str, dict[str, object]] = {}

    for classifier_id in CLASSIFIER_IDS:
        record: Mapping[str, object] = registry[classifier_id]
        if classifier_id == "Kdet":
            statistics[classifier_id] = {
                "modality": "—",
                "parameters": None,
                "confidence_threshold": None,
                "precision": 1.0,
                "success_rate": 1.0,
                "execution_time_ms": PAPER_KDET_COST_MS,
                "scope_samples": int(len(labels)),
                "accepted_samples": int(len(labels)),
                "correct_accepted_samples": int(len(labels)),
            }
            continue

        scoped = outcomes.loc[outcomes["candidate_id"] == classifier_id].copy()
        if classifier_id in {"K4", "K5", "K6"}:
            intended_group = "suv" if classifier_id == "K4" else "coupe"
            in_domain_ids = labels.index[
                labels["true_intermediate_label"] == intended_group
            ]
            scoped = scoped.loc[scoped["sample_id"].isin(in_domain_ids)]

        confidence_threshold = threshold_hi_for_ki(classifier_id)
        if confidence_threshold is None:
            raise ValueError(f"{classifier_id} unexpectedly has no confidence threshold.")
        accepted = scoped.loc[scoped["confidence"] >= confidence_threshold]
        if accepted.empty:
            raise ValueError(f"{classifier_id} accepts no empirical samples.")

        if classifier_id in {"K0", "K1"}:
            class_names = INTERMEDIATE_CLASS_NAMES
            truth_column = "true_intermediate_label"
        else:
            class_names = GLOBAL_CLASS_NAMES
            truth_column = "true_global_label"

        predictions = accepted["prediction"].map(
            lambda index: class_names[int(index)]
        ).reset_index(drop=True)
        truth = labels.loc[accepted["sample_id"], truth_column].reset_index(drop=True)
        precision = float(predictions.eq(truth).mean())
        success_rate = float(len(accepted) / len(scoped))

        statistics[classifier_id] = {
            "modality": "Acoustic" if record["modality"] == "mic" else "Both",
            "parameters": int(record["num_params"]),
            "confidence_threshold": float(confidence_threshold),
            "precision": precision,
            "success_rate": success_rate,
            "execution_time_ms": float(record["runtime_ms"]),
            "scope_samples": int(len(scoped)),
            "accepted_samples": int(len(accepted)),
            "correct_accepted_samples": int(predictions.eq(truth).sum()),
        }

    return statistics


def plot_model_statistics_table(
    registry_path: Path = DEFAULT_REGISTRY,
    outcomes_path: Path = DEFAULT_OUTCOMES,
    output_path: Path = DEFAULT_OUTPUT,
    statistics_output_path: Path = DEFAULT_STATISTICS_OUTPUT,
    profile_report_path: Path | None = None,
) -> Path:
    if profile_report_path is not None:
        profile_report = json.loads(profile_report_path.read_text(encoding="utf-8"))
        statistics = profile_report["classifier_table"]
        packet = {
            "schema_version": "m3n-vc-model-statistics/v2",
            "dataset": "m3n_vc/h24",
            "profile_report": str(profile_report_path.resolve()),
            "profile_report_sha256": hashlib.sha256(
                profile_report_path.read_bytes()
            ).hexdigest(),
            "classifier_training_split": profile_report[
                "classifier_training_split"
            ],
            "classifier_testing_split": profile_report["classifier_testing_split"],
            "definitions": profile_report["definitions"],
            "classifiers": statistics,
        }
    else:
        statistics = _classifier_statistics(registry_path, outcomes_path)
        payload = load_empirical_outcomes(outcomes_path)
        labels = payload["labels"]
        packet = {
            "schema_version": "m3n-vc-model-statistics/v1",
            "dataset": "m3n_vc/h24",
            "outcomes": str(outcomes_path.resolve()),
            "outcomes_sha256": hashlib.sha256(outcomes_path.read_bytes()).hexdigest(),
            "registry": str(registry_path.resolve()),
            "registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "sample_count": int(len(labels)),
            "runs": sorted(str(run_id) for run_id in labels["run_id"].unique()),
            "thresholds_by_level": {
                level: threshold_hi_for_ki(classifier_id)
                for classifier_id, level in (
                    ("K0", "intermediate"),
                    ("K2", "global"),
                    ("K4", "specialized"),
                )
            },
            "definitions": {
                "precision": "P(correct | accepted)",
                "success_rate": "P(accepted)",
            },
            "classifiers": statistics,
        }
    statistics_output_path.parent.mkdir(parents=True, exist_ok=True)
    statistics_output_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    statistics = json.loads(statistics_output_path.read_text(encoding="utf-8"))[
        "classifiers"
    ]
    row_labels = (
        "Modality",
        "Parameter count",
        "Confidence threshold",
        "Precision",
        "Success rate",
        "Execution time (ms)",
    )
    values = {
        classifier_id: (
            str(statistics[classifier_id]["modality"]),
            (
                "—"
                if statistics[classifier_id]["parameters"] is None
                else f'{int(statistics[classifier_id]["parameters"]):,}'
            ),
            (
                "—"
                if statistics[classifier_id]["confidence_threshold"] is None
                else f'{float(statistics[classifier_id]["confidence_threshold"]):.2f}'
            ),
            f'{100.0 * float(statistics[classifier_id]["precision"]):.2f}%',
            f'{100.0 * float(statistics[classifier_id]["success_rate"]):.2f}%',
            (
                f'{float(statistics[classifier_id]["execution_time_ms"]):,.0f}'
                if classifier_id == "Kdet"
                else f'{float(statistics[classifier_id]["execution_time_ms"]):.3f}'
            ),
        )
        for classifier_id in CLASSIFIER_IDS
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 11,
        }
    )
    figure, axis = plt.subplots(figsize=(13.2, 4.4))
    axis.set_xlim(0.0, 10.9)
    axis.set_ylim(0.0, 6.35)
    axis.axis("off")

    label_boundary = 1.85
    column_width = (10.9 - label_boundary) / len(CLASSIFIER_IDS)
    centers = [
        label_boundary + column_width * (index + 0.5)
        for index in range(len(CLASSIFIER_IDS))
    ]

    group_headers = (
        ("Intermediate", 0, 1),
        ("Global", 2, 3),
        ("SUV", 4, 4),
        ("COUPE", 5, 6),
        ("Fallback", 7, 7),
    )
    axis.text(label_boundary / 2.0, 5.82, "Classifiers", ha="center", va="center")
    for label, start, end in group_headers:
        axis.text(
            (centers[start] + centers[end]) / 2.0,
            5.95,
            label,
            ha="center",
            va="center",
        )
    for center, classifier_id in zip(centers, CLASSIFIER_IDS, strict=True):
        suffix = r"\mathrm{det}" if classifier_id == "Kdet" else classifier_id[1:]
        axis.text(center, 5.48, rf"$K_{{{suffix}}}$", ha="center", va="center", fontsize=13)

    header_rule_y = 5.18
    bottom_rule_y = 0.68
    axis.hlines(header_rule_y, 0.0, 10.9, color="#333333", linewidth=0.8)
    axis.hlines(bottom_rule_y, 0.0, 10.9, color="#333333", linewidth=0.8)

    group_boundaries = (label_boundary, centers[1] + column_width / 2.0,
                        centers[3] + column_width / 2.0,
                        centers[4] + column_width / 2.0,
                        centers[6] + column_width / 2.0)
    for boundary in group_boundaries:
        axis.vlines(boundary, bottom_rule_y, 6.18, color="#555555", linewidth=0.75)

    row_y = (4.78, 4.08, 3.38, 2.68, 1.98, 1.28)
    for label, y in zip(row_labels, row_y, strict=True):
        axis.text(label_boundary - 0.14, y, label, ha="right", va="center")
    for center, classifier_id in zip(centers, CLASSIFIER_IDS, strict=True):
        for value, y in zip(values[classifier_id], row_y, strict=True):
            axis.text(center, y, value, ha="center", va="center")

    figure.text(
        0.5,
        0.025,
        "Precision = P(correct | accepted); success rate = P(accepted). "
        "Confidence threshold = 0.90 for global K2–K3 and 0.95 for "
        "intermediate/specialized K0–K1/K4–K6. "
        + (
            "Statistics use the classifier-testing split (runs 1, 3, 5, 7 plus "
            "the held-out 20% of run8). "
            if profile_report_path is not None
            else "Statistics use h24 runs 1, 3, 5, 7, and 9 (9,546 samples). "
        )
        + "SUV and COUPE statistics use in-domain samples. Execution time is the measured mean "
        "for K0–K6; Kdet is the perfect 10,000 ms paper-mode fallback.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.tight_layout(rect=(0.01, 0.075, 0.99, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--statistics-output",
        type=Path,
        default=DEFAULT_STATISTICS_OUTPUT,
    )
    parser.add_argument(
        "--profile-report",
        type=Path,
        default=None,
        help="Use the Jetson classifier-profile packet instead of empirical outcomes.",
    )
    args = parser.parse_args()
    print(
        plot_model_statistics_table(
            args.registry,
            args.outcomes,
            args.output,
            args.statistics_output,
            args.profile_report,
        )
    )


if __name__ == "__main__":
    main()
