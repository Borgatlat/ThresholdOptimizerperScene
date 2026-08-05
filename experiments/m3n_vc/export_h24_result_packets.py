"""Export completed h24 methods into standardized figure/result packets."""

from __future__ import annotations

import argparse
from pathlib import Path

from cascade_profile import profile_from_payload
from empirical_outcomes import load_empirical_outcomes
from experiments.m3n_vc._legacy_h24_report_plot import (
    DEFAULT_APPROXIMATE_REPORT,
    DEFAULT_BASELINE_REPORT,
    DEFAULT_BRUTE_FORCE_REPORT,
    DEFAULT_BRUTE_FORCE_RESULTS,
    DEFAULT_OUTCOMES,
    load_methods,
)
from result_packets import create_result_packet, write_result_packet


DEFAULT_OUTPUT_DIR = Path("checkpoints/result_packets/m3n_vc_h24")


def export_packets(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    methods = load_methods(
        DEFAULT_BASELINE_REPORT,
        DEFAULT_APPROXIMATE_REPORT,
        DEFAULT_BRUTE_FORCE_REPORT,
        DEFAULT_BRUTE_FORCE_RESULTS,
    )
    profile = profile_from_payload(load_empirical_outcomes(DEFAULT_OUTCOMES))
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for method in methods:
        packet = create_result_packet(
            profile=profile,
            method_id=method.key,
            method_label=method.label,
            target_accuracy=method.target_accuracy,
            layout={
                "initial": method.layout["initial"],
                "branches": method.layout["specialized"],
                "detector": "detector",
            },
            validation=method.validation,
            test=method.holdout,
            provenance={"source": str(method.source)},
        )
        path = write_result_packet(packet, output_dir / f"{method.key}.json")
        written.append(path)
        print(f"Wrote {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    export_packets(args.output_dir)


if __name__ == "__main__":
    main()
