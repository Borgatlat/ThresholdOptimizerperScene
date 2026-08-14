"""Plot the five h24 methods using standardized result packets only."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_result_packets import plot_packets


DEFAULT_PACKET_DIR = Path("checkpoints/result_packets/m3n_vc_h24")
DEFAULT_OUTPUT_DIR = Path("checkpoints/figures/h24_packet_comparison")
METHOD_ORDER = (
    "baseline.json",
    "baseline_annealed.json",
    "linear_k3.json",
    "approximate_joint.json",
    "brute_force.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-dir", type=Path, default=DEFAULT_PACKET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    packet_paths = [args.packet_dir / name for name in METHOD_ORDER]
    missing = [str(path) for path in packet_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing standardized result packets; run "
            "`python -m experiments.m3n_vc.export_h24_result_packets`: "
            + ", ".join(missing)
        )
    plot_packets(packet_paths, args.output_dir)


if __name__ == "__main__":
    main()
