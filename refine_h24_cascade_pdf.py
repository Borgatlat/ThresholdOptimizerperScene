"""Create a publication-ready, five-page h24 cascade-layout PDF.

This preserves the one-layout-per-page format of ``Cascade 096 layouts.pdf``
while restoring the holdout threshold, routing, and performance annotations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter

from empirical_outcomes import load_empirical_outcomes
from plot_h24_method_comparison import (
    DEFAULT_APPROXIMATE_REPORT,
    DEFAULT_BASELINE_REPORT,
    DEFAULT_BRUTE_FORCE_REPORT,
    DEFAULT_BRUTE_FORCE_RESULTS,
    DEFAULT_OUTCOMES,
    _draw_diagram_panel,
    _holdout_evaluator,
    load_methods,
    occurrence_route_counts,
)
from threshold_optimizer import split_empirical_outcomes


DEFAULT_OUTPUT = Path(
    r"C:\Users\TheSandwichCoder\Downloads\Cascade 096 layouts refined.pdf"
)


def create_refined_pdf(
    output: Path = DEFAULT_OUTPUT,
    *,
    baseline_report: Path = DEFAULT_BASELINE_REPORT,
    approximate_report: Path = DEFAULT_APPROXIMATE_REPORT,
    brute_force_report: Path = DEFAULT_BRUTE_FORCE_REPORT,
    brute_force_results: Path = DEFAULT_BRUTE_FORCE_RESULTS,
    outcomes: Path = DEFAULT_OUTCOMES,
) -> None:
    methods = load_methods(
        baseline_report,
        approximate_report,
        brute_force_report,
        brute_force_results,
    )
    payload = load_empirical_outcomes(outcomes)
    _, holdout_payload, _ = split_empirical_outcomes(
        payload,
        holdout_fraction=0.20,
        split_strategy="blocked_per_run",
        random_seed=0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    cmap = matplotlib.colormaps["Blues"]
    norm = Normalize(vmin=0.0, vmax=1.0)
    with PdfPages(
        output,
        metadata={
            "Title": "h24 cascade layouts at target accuracy 0.9662",
            "Subject": "Holdout thresholds and terminal-routing reliance",
        },
    ) as pdf:
        for method in methods:
            figure, axis = plt.subplots(figsize=(10.0, 5.625))
            figure.subplots_adjust(left=0.045, right=0.955, top=0.82, bottom=0.18)

            evaluator = _holdout_evaluator(method.layout, holdout_payload)
            thresholds = {
                str(key): float(value)
                for key, value in method.holdout["thresholds"].items()
            }
            occurrence_counts = occurrence_route_counts(evaluator, thresholds)
            _draw_diagram_panel(
                axis,
                method,
                evaluator,
                occurrence_counts,
                cmap,
                norm,
            )

            colorbar_axis = figure.add_axes((0.22, 0.075, 0.56, 0.026))
            colorbar = figure.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                cax=colorbar_axis,
                orientation="horizontal",
            )
            colorbar.set_label(
                "Share of holdout samples whose final decision occurs at this box",
                fontsize=8,
            )
            colorbar.ax.tick_params(labelsize=7)
            colorbar.ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))

            pdf.savefig(figure, dpi=300, facecolor="white")
            plt.close(figure)

    print(f"Wrote {output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument(
        "--approximate-report", type=Path, default=DEFAULT_APPROXIMATE_REPORT
    )
    parser.add_argument(
        "--brute-force-report", type=Path, default=DEFAULT_BRUTE_FORCE_REPORT
    )
    parser.add_argument(
        "--brute-force-results", type=Path, default=DEFAULT_BRUTE_FORCE_RESULTS
    )
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    return parser


def main() -> None:
    args = _parser().parse_args()
    create_refined_pdf(
        args.output,
        baseline_report=args.baseline_report,
        approximate_report=args.approximate_report,
        brute_force_report=args.brute_force_report,
        brute_force_results=args.brute_force_results,
        outcomes=args.outcomes,
    )


if __name__ == "__main__":
    main()
