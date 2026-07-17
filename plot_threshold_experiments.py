"""Publication-ready matplotlib figures for threshold-optimizer experiments.

Focus: accuracy vs efficiency (expected cost / speedup). Figures are written as
high-DPI PNGs suitable for a research paper draft.

Run:
    python plot_threshold_experiments.py
    python plot_threshold_experiments.py --show-paths
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DEFAULT_RESULTS_DIR = Path("checkpoints/threshold_experiments")
DEFAULT_FIGURES_DIR = Path("checkpoints/figures/threshold_experiments")

# Research-paper palette: muted, print-safe, not the usual AI purple/glow look.
C_BASE = "#4C6A80"       # slate blue — baseline
C_OPT = "#C45C26"        # burnt orange — optimized
C_PAPER = "#2F5D50"      # deep teal — paper Kdet
C_TRAINED = "#8B3A3A"    # brick — trained Kdet
C_ZERO = "#6B7280"       # gray — zero-shot transfer
C_RETUNE = "#C45C26"     # orange — retuned
C_GRID = "#D0D5DD"
C_TEXT = "#1F2937"

LAYOUT_ORDER = [
    "single_global",
    "global_only",
    "three_global",
    "three_linear",
    "hierarchy_classic",
    "k0_k2_k3_hierarchy",
    "both_identifiers",
    "dp_optimal",
]
LAYOUT_LABELS = {
    "single_global": "K3→det",
    "global_only": "K2→K3→det",
    "three_global": "K2→K3→det",
    "three_linear": "K0→K3→det",
    "hierarchy_classic": "K0→spec",
    "k0_k2_k3_hierarchy": "K0→K2→K3",
    "both_identifiers": "K0/K1→…",
    "dp_optimal": "DP-optimal",
}
SCENE_ORDER = ["h24", "h08", "s31", "a06", "i29"]


def _setup_style() -> None:
    """Global rcParams for a journal-like look (serif, clean axes, no glitter)."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.edgecolor": C_TEXT,
            "axes.labelcolor": C_TEXT,
            "xtick.color": C_TEXT,
            "ytick.color": C_TEXT,
            "text.color": C_TEXT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,  # editable text if someone re-exports later
            "ps.fonttype": 42,
        }
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def _anneal_block(run: dict) -> dict:
    return run.get("annealing") or {}


def _baseline_block(run: dict) -> dict:
    return run.get("baseline") or {}


def fig_layouts_accuracy_cost(results_dir: Path, figures_dir: Path) -> Path:
    """Grouped bars: baseline vs optimized accuracy and cost for each layout."""
    summary = _load(results_dir / "layouts_h24_paper" / "summary.json")
    # Deduplicate three_global vs global_only (same topology in this suite).
    names = [n for n in LAYOUT_ORDER if n in summary["runs"] and n != "three_global"]
    labels = [LAYOUT_LABELS[n] for n in names]

    base_acc = [100 * float(_baseline_block(summary["runs"][n])["holdout_accuracy"]) for n in names]
    opt_acc = [100 * float(_anneal_block(summary["runs"][n])["holdout_accuracy"]) for n in names]
    base_cost = [float(_baseline_block(summary["runs"][n])["holdout_cost_ms"]) for n in names]
    opt_cost = [float(_anneal_block(summary["runs"][n])["holdout_cost_ms"]) for n in names]

    x = np.arange(len(names))
    width = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)

    axes[0].bar(x - width / 2, base_acc, width, color=C_BASE, label="Baseline thresholds", zorder=3)
    axes[0].bar(x + width / 2, opt_acc, width, color=C_OPT, label="Optimized thresholds", zorder=3)
    axes[0].set_ylabel("Holdout accuracy (%)")
    axes[0].set_ylim(94.0, 100.0)
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    axes[0].set_title("(a) Accuracy")
    axes[0].legend(frameon=False, loc="lower left")

    axes[1].bar(x - width / 2, base_cost, width, color=C_BASE, label="Baseline thresholds", zorder=3)
    axes[1].bar(x + width / 2, opt_cost, width, color=C_OPT, label="Optimized thresholds", zorder=3)
    axes[1].set_ylabel("Expected cost (ms)")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    axes[1].set_title("(b) Efficiency")
    axes[1].legend(frameon=False, loc="upper right")

    fig.suptitle("Cascade layout × threshold optimization (h24, paper Kdet)", y=1.02)
    return _save(fig, figures_dir / "fig1_layouts_accuracy_cost.png")


def fig_layouts_pareto(results_dir: Path, figures_dir: Path) -> Path:
    """Accuracy–cost Pareto-style scatter: arrows from baseline → optimized."""
    summary = _load(results_dir / "layouts_h24_paper" / "summary.json")
    names = [n for n in LAYOUT_ORDER if n in summary["runs"] and n != "three_global"]

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    for name in names:
        b = _baseline_block(summary["runs"][name])
        a = _anneal_block(summary["runs"][name])
        x0, y0 = float(b["holdout_cost_ms"]), 100 * float(b["holdout_accuracy"])
        x1, y1 = float(a["holdout_cost_ms"]), 100 * float(a["holdout_accuracy"])
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="#9CA3AF", lw=1.2),
            zorder=2,
        )
        ax.scatter([x0], [y0], s=42, color=C_BASE, zorder=3, edgecolors="white", linewidths=0.6)
        ax.scatter([x1], [y1], s=54, color=C_OPT, zorder=4, edgecolors="white", linewidths=0.6)
        # Label the optimized point (the paper-relevant operating point).
        ax.annotate(
            LAYOUT_LABELS[name],
            (x1, y1),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    ax.set_xlabel("Expected holdout cost (ms)  → worse")
    ax.set_ylabel("Holdout accuracy (%)  → better")
    ax.invert_xaxis()  # left = faster, so “up-left” is the desirable quadrant
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.7)
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.7)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BASE, markersize=8, label="Baseline"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OPT, markersize=8, label="Optimized"),
        Line2D([0], [0], color="#9CA3AF", lw=1.2, label="Threshold tuning"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="lower right")
    ax.set_title("Accuracy–efficiency trade-off across cascade layouts (h24)")
    return _save(fig, figures_dir / "fig2_layouts_pareto.png")


def fig_layouts_speedup(results_dir: Path, figures_dir: Path) -> Path:
    """Speedup bars with accuracy delta annotations."""
    summary = _load(results_dir / "layouts_h24_paper" / "summary.json")
    names = [n for n in LAYOUT_ORDER if n in summary["runs"] and n != "three_global"]
    labels = [LAYOUT_LABELS[n] for n in names]
    speedups = [
        float(_anneal_block(summary["runs"][n]).get("holdout_speedup_vs_baseline") or 1.0)
        for n in names
    ]
    dacc = [
        100
        * (
            float(_anneal_block(summary["runs"][n])["holdout_accuracy"])
            - float(_baseline_block(summary["runs"][n])["holdout_accuracy"])
        )
        for n in names
    ]

    # Sort by speedup for a clean narrative figure.
    order = np.argsort(speedups)
    labels = [labels[i] for i in order]
    speedups = [speedups[i] for i in order]
    dacc = [dacc[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    y = np.arange(len(labels))
    bars = ax.barh(y, speedups, color=C_OPT, height=0.62, zorder=3)
    ax.axvline(1.0, color=C_TEXT, lw=0.9, linestyle="--", alpha=0.55)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Holdout speedup vs. baseline thresholds (×)")
    ax.set_title("Efficiency gain from threshold optimization by layout")
    ax.xaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    for bar, s, da in zip(bars, speedups, dacc, strict=True):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{s:.2f}×  (Δacc {da:+.2f} pp)",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, max(speedups) * 1.35)
    return _save(fig, figures_dir / "fig3_layouts_speedup.png")


def fig_targets_tradeoff(results_dir: Path, figures_dir: Path) -> Path:
    """Target accuracy sweep under paper and trained Kdet."""
    summary = _load(results_dir / "targets_h24" / "summary.json")
    target_keys = ["baseline", "acc_0.90", "acc_0.95", "acc_0.98"]
    xlabels = ["Baseline\ntarget", "0.90", "0.95", "0.98"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for mode, color, marker in (("paper", C_PAPER, "o"), ("trained", C_TRAINED, "s")):
        accs, costs, targets = [], [], []
        for key in target_keys:
            run = summary["runs"][f"{mode}_{key}"]
            a = _anneal_block(run)
            accs.append(100 * float(a["holdout_accuracy"]))
            costs.append(float(a["holdout_cost_ms"]))
            targets.append(100 * float(run["target_accuracy"]))
        axes[0].plot(xlabels, accs, marker=marker, color=color, lw=1.8, label=f"{mode} Kdet")
        axes[0].plot(xlabels, targets, color=color, lw=1.0, linestyle=":", alpha=0.7)
        axes[1].plot(xlabels, costs, marker=marker, color=color, lw=1.8, label=f"{mode} Kdet")

    axes[0].set_ylabel("Holdout accuracy (%)")
    axes[0].set_title("(a) Achieved accuracy")
    axes[0].yaxis.grid(True, color=C_GRID, linewidth=0.7)
    axes[0].legend(frameon=False)
    # Dotted lines = requested targets; solid = achieved holdout accuracy.
    axes[0].plot([], [], color="#9CA3AF", linestyle=":", lw=1.0, label="Requested target")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, loc="lower left")

    axes[1].set_ylabel("Expected holdout cost (ms)")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Efficiency (log scale)")
    axes[1].yaxis.grid(True, color=C_GRID, linewidth=0.7, which="both")
    axes[1].legend(frameon=False)

    fig.suptitle("Accuracy target sweep on h24 (DP-optimal layout)", y=1.02)
    return _save(fig, figures_dir / "fig4_targets_accuracy_cost.png")


def fig_scenes_trained(results_dir: Path, figures_dir: Path) -> Path:
    """Per-scene trained-Kdet optimization: accuracy and cost."""
    summary = _load(results_dir / "scenes_trained_baseline_target" / "summary.json")
    scenes = [s for s in SCENE_ORDER if s in summary["runs"] and summary["runs"][s].get("status") == "ok"]

    base_acc = [100 * float(_baseline_block(summary["runs"][s])["holdout_accuracy"]) for s in scenes]
    opt_acc = [100 * float(_anneal_block(summary["runs"][s])["holdout_accuracy"]) for s in scenes]
    base_cost = [float(_baseline_block(summary["runs"][s])["holdout_cost_ms"]) for s in scenes]
    opt_cost = [float(_anneal_block(summary["runs"][s])["holdout_cost_ms"]) for s in scenes]
    speedups = [
        float(_anneal_block(summary["runs"][s]).get("holdout_speedup_vs_baseline") or 1.0)
        for s in scenes
    ]

    x = np.arange(len(scenes))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)

    axes[0].bar(x - width / 2, base_acc, width, color=C_BASE, label="Baseline", zorder=3)
    axes[0].bar(x + width / 2, opt_acc, width, color=C_OPT, label="Optimized", zorder=3)
    axes[0].set_ylabel("Holdout accuracy (%)")
    axes[0].set_xticks(x, scenes)
    axes[0].set_ylim(0, 100)
    axes[0].yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    axes[0].set_title("(a) Accuracy")
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, base_cost, width, color=C_BASE, label="Baseline", zorder=3)
    axes[1].bar(x + width / 2, opt_cost, width, color=C_OPT, label="Optimized", zorder=3)
    axes[1].set_ylabel("Expected cost (ms)")
    axes[1].set_xticks(x, scenes)
    axes[1].yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    axes[1].set_title("(b) Efficiency")
    axes[1].legend(frameon=False)

    axes[2].bar(x, speedups, color=C_OPT, zorder=3)
    axes[2].axhline(1.0, color=C_TEXT, lw=0.9, linestyle="--", alpha=0.55)
    axes[2].set_ylabel("Speedup (×)")
    axes[2].set_xticks(x, scenes)
    axes[2].yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    axes[2].set_title("(c) Speedup vs baseline")
    for i, s in enumerate(speedups):
        axes[2].text(i, s + 0.05, f"{s:.2f}×", ha="center", fontsize=8)

    fig.suptitle("Per-scene threshold optimization with trained Kdet", y=1.03)
    return _save(fig, figures_dir / "fig5_scenes_trained_kdet.png")


def fig_transfer(results_dir: Path, figures_dir: Path) -> Path:
    """h24 policy zero-shot vs threshold retune on frozen h24 layout."""
    summary = _load(results_dir / "transfer_h24_layout" / "summary.json")
    scenes = [s for s in SCENE_ORDER if s in summary["zero_shot"]]

    zs_acc = [100 * float(summary["zero_shot"][s]["accuracy"]) for s in scenes]
    zs_cost = [float(summary["zero_shot"][s]["expected_cost_ms"]) for s in scenes]
    rt_acc, rt_cost = [], []
    for s in scenes:
        run = summary["retune_thresholds_on_frozen_layout"][s]
        a = _anneal_block(run)
        # For h24 source scene, zero-shot uses full data; retune reports holdout.
        # Keep both as reported operating points.
        rt_acc.append(100 * float(a["holdout_accuracy"]))
        rt_cost.append(float(a["holdout_cost_ms"]))

    x = np.arange(len(scenes))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)

    axes[0].bar(x - width / 2, zs_acc, width, color=C_ZERO, label="h24 thresholds (zero-shot)", zorder=3)
    axes[0].bar(x + width / 2, rt_acc, width, color=C_RETUNE, label="Retuned on scene (frozen layout)", zorder=3)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_xticks(x, scenes)
    axes[0].set_ylim(0, 100)
    axes[0].yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    axes[0].set_title("(a) Accuracy")
    axes[0].legend(frameon=False, loc="lower left")

    axes[1].bar(x - width / 2, zs_cost, width, color=C_ZERO, label="h24 thresholds (zero-shot)", zorder=3)
    axes[1].bar(x + width / 2, rt_cost, width, color=C_RETUNE, label="Retuned on scene (frozen layout)", zorder=3)
    axes[1].set_ylabel("Expected cost (ms)")
    axes[1].set_xticks(x, scenes)
    axes[1].yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
    axes[1].set_title("(b) Efficiency")
    axes[1].legend(frameon=False, loc="upper left")

    fig.suptitle("Transfer of h24 cascade layout across scenes", y=1.02)
    return _save(fig, figures_dir / "fig6_transfer_zero_shot_vs_retune.png")


def fig_layouts_by_scene_heatmaps(results_dir: Path, figures_dir: Path) -> Path:
    """Heatmaps of holdout accuracy and speedup for layout × scene."""
    summary = _load(results_dir / "layouts_by_scene_paper" / "summary.json")
    layouts = ["dp_optimal", "global_only", "hierarchy_classic", "three_linear"]
    layout_labs = [LAYOUT_LABELS[l] for l in layouts]
    scenes = [s for s in SCENE_ORDER]

    acc = np.full((len(layouts), len(scenes)), np.nan)
    spd = np.full((len(layouts), len(scenes)), np.nan)
    for i, layout in enumerate(layouts):
        for j, scene in enumerate(scenes):
            key = f"{scene}__{layout}"
            run = summary["runs"].get(key)
            if not run or run.get("status") != "ok":
                continue
            a = _anneal_block(run)
            acc[i, j] = 100 * float(a["holdout_accuracy"])
            spd[i, j] = float(a.get("holdout_speedup_vs_baseline") or np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for ax, data, title, cmap, fmt, vmin, vmax, cbar_label in (
        (axes[0], acc, "(a) Holdout accuracy (%)", "YlGn", ".1f", 50, 100, "Accuracy (%)"),
        (axes[1], spd, "(b) Speedup vs baseline (×)", "YlOrBr", ".2f", 0.7, 4.2, "Speedup (×)"),
    ):
        im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(np.arange(len(scenes)), scenes)
        ax.set_yticks(np.arange(len(layouts)), layout_labs)
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                # Dark text on light cells, light text on dark cells.
                text_color = "white" if (val - vmin) / (vmax - vmin) > 0.62 else C_TEXT
                ax.text(j, i, format(val, fmt), ha="center", va="center", color=text_color, fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=9)

    fig.suptitle("Layout × scene threshold optimization (paper Kdet)", y=1.03)
    return _save(fig, figures_dir / "fig7_layouts_by_scene_heatmaps.png")


def fig_search_settings(results_dir: Path, figures_dir: Path) -> Path:
    """Quantile grid / split strategy sensitivity."""
    summary = _load(results_dir / "search_settings_h24" / "summary.json")
    order = ["q10_blocked", "q25_blocked", "q50_blocked", "q100_blocked", "q50_random"]
    labels = ["q=10\nblocked", "q=25\nblocked", "q=50\nblocked", "q=100\nblocked", "q=50\nrandom"]
    acc = [100 * float(_anneal_block(summary["runs"][k])["holdout_accuracy"]) for k in order]
    cost = [float(_anneal_block(summary["runs"][k])["holdout_cost_ms"]) for k in order]
    spd = [
        float(_anneal_block(summary["runs"][k]).get("holdout_speedup_vs_baseline") or 1.0)
        for k in order
    ]

    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), constrained_layout=True)
    axes[0].bar(x, acc, color=C_PAPER, zorder=3)
    axes[0].set_ylim(94, 98)
    axes[0].set_ylabel("Holdout accuracy (%)")
    axes[0].set_title("(a) Accuracy")
    axes[1].bar(x, cost, color=C_OPT, zorder=3)
    axes[1].set_ylabel("Expected cost (ms)")
    axes[1].set_title("(b) Cost")
    axes[2].bar(x, spd, color=C_BASE, zorder=3)
    axes[2].axhline(1.0, color=C_TEXT, lw=0.9, linestyle="--", alpha=0.55)
    axes[2].set_ylabel("Speedup (×)")
    axes[2].set_title("(c) Speedup")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)

    fig.suptitle("Sensitivity to threshold-grid density and holdout split (h24)", y=1.04)
    return _save(fig, figures_dir / "fig8_search_settings.png")


def fig_main_summary(results_dir: Path, figures_dir: Path) -> Path:
    """One composite figure suitable as a paper 'teaser' / main results panel."""
    layouts = _load(results_dir / "layouts_h24_paper" / "summary.json")
    transfer = _load(results_dir / "transfer_h24_layout" / "summary.json")
    scenes = _load(results_dir / "scenes_trained_baseline_target" / "summary.json")

    fig = plt.figure(figsize=(11.8, 7.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # --- Panel A: layout Pareto ---
    ax = fig.add_subplot(gs[0, 0])
    names = [n for n in LAYOUT_ORDER if n in layouts["runs"] and n != "three_global"]
    for name in names:
        b = _baseline_block(layouts["runs"][name])
        a = _anneal_block(layouts["runs"][name])
        x0, y0 = float(b["holdout_cost_ms"]), 100 * float(b["holdout_accuracy"])
        x1, y1 = float(a["holdout_cost_ms"]), 100 * float(a["holdout_accuracy"])
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#B0B7C3", lw=1.0))
        ax.scatter([x1], [y1], s=48, color=C_OPT, zorder=4, edgecolors="white", linewidths=0.5)
        ax.annotate(LAYOUT_LABELS[name], (x1, y1), textcoords="offset points", xytext=(5, 3), fontsize=7.5)
    ax.invert_xaxis()
    ax.set_xlabel("Expected cost (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("A. Layout trade-off (h24, paper Kdet)")
    ax.grid(True, color=C_GRID, linewidth=0.6)

    # --- Panel B: speedup by layout ---
    ax = fig.add_subplot(gs[0, 1])
    names = [n for n in names]
    speedups = [
        float(_anneal_block(layouts["runs"][n]).get("holdout_speedup_vs_baseline") or 1.0)
        for n in names
    ]
    order = np.argsort(speedups)
    labs = [LAYOUT_LABELS[names[i]] for i in order]
    vals = [speedups[i] for i in order]
    ax.barh(np.arange(len(labs)), vals, color=C_OPT, height=0.65, zorder=3)
    ax.axvline(1.0, color=C_TEXT, lw=0.8, linestyle="--", alpha=0.5)
    ax.set_yticks(np.arange(len(labs)), labs)
    ax.set_xlabel("Speedup (×)")
    ax.set_title("B. Efficiency gain by layout")
    ax.grid(axis="x", color=C_GRID, linewidth=0.6, zorder=0)

    # --- Panel C: transfer accuracy ---
    ax = fig.add_subplot(gs[1, 0])
    scene_ids = [s for s in SCENE_ORDER if s in transfer["zero_shot"]]
    x = np.arange(len(scene_ids))
    width = 0.36
    zs = [100 * float(transfer["zero_shot"][s]["accuracy"]) for s in scene_ids]
    rt = [
        100 * float(_anneal_block(transfer["retune_thresholds_on_frozen_layout"][s])["holdout_accuracy"])
        for s in scene_ids
    ]
    ax.bar(x - width / 2, zs, width, color=C_ZERO, label="Zero-shot h24 thresholds", zorder=3)
    ax.bar(x + width / 2, rt, width, color=C_RETUNE, label="Retune thresholds", zorder=3)
    ax.set_xticks(x, scene_ids)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("C. Cross-scene transfer (frozen h24 layout)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=C_GRID, linewidth=0.6, zorder=0)

    # --- Panel D: trained Kdet per scene ---
    ax = fig.add_subplot(gs[1, 1])
    scene_ids = [s for s in SCENE_ORDER if scenes["runs"].get(s, {}).get("status") == "ok"]
    x = np.arange(len(scene_ids))
    base = [100 * float(_baseline_block(scenes["runs"][s])["holdout_accuracy"]) for s in scene_ids]
    opt = [100 * float(_anneal_block(scenes["runs"][s])["holdout_accuracy"]) for s in scene_ids]
    ax.plot(x, base, marker="o", color=C_BASE, lw=1.6, label="Baseline")
    ax.plot(x, opt, marker="s", color=C_OPT, lw=1.6, label="Optimized")
    ax.set_xticks(x, scene_ids)
    ax.set_ylabel("Holdout accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("D. Trained-Kdet per-scene retune")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, color=C_GRID, linewidth=0.6)

    fig.suptitle(
        "Threshold optimization for hierarchical IDK cascades: accuracy and efficiency",
        fontsize=12,
        y=1.01,
    )
    return _save(fig, figures_dir / "fig0_main_summary.png")


def fig_per_scene_thresholds(results_dir: Path, figures_dir: Path) -> Path | None:
    """Per-scene threshold bank: structure-flexible vs shared-h24 wiring."""
    root = results_dir / "per_scene_thresholds"
    comparison_path = root / "COMPARISON.json"
    if not comparison_path.is_file():
        print(f"Skipping per-scene figures (missing {comparison_path})")
        return None

    table = _load(comparison_path)["table"]
    # Prefer paper Kdet for the main paper figure (matches prior suites).
    paper_rows = [r for r in table if r.get("detector_mode") == "paper"]
    if not paper_rows:
        paper_rows = table

    scenes = [s for s in SCENE_ORDER if any(r["scene"] == s for r in paper_rows)]
    modes = []
    for r in paper_rows:
        if r["mode"] not in modes:
            modes.append(r["mode"])

    mode_labels = {
        "per_scene_structure": "Per-scene structure + thresholds",
        "shared_h24_structure": "Shared h24 structure, per-scene thresholds",
    }
    mode_colors = {
        "per_scene_structure": C_OPT,
        "shared_h24_structure": C_PAPER,
    }

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), constrained_layout=True)
    x = np.arange(len(scenes))
    width = 0.36 if len(modes) <= 2 else 0.25

    for ax, field, title, ylabel, is_pct in (
        (axes[0], "opt_holdout_acc", "(a) Holdout accuracy", "Accuracy (%)", True),
        (axes[1], "opt_holdout_cost_ms", "(b) Expected cost", "Cost (ms)", False),
        (axes[2], "holdout_speedup_vs_baseline", "(c) Speedup vs baseline", "Speedup (×)", False),
    ):
        for i, mode in enumerate(modes):
            vals = []
            for scene in scenes:
                match = next(
                    (r for r in paper_rows if r["scene"] == scene and r["mode"] == mode),
                    None,
                )
                v = match.get(field) if match else None
                if v is None:
                    vals.append(0.0)
                else:
                    vals.append(100 * float(v) if is_pct else float(v))
            offset = (i - (len(modes) - 1) / 2) * width
            ax.bar(
                x + offset,
                vals,
                width,
                color=mode_colors.get(mode, C_BASE),
                label=mode_labels.get(mode, mode),
                zorder=3,
            )
        ax.set_xticks(x, scenes)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.yaxis.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
        if field == "holdout_speedup_vs_baseline":
            ax.axhline(1.0, color=C_TEXT, lw=0.9, linestyle="--", alpha=0.55)
        if field == "opt_holdout_acc":
            ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Per-scene threshold bank (paper Kdet, baseline-accuracy target)", y=1.03)
    return _save(fig, figures_dir / "fig9_per_scene_thresholds.png")


def fig_per_scene_threshold_heatmap(results_dir: Path, figures_dir: Path) -> Path | None:
    """Heatmap of optimized H_i values in the per-scene bank."""
    bank_path = (
        results_dir
        / "per_scene_thresholds"
        / "scene_threshold_bank_paper.json"
    )
    if not bank_path.is_file():
        # Fall back to mode-qualified name.
        alt = results_dir / "per_scene_thresholds" / "scene_threshold_bank_per_scene_structure_paper.json"
        bank_path = alt if alt.is_file() else bank_path
    if not bank_path.is_file():
        print(f"Skipping threshold heatmap (missing bank at {bank_path})")
        return None

    bank = _load(bank_path)["threshold_bank"]
    kis = ["K0", "K1", "K2", "K3", "K4", "K5", "K6"]
    scenes = [s for s in SCENE_ORDER if s in bank]
    data = np.array([[float(bank[s][k]) for k in kis] for s in scenes])

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    im = ax.imshow(data, cmap="YlOrBr", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(kis)), kis)
    ax.set_yticks(np.arange(len(scenes)), scenes)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Scene")
    ax.set_title("Optimized per-scene confidence thresholds (paper Kdet)")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = "white" if val > 0.62 else C_TEXT
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Threshold Hᵢ", fontsize=9)
    return _save(fig, figures_dir / "fig10_per_scene_threshold_values.png")


def write_plot_manifest(figures_dir: Path, paths: list[Path]) -> Path:
    payload = {
        "description": "Publication figures for threshold-optimizer experiments.",
        "focus": ["accuracy", "expected_cost_ms", "speedup"],
        "figures": [
            {
                "file": str(path),
                "name": path.name,
            }
            for path in paths
        ],
    }
    out = figures_dir / "plot_manifest.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--show-paths", action="store_true")
    args = parser.parse_args()

    _setup_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        fig_main_summary(args.results_dir, args.figures_dir),
        fig_layouts_accuracy_cost(args.results_dir, args.figures_dir),
        fig_layouts_pareto(args.results_dir, args.figures_dir),
        fig_layouts_speedup(args.results_dir, args.figures_dir),
        fig_targets_tradeoff(args.results_dir, args.figures_dir),
        fig_scenes_trained(args.results_dir, args.figures_dir),
        fig_transfer(args.results_dir, args.figures_dir),
        fig_layouts_by_scene_heatmaps(args.results_dir, args.figures_dir),
        fig_search_settings(args.results_dir, args.figures_dir),
    ]
    for optional in (
        fig_per_scene_thresholds(args.results_dir, args.figures_dir),
        fig_per_scene_threshold_heatmap(args.results_dir, args.figures_dir),
    ):
        if optional is not None:
            paths.append(optional)
    write_plot_manifest(args.figures_dir, paths)

    if args.show_paths:
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()
