"""Experiment A: quantify how much h24's cascade degrades on other scenes,
using h24's structure and h24's thresholds COMPLETELY UNCHANGED.

Why this matters before the threshold optimizer exists
---------------------------------------------------------
This is not "does h24's cascade work elsewhere" for its own sake -- it's
the PROBLEM-STATEMENT evidence the whole paper is built around. The
threshold optimizer (your partner's part) is meant to fix whatever
degradation this script reveals; its results are only meaningful compared
against this baseline. You don't need the optimizer to exist to produce
this -- you need h24's own outcomes (for the reference structure and
thresholds) plus every other scene's outcomes (already have these), and
nothing else.

What this reports, per scene
------------------------------
- Overall accuracy (h24's cascade structure + h24's thresholds, replayed
  on that scene's data)
- Per-classifier mean confidence and accept rate, compared to h24's own
  values -- this is the actual "domain shift" signal: a classifier can
  keep the same raw discriminative ability while its confidence
  distribution shifts enough that h24's threshold no longer means what it
  used to (accept rate swings even though the classifier itself didn't
  get "worse").

Usage
-----
    python baseline_degradation.py
    python baseline_degradation.py --scenes h08 s31 a06 i29
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from empirical_outcomes import DEFAULT_OUTPUT_PATH, load_empirical_outcomes
from hierarchy_optimizer import Cascade, PAPER_DETECTOR_COST_MS, optimize_empirical_hierarchy
from utils.labels import GLOBAL_CLASS_NAMES, INTERMEDIATE_CLASS_NAMES, KI_REGISTRY, threshold_hi_for_ki

REAL_CLASSIFIERS = ["K0", "K1", "K2", "K3", "K4", "K5", "K6"]
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")


class CascadeData:
    """Pivots one scene's empirical_outcomes payload into per-classifier
    arrays once, so replaying the cascade against it is cheap."""

    def __init__(self, outcomes_path: str | Path):
        payload = load_empirical_outcomes(outcomes_path)
        outcomes: pd.DataFrame = payload["outcomes"]
        labels: pd.DataFrame = payload["labels"]
        candidates: pd.DataFrame = payload["candidates"]

        self.n_samples = len(labels)
        self.confidence: dict[str, np.ndarray] = {}
        self.prediction: dict[str, np.ndarray] = {}
        self.cost: dict[str, float] = {}

        for cid in list(KI_REGISTRY.keys()):
            rows = outcomes[outcomes["candidate_id"] == cid].sort_values("sample_id")
            if len(rows) != self.n_samples:
                raise ValueError(
                    f"{cid}: expected {self.n_samples} rows, got {len(rows)} -- "
                    f"outcomes pickle may be stale/partial"
                )
            self.confidence[cid] = rows["confidence"].to_numpy()
            self.prediction[cid] = rows["prediction"].to_numpy()
            row = candidates[candidates["id"] == cid]
            self.cost[cid] = float(row["cost"].iloc[0]) if len(row) else float("nan")

        det_meta = payload["detector"]
        self.cost["Kdet"] = float(det_meta["cost"])

        global_lookup = {name: i for i, name in enumerate(GLOBAL_CLASS_NAMES)}
        true_label = labels["true_global_label"].map(global_lookup).to_numpy()
        if np.any(pd.isna(true_label)):
            raise ValueError("Some true_global_label values didn't map into GLOBAL_CLASS_NAMES")
        self.true_label = true_label.astype(np.int64)


def _run_chain(
    data: CascadeData,
    chain: list[str],
    thresholds: dict[str, float],
    active_idx: np.ndarray,
    total_cost: np.ndarray,
    final_pred: np.ndarray,
    specialized: dict[tuple[str, str], list[str]],
) -> np.ndarray:
    """Walk `chain` (no detector) for samples in active_idx. Returns the
    subset that reached the end without being accepted anywhere."""
    remaining = active_idx
    for ki_name in chain:
        if len(remaining) == 0:
            break
        total_cost[remaining] += data.cost[ki_name]
        conf = data.confidence[ki_name][remaining]
        accepted_mask = conf >= thresholds[ki_name]
        accepted_idx = remaining[accepted_mask]
        remaining = remaining[~accepted_mask]

        if len(accepted_idx) == 0:
            continue

        is_intermediate = KI_REGISTRY[ki_name].level == "intermediate"
        if is_intermediate:
            raw_pred = data.prediction[ki_name][accepted_idx]
            for group_idx, group_name in enumerate(INTERMEDIATE_CLASS_NAMES):
                group_mask = raw_pred == group_idx
                if not np.any(group_mask):
                    continue
                group_samples = accepted_idx[group_mask]
                spec_chain = specialized.get((ki_name, group_name))
                if spec_chain is None:
                    remaining = np.concatenate([remaining, group_samples])
                else:
                    leftover = _run_chain(
                        data, spec_chain, thresholds, group_samples,
                        total_cost, final_pred, specialized,
                    )
                    remaining = np.concatenate([remaining, leftover])
        else:
            final_pred[accepted_idx] = data.prediction[ki_name][accepted_idx]

    return remaining


def simulate(data: CascadeData, cascade: Cascade, thresholds: dict[str, float]) -> tuple[float, float]:
    """Returns (accuracy, avg_cost_ms) for replaying `cascade`'s structure
    with `thresholds` against `data` -- no model inference, pure replay of
    already-collected confidence/prediction arrays."""
    n = data.n_samples
    total_cost = np.zeros(n)
    final_pred = np.full(n, -1, dtype=np.int64)
    all_idx = np.arange(n)

    initial_without_det = [c for c in cascade.initial if c != cascade.detector]
    # cascade.specialized chains ALSO always end with the same "detector"
    # placeholder appended by HierarchyOptimizer.synthesize()/
    # _synthesize_specialized() (see hierarchy_optimizer.py: every chain,
    # initial or specialized, terminates with `next_id == self.detector_id`
    # before breaking). _run_chain only ever expects REAL classifier ids in
    # the chains it walks -- strip the placeholder from every specialized
    # chain too, not just the top-level initial one, or the recursive call
    # into a specialized chain crashes trying to look up a cost for the
    # literal string "detector" (which isn't a real classifier id).
    specialized_without_det = {
        key: [c for c in chain if c != cascade.detector]
        for key, chain in cascade.specialized.items()
    }
    leftover = _run_chain(
        data, initial_without_det, thresholds, all_idx,
        total_cost, final_pred, specialized_without_det,
    )
    if len(leftover) > 0:
        total_cost[leftover] += data.cost["Kdet"]
        final_pred[leftover] = data.prediction["Kdet"][leftover]

    accuracy = float(np.mean(final_pred == data.true_label))
    avg_cost = float(np.mean(total_cost))
    return accuracy, avg_cost


def per_classifier_stats(data: CascadeData, thresholds: dict[str, float]) -> dict[str, dict[str, float]]:
    """Mean confidence and accept rate per classifier, using the given
    threshold vector -- the actual domain-shift signal to report."""
    stats = {}
    for cid in REAL_CLASSIFIERS:
        conf = data.confidence[cid]
        stats[cid] = {
            "mean_confidence": float(np.mean(conf)),
            "accept_rate": float(np.mean(conf >= thresholds[cid])),
        }
    return stats


def run_baseline_degradation(
    scenes: list[str],
    h24_outcomes_path: str | Path = DEFAULT_OUTPUT_PATH,
    other_outcomes_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    detector_mode: str = "paper",
    detector_cost_ms: float = PAPER_DETECTOR_COST_MS,
) -> dict:
    """h24's structure and h24's thresholds, completely unchanged,
    replayed against every scene in `scenes`."""
    print("Building h24's DP-optimal cascade structure (unchanged)...")
    optimizer, cascade = optimize_empirical_hierarchy(h24_outcomes_path, detector_mode, detector_cost_ms)
    h24_thresholds = {cid: threshold_hi_for_ki(cid) for cid in REAL_CLASSIFIERS}
    print(f"h24 thresholds (fixed, used everywhere below): {h24_thresholds}\n")

    print("=== h24 (reference) ===")
    h24_data = CascadeData(h24_outcomes_path)
    h24_acc, h24_cost = simulate(h24_data, cascade, h24_thresholds)
    h24_stats = per_classifier_stats(h24_data, h24_thresholds)
    print(f"  accuracy={h24_acc:.4f}  avg_cost={h24_cost:.3f}ms")
    for cid, s in h24_stats.items():
        print(f"  {cid}: mean_conf={s['mean_confidence']:.3f}  accept_rate={s['accept_rate']:.3f}")

    results = {"h24": {"accuracy": h24_acc, "avg_cost_ms": h24_cost, "stats": h24_stats}}

    for scene in scenes:
        path = Path(other_outcomes_dir) / f"empirical_outcomes_{scene}.pkl"
        if not path.exists():
            print(f"\n=== {scene}: SKIPPED, {path} not found ===")
            continue

        print(f"\n=== {scene} ===")
        data = CascadeData(path)
        acc, cost = simulate(data, cascade, h24_thresholds)
        stats = per_classifier_stats(data, h24_thresholds)
        print(f"  accuracy={acc:.4f} (h24 was {h24_acc:.4f}, "
              f"delta={acc - h24_acc:+.4f})  avg_cost={cost:.3f}ms")
        for cid, s in stats.items():
            h24_s = h24_stats[cid]
            print(f"  {cid}: mean_conf={s['mean_confidence']:.3f} "
                  f"(h24={h24_s['mean_confidence']:.3f}, "
                  f"delta={s['mean_confidence'] - h24_s['mean_confidence']:+.3f})  "
                  f"accept_rate={s['accept_rate']:.3f} "
                  f"(h24={h24_s['accept_rate']:.3f}, "
                  f"delta={s['accept_rate'] - h24_s['accept_rate']:+.3f})")

        results[scene] = {"accuracy": acc, "avg_cost_ms": cost, "stats": stats}

    print(f"\n{'-' * 70}")
    print("SUMMARY -- this is the problem the threshold optimizer needs to fix")
    print(f"{'-' * 70}")
    print(f"{'scene':>8}  {'accuracy':>10}  {'delta_vs_h24':>12}")
    for scene, r in results.items():
        delta = r["accuracy"] - h24_acc if scene != "h24" else 0.0
        print(f"{scene:>8}  {r['accuracy']:>10.4f}  {delta:>+12.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=["h08", "s31", "a06", "i29"],
                        help="Scenes to compare against h24 (default: the ones you have so far)")
    parser.add_argument("--detector-mode", choices=["paper", "trained"], default="paper")
    parser.add_argument("--detector-cost-ms", type=float, default=PAPER_DETECTOR_COST_MS)
    args = parser.parse_args()

    run_baseline_degradation(
        scenes=args.scenes,
        detector_mode=args.detector_mode,
        detector_cost_ms=args.detector_cost_ms,
    )