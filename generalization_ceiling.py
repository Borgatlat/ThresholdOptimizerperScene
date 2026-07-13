"""Answers a real methodological question: given every classifier (K0-K6,
Kdet) is trained ONLY on h24, is threshold re-optimization actually a
fair/meaningful fix on other scenes, or is the degradation we're seeing a
symptom of the classifiers themselves not generalizing at all -- something
no threshold vector could ever fix?

The distinction that matters
-----------------------------
1. CALIBRATION DRIFT: the classifier still ranks correct answers above
   incorrect ones about as well as it did on h24, but the confidence
   NUMBER attached to that ranking has shifted -- so a threshold tuned on
   h24's confidence distribution no longer means the same thing. This is
   exactly what threshold re-optimization (the paper's main contribution)
   fixes.
2. GENUINE FEATURE FAILURE: the classifier's learned features don't
   discriminate well under the new scene's conditions AT ALL, regardless
   of threshold. No threshold vector can fix this -- only retraining can.

baseline_degradation.py's threshold-dependent accept-rate/accuracy numbers
can't tell these apart on their own: a big accuracy drop could be either.
This script isolates it by computing each classifier's RAW, THRESHOLD-FREE
accuracy per scene -- literally "was the argmax prediction correct",
ignoring accept/reject entirely -- and comparing that to h24's own raw
accuracy. If raw accuracy holds up while threshold-dependent accept
rate/accuracy swings, that's (1), fixable. If raw accuracy itself
collapses, that's (2), a hard ceiling no threshold optimizer can cross --
worth reporting explicitly rather than treated as something more
threshold-tuning would eventually solve.

Also reports Kdet's own raw accuracy per scene separately: Kdet is the
final fallback for every sample that never gets accepted, and it's ALSO
only trained on h24. If Kdet's own accuracy degrades on a scene, no amount
of retuning K0-K6's thresholds can fix that -- you're just choosing how
often to hand samples to an already-degraded fallback.

Usage
-----
    python generalization_ceiling.py --scenes h08 s31 a06 i29
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from empirical_outcomes import DEFAULT_OUTPUT_PATH, load_empirical_outcomes
from utils.labels import GLOBAL_CLASS_NAMES, INTERMEDIATE_CLASS_NAMES

REAL_CLASSIFIERS = ["K0", "K1", "K2", "K3", "K4", "K5", "K6"]
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")

# Which intermediate group each specialized classifier's samples should be
# restricted to -- K4 (SUV specialized) is only meaningful/trained on
# actual SUV-group samples, etc. Matches utils.labels.KI_REGISTRY's
# specialized classifiers.
SPECIALIZED_GROUP = {"K4": "suv", "K6": "coupe", "K5": "coupe"}


def _load_scene_frame(outcomes_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = load_empirical_outcomes(outcomes_path)
    return payload["outcomes"], payload["labels"]


def raw_accuracy_per_classifier(outcomes_path: str | Path) -> dict[str, dict]:
    """Threshold-free accuracy per classifier: is the raw argmax prediction
    correct, regardless of whether the classifier would have accepted or
    IDK'd. K0/K1 are compared against the true INTERMEDIATE label (that's
    what they predict); K2/K3 against the true GLOBAL label; K4/K5/K6
    against the true GLOBAL label but restricted to samples whose true
    intermediate group matches that classifier's specialization (a coupe
    specialist has no meaningful "raw accuracy" on SUV samples).
    """
    outcomes, labels = _load_scene_frame(outcomes_path)

    global_lookup = {name: i for i, name in enumerate(GLOBAL_CLASS_NAMES)}
    intermediate_lookup = {name: i for i, name in enumerate(INTERMEDIATE_CLASS_NAMES)}
    true_global = labels["true_global_label"].map(global_lookup).to_numpy()
    true_intermediate = labels["true_intermediate_label"].map(intermediate_lookup).to_numpy()

    results = {}
    for cid in REAL_CLASSIFIERS:
        rows = outcomes[outcomes["candidate_id"] == cid].sort_values("sample_id")
        pred = rows["prediction"].to_numpy()

        if cid in ("K0", "K1"):
            correct = pred == true_intermediate
            n = len(correct)
        elif cid in SPECIALIZED_GROUP:
            group = SPECIALIZED_GROUP[cid]
            group_idx = intermediate_lookup[group]
            mask = true_intermediate == group_idx
            correct = pred[mask] == true_global[mask]
            n = int(mask.sum())
        else:  # K2, K3: global 5-class
            correct = pred == true_global
            n = len(correct)

        results[cid] = {
            "raw_accuracy": float(np.mean(correct)) if n > 0 else float("nan"),
            "n": n,
        }

    # Kdet: same global comparison, no restriction (it's the universal fallback)
    kdet_rows = outcomes[outcomes["candidate_id"] == "Kdet"].sort_values("sample_id")
    kdet_pred = kdet_rows["prediction"].to_numpy()
    results["Kdet"] = {
        "raw_accuracy": float(np.mean(kdet_pred == true_global)),
        "n": len(true_global),
    }

    return results


def run_ceiling_analysis(
    scenes: list[str],
    h24_outcomes_path: str | Path = DEFAULT_OUTPUT_PATH,
    other_outcomes_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> dict:
    print("=== h24 (reference) raw accuracy per classifier ===")
    h24_raw = raw_accuracy_per_classifier(h24_outcomes_path)
    for cid, r in h24_raw.items():
        print(f"  {cid}: raw_accuracy={r['raw_accuracy']:.4f}  (n={r['n']})")

    results = {"h24": h24_raw}
    for scene in scenes:
        path = Path(other_outcomes_dir) / f"empirical_outcomes_{scene}.pkl"
        if not path.exists():
            print(f"\n=== {scene}: SKIPPED, {path} not found ===")
            continue

        print(f"\n=== {scene} ===")
        scene_raw = raw_accuracy_per_classifier(path)
        for cid, r in scene_raw.items():
            h24_r = h24_raw[cid]["raw_accuracy"]
            delta = r["raw_accuracy"] - h24_r
            flag = "  <-- LARGE DROP, likely genuine feature failure, not just calibration" \
                if delta < -0.15 else ""
            print(f"  {cid}: raw_accuracy={r['raw_accuracy']:.4f} "
                  f"(h24={h24_r:.4f}, delta={delta:+.4f}, n={r['n']}){flag}")
        results[scene] = scene_raw

    print(f"\n{'-' * 70}")
    print("INTERPRETATION GUIDE")
    print(f"{'-' * 70}")
    print("Small raw-accuracy drops (roughly <0.10-0.15) alongside the large")
    print("threshold-dependent accuracy drops seen in baseline_degradation.py")
    print("=> mostly CALIBRATION DRIFT -- your threshold optimizer's actual target.")
    print("Large raw-accuracy drops here")
    print("=> genuine feature/generalization failure for that classifier on that")
    print("   scene -- no threshold vector can fix this; it's a hard ceiling on")
    print("   how much accuracy threshold-only adaptation can ever recover there.")
    print("Kdet's own raw accuracy matters separately: it's the universal fallback,")
    print("also h24-only trained -- if IT degrades, retuning K0-K6's thresholds")
    print("can change how OFTEN samples reach Kdet, but never fix what Kdet says")
    print("once they get there.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=["h08", "s31", "a06", "i29"])
    args = parser.parse_args()
    run_ceiling_analysis(scenes=args.scenes)