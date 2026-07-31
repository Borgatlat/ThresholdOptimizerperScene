# Joint-Opt Bake-off — Teammate Handoff (FILLED 2026-07-30)

**Status:** ready to share.  
**Locked \(F\):** minimize validation expected cost (ms) s.t. \(\mathrm{Acc}_{\mathrm{val}} \ge 0.98337\); report holdout separately.  
**K1:** removed. Scene: **h24 only** for this bake-off.

---

## 1. One-sentence claim

> On h24 under the locked contract, nested joint search (**A4**) matches the teammate SA bank (~1627 ms val / ~1531 ms holdout at Acc 0.983376) by changing layout to `K0→K3→detector`; threshold-only **A1** is feasible but ~28 ms more expensive.

---

## 2. Toy convergence (approximator proof)

| Method | Result | Matches exhaustive? |
|--------|--------|---------------------|
| Exhaustive vs SA (A1) | cost_gap = 0 | **Y** |
| A4 layout family | layouts_match = true, cost_gap = 0 | **Y** |

Source: `checkpoints/joint_opt/toy/convergence.json` (`passed: true`).

---

## 3. Real-scene bake-off (h24)

| Scene | Method | Joint? | Val Acc | Val cost (ms) | Holdout Acc | Holdout cost (ms) | Feasible? | Initial layout |
|-------|--------|--------|---------|---------------|-------------|-------------------|-----------|----------------|
| h24 | A1 thresholds-only | no | 0.983376 | 1655.51 | 0.985161 | 1557.62 | yes | K0→K3→K2→detector |
| h24 | A2 layout-only | no | 0.975952 | 4101.05 | 0.978065 | 4116.16 | **no** | K3→K2→detector |
| h24 | A3 alternating | **yes** | 0.983376 | 1647.27 | 0.985161 | 1557.42 | yes | K0→K3→K2→detector |
| h24 | **A4 nested** | **yes** | 0.983376 | **1627.23** | 0.984516 | **1530.94** | yes | **K0→K3→detector** |

**Winner under \(F\):** **A4** (lowest feasible val cost).  
**Target Acc floor hit:** yes (val Acc = 0.983376 ≥ 0.98337).

Artifacts:
- `checkpoints/joint_opt/h24/a4_nested.json` ← send this as the joint bank
- `checkpoints/joint_opt/h24/joint_bakeoff.md`
- `checkpoints/joint_opt/h24/a1_threshold.json` (thresholds-only baseline)

---

## 4. Deltas (cost; lower is better under \(F\))

| Compare | Val cost delta | Holdout cost delta | Note |
|---------|----------------|--------------------|------|
| A4 − A1 | **−28.3 ms** | **−26.7 ms** | joint layout change helps |
| A3 − A1 | −8.2 ms | −0.2 ms | small gain, same deep layout |
| A2 alone | infeasible | — | must retune H after changing S |

---

## 5. Five meeting bullets

1. **A1 ≠ joint.** A1 freezes DP layout and only anneals thresholds; high \(H\approx 1\) soft-disables Kis but cannot remove them.
2. **A3/A4 are the joint optimizers** (layout \(S\) and thresholds \(H\) both change).
3. **A4 wins** under the locked objective and **matches teammate holdout cost ~1530.94 ms** with initial `K0→K3→detector`.
4. **A2 without SA fails** the Acc floor — joint retune of \(H\) after layout change is required.
5. **Next paper step (not this bake-off):** live scene detector should hot-swap full banks \((H,S)\), not thresholds only.

---

## 6. Caveats

- [x] A3 is block-coordinate — not guaranteed global optimum (A4 found a better discrete layout).
- [x] Toy proves SA≈exhaustive on a tiny grid; not a neural-net claim.
- [x] K1 removed for all rows.
- [x] High thresholds (e.g. K0≈0.996) also appear in the teammate bank — expected under Acc-floor SA.
- [x] Do not claim joint opt as sole paper novelty; lead with scene-aware maintenance + OOD.

---

## 7. What to send your teammate

1. This file  
2. `checkpoints/joint_opt/h24/a4_nested.json`  
3. `checkpoints/joint_opt/h24/joint_bakeoff.json`  

One-liner chat message:

> Joint bake-off done on h24. A4 (nested layout×SA) is the joint method; val Acc 0.983376, val cost 1627.23, holdout Acc 0.984516, holdout cost 1530.94 — matches your bank. A1 was thresholds-only and ~28 ms slower.

---

## 8. Bonus today: live layout swap (s31, n=300)

Wired `swap_layout=True` so Detector/Oracle can hot-swap `scene_banks/<scene>/synthesized_cascades.json` plus H.

| Condition | Cascade Acc | Mean latency (ms) | Layout swap? |
|-----------|-------------|-------------------|--------------|
| B (fixed) | 0.340 | 115 | no |
| Detector H-only | 0.830 | 7773 | no |
| **Detector H+S** | **0.860** | 7754 | **yes** |
| Oracle H-only | 0.843 | 6086 | no |
| Oracle H+S | 0.847 | 6045 | yes |

**+3.0 pts** Detector Acc from adding layout swap on s31 (same H bank).  
Artifact: `checkpoints/joint_opt/h24/layout_swap_ablation_s31.json`
