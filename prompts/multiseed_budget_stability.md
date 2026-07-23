# Prompt: Multi-seed stability of accuracy-budget (+ winning stacked) speedups

Copy-paste this into an agent (or keep as the experiment contract).

---

Repo: ThresholdOptimizerperScene, branch off `cursor/threshold-optimizer-experiments-8590`.

## GOAL
Check whether the big speedups we saw from accuracy budgets (and the
winning paper stacked lower-Kdet recipe) are **stable across random seeds**,
or just lucky on seed=0 / one holdout split.

Question: across multiple train/holdout partitions, do we still get large
speedups at 2–3pp budget (and at stacked Kdet=1000 + 2pp on paper) with
holdout Δacc staying near the budget — without a major accuracy collapse?

## CONTEXT (already done — do not redo)
- `experiment_accuracy_budget_pareto.py`: on h24/paper, 2–3pp budget gave
  ~2.7×–8× vs protect-baseline with ~2–3pp holdout acc drop (seed 0,
  blocked_per_run).
- `experiment_stacked_recipe.py`: paper lower-Kdet stacks often win; on
  trained, stacking usually ties budget-only.
- `threshold_optimizer.py`: `split_empirical_outcomes` —
  `blocked_per_run` is **deterministic** (seed ignored for the mask);
  use `random_per_run` when you want different partitions per seed.
- DO NOT train per-scene classifiers
- DO NOT implement scene switching
- DO NOT re-run full prior suites; reuse their APIs/helpers

## METHOD
Primary focus: **scene=h24** first. Also report h08 as a second scene
(same protocol). detector_mode: **paper** (required) and **trained**
(secondary reference).

Seeds: `0, 1, 2, 3, 4` (five seeds).

For each (scene, detector_mode, seed):
  1) Split outcomes with **`random_per_run` 80/20** and that seed
     (so the holdout partition actually changes). Also store one
     `blocked_per_run` reference at seed=0 for comparison to prior runs.
  2) On validation, synthesize DP once at the recipe's Kdet cost
     (paper default 10_000 ms; stacked-low uses 1_000 ms).
  3) Run these RECIPES (named configs), same spirit as stacked/budget:
       A) `baseline_protect` — cost=10000 (paper) / measured (trained),
          DP order, budget=0pp
       B) `budget_2pp` — same layout, budget=2pp
       C) `budget_3pp` — same layout, budget=3pp
       D) `stacked_kdet1000_budget_2pp` — **paper only**
          cost=1000, DP order re-synthesized at 1000, budget=2pp
  4) Anneal thresholds on validation with that floor (8000 iters);
     freeze; evaluate on that seed's holdout.
  5) Metrics per (recipe, seed) on holdout:
       micro / macro / worst-class accuracy
       expected cost ms
       speedup vs that seed's `baseline_protect`
       Δacc / Δcost vs that seed's `baseline_protect`
       feasibility (holdout micro >= floor)

Aggregate across seeds per (scene, detector_mode, recipe):
  - mean ± std of holdout acc, cost, speedup, Δacc
  - fraction of seeds with Δacc >= −3pp vs that seed's baseline_protect
  - fraction of seeds where budget_3pp speedup >= 2×

Baselines:
  - per-seed `baseline_protect`
  - compare aggregates to the single-seed accuracy-budget / stacked numbers

## DELIVERABLES
- `experiment_multiseed_stability.py`
- results JSON under `checkpoints/threshold_experiments/multiseed_stability/`
- short `COMPARISON.md` (per scene/mode: mean±std speedup & Δacc; is the
  effect stable?)
- paper-ready matplotlib PNG(s) in `checkpoints/figures/threshold_experiments/`
  (e.g. h24 box/strip of speedup by recipe across seeds; Δacc by recipe)
- commit + push + update PR

## ACCEPTANCE
- every recipe for a (scene, detector_mode, seed) shares that seed's holdout
- seeds use `random_per_run` so partitions differ (document that
  `blocked_per_run` would NOT vary with seed)
- report clearly if speedups are unstable (high std / frequent >3pp drops)
  — negative result is fine
- comments explain WHY multi-seed matters (one split can overfit; anneal is
  stochastic; blocked_per_run hides partition variance)

## CONSTRAINTS
- reuse `split_empirical_outcomes`, `FixedLayoutThresholdEvaluator`,
  `optimize_fixed_layout_thresholds_simulated_annealing`, `make_cascade` /
  HierarchyOptimizer.synthesize
- default anneal iterations 8000
- cached empirical outcomes only; no new neural inference
- keep scene set small (h24 + h08) — this is a stability check, not a full redo
