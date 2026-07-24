# Prompt: Kdet × accuracy-budget interaction (factorial)

Copy-paste this into an agent (or keep as the experiment contract).

---

Repo: ThresholdOptimizerperScene, branch off `cursor/threshold-optimizer-experiments-8590`.

## GOAL
Separate the two paper speed levers we keep mixing: **detector cost (Kdet)**
and **accuracy budget**.

Question: on paper mode, is the huge speedup from “stacked lower-Kdet + 2pp”
mostly **Kdet alone**, mostly **budget alone**, or a real **interaction**
(synergy — better together than either lever predicts)?

Also: which (Kdet, budget) cells stay within −3pp of protect-baseline on
holdout, and which cells are fragile?

## CONTEXT (already done — do not redo)
- `experiment_accuracy_budget_pareto.py`: at Kdet=10_000, 2–3pp budgets buy
  large speedups on h24/paper (single seed, blocked_per_run).
- `experiment_detector_cost_sensitivity.py`: lower paper Kdet changes DP
  structure / accounted cost; accuracy often stays flat.
- `experiment_stacked_recipe.py`: combining order+budget+Kdet — paper
  lower-Kdet stacks often win; order+budget alone rarely beats budget-only.
- `experiment_multiseed_stability.py`: on h24/paper across 5 `random_per_run`
  seeds, `budget_2pp` (~3×) and `stacked_kdet1000_budget_2pp` (~21×) are
  **stable**; `budget_3pp` is **unstable** (high std, often >3pp Δacc).
- Important API fact: `blocked_per_run` ignores seed for the holdout mask;
  use `random_per_run` when you need partition variance.
- DO NOT train per-scene classifiers
- DO NOT implement scene switching
- DO NOT re-run full prior suites; reuse their APIs/helpers
- DO NOT redo sequence-order search here (order was a weak add-on in stacked)

## METHOD
Primary: **scene=h24**, **detector_mode=paper** (required).
Secondary: **h08/paper** with the same grid (smaller claim check).
Optional reference: **trained** only at the scene’s measured Kdet (no
synthetic Kdet grid — still report one budget sweep at measured cost so
paper vs trained stays comparable).

### Factorial grid (paper only)
Freeze DP layout **per Kdet** (re-synthesize when Kdet changes; keep that
layout for all budgets at that Kdet).

  Kdet_ms ∈ {10_000, 5_000, 2_000, 1_000}
  budget ∈ {0.00, 0.01, 0.02, 0.03}   # 0 / 1 / 2 / 3 pp

For each (scene, seed, Kdet, budget):
  1) Split outcomes **once per seed** with `random_per_run` 80/20
     (SAME split for every cell in that seed — critical for fair comparison).
  2) On validation, `HierarchyOptimizer(..., detector_cost_ms=Kdet).synthesize()`.
  3) Anneal thresholds with floor = collection_micro − budget (8000 iters,
     `constraint_metric=micro`).
  4) Freeze (cascade, thresholds); evaluate on that seed’s holdout.

Seeds: `0, 1, 2` (three is enough for an interaction screen; cite
multiseed if you need 5 later). Also store one `blocked_per_run` seed=0
reference row for continuity with older figures.

### Metrics per cell (holdout), then aggregate across seeds
  - micro / macro / worst-class accuracy
  - expected cost ms
  - speedup vs that seed’s (Kdet=10_000, budget=0) **protect cell**
  - Δacc / Δcost vs that protect cell
  - feasibility (holdout micro ≥ floor)
  - fraction of seeds with Δacc ≥ −3pp vs protect

### Interaction analysis (required in COMPARISON.md)
Define, per seed, on the protect cell cost C00 and each cell cost C(k,b):

  speedup(k,b) = C00 / C(k,b)

Report whether levers are roughly **multiplicative / additive in log-cost**:
  - Kdet-only effect at budget 0: speedup(k, 0) vs speedup(10000, 0)
  - budget-only effect at Kdet 10000: speedup(10000, b) vs speedup(10000, 0)
  - predicted independent combo (optional simple model):
      pred(k,b) ≈ speedup(k,0) * speedup(10000,b)
    (works if effects multiply on the speedup scale)
  - residual: speedup(k,b) − pred(k,b)
    → positive residual = synergy; near zero = independent; negative = interference

Call out the previously winning cell `(1000, 2pp)` explicitly vs
`(1000, 0pp)` and `(10000, 2pp)`.

## DELIVERABLES
- `experiment_kdet_budget_interaction.py`
- results JSON under `checkpoints/threshold_experiments/kdet_budget_interaction/`
- short `COMPARISON.md` with:
  - heatmap-style markdown table (mean speedup; mean Δacc) for h24/paper
  - verdict: Kdet-alone vs budget-alone vs synergy at (1000, 2pp)
  - which cells are “safe” (mean Δacc ≥ −3pp and ≥80% seeds pass that gate)
- paper-ready matplotlib PNG(s) in `checkpoints/figures/threshold_experiments/`:
  - heatmap of mean speedup (Kdet × budget) for h24/paper
  - heatmap of mean Δacc (pp)
  - line plot: speedup vs budget for each Kdet curve
- commit + push + update PR

## ACCEPTANCE
- every (Kdet, budget) cell for a given seed shares that seed’s holdout split
- DP is re-synthesized when Kdet changes; layout is frozen across budgets
  at the same Kdet
- trained mode does **not** invent synthetic Kdet values
- clearly state if (1000, 2pp) wins only because Kdet accounting changed
  (Kdet-alone already huge) — negative “no synergy” result is fine
- comments explain WHY an interaction test matters (stacking confounded the
  two levers; multi-seed showed 3pp is fragile, so the grid must mark unsafe
  cells)

## CONSTRAINTS
- reuse `split_empirical_outcomes`, `FixedLayoutThresholdEvaluator`,
  `optimize_fixed_layout_thresholds_simulated_annealing`,
  `HierarchyOptimizer.synthesize` / `make_cascade`
- default anneal iterations 8000
- cached empirical outcomes only; no new neural inference
- keep scope to h24 (+ h08); do not expand to all scenes unless h24 shows
  a surprising interaction that needs confirmation
- no order search, no alternating structure, no per-scene training

## WHY THIS IS THE RIGHT NEXT EXPERIMENT (for the human)
Stacked recipes answered “does combining help?” but confounded Kdet with
budget. Multi-seed answered “is 2pp/3pp luck?” This prompt answers the
causal paper question: **what should we claim — cheaper detector, small
accuracy give, or both together?**
