# Multi-seed Stability — Comparison

Question: are accuracy-budget / stacked-Kdet speedups **stable** across `random_per_run` seeds, or seed-0 luck?

Protocol: 5 seeds, `random_per_run` 80/20, anneal 8000 iters. `blocked_per_run` seed=0 kept only as a reference (that split does **not** vary with seed).

| scene | detector | recipe | mean±std speedup | mean±std Δacc | frac Δacc≥−3pp | frac ≥2× | stable? |
|---|---|---|---:|---:|---:|---:|---|
| h24 | paper | baseline_protect | 1.0000±0.0000 | 0.0000±0.0000 | 1.0000 | 0.0000 | no |
| h24 | paper | budget_2pp | 2.9508±0.5734 | -0.0194±0.0034 | 1.0000 | 1.0000 | YES |
| h24 | paper | budget_3pp | 12.9347±4.3196 | -0.0316±0.0058 | 0.4000 | 1.0000 | no |
| h24 | paper | stacked_kdet1000_budget_2pp | 20.8956±1.8307 | -0.0206±0.0028 | 1.0000 | 1.0000 | YES |
| h08 | paper | baseline_protect | 1.0000±0.0000 | 0.0000±0.0000 | 1.0000 | 0.0000 | no |
| h08 | paper | budget_2pp | 1.1754±0.0280 | -0.0181±0.0031 | 1.0000 | 0.0000 | no |
| h08 | paper | budget_3pp | 1.2944±0.0390 | -0.0286±0.0032 | 0.8000 | 0.0000 | no |
| h08 | paper | stacked_kdet1000_budget_2pp | 11.1523±0.3068 | -0.0188±0.0034 | 1.0000 | 1.0000 | YES |
| h24 | trained | baseline_protect | 1.0000±0.0000 | 0.0000±0.0000 | 1.0000 | 0.0000 | no |
| h24 | trained | budget_2pp | 1.3783±0.0311 | -0.0159±0.0051 | 1.0000 | 0.0000 | no |
| h24 | trained | budget_3pp | 1.5653±0.0305 | -0.0267±0.0053 | 0.6000 | 0.0000 | no |
| h24 | trained | stacked_kdet1000_budget_2pp | skipped | | | | |
| h08 | trained | baseline_protect | 1.0000±0.0000 | 0.0000±0.0000 | 1.0000 | 0.0000 | no |
| h08 | trained | budget_2pp | 1.6100±0.0672 | -0.0191±0.0039 | 1.0000 | 0.0000 | no |
| h08 | trained | budget_3pp | 1.8885±0.0898 | -0.0293±0.0055 | 0.4000 | 0.2000 | no |
| h08 | trained | stacked_kdet1000_budget_2pp | skipped | | | | |

## Verdict

- **h24/paper** stable recipes: `budget_2pp`, `stacked_kdet1000_budget_2pp`
- **h08/paper** stable recipes: `stacked_kdet1000_budget_2pp`
- **h24/trained** stable recipes: _none_
- **h08/trained** stable recipes: _none_

If `budget_3pp` or `stacked_kdet1000_budget_2pp` is **not** stable, treat the single-seed vacation numbers as provisional for a paper.

