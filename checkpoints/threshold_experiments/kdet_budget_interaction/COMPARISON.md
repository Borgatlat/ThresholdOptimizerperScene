# Kdet × Accuracy-Budget Interaction — Comparison

Question: is `(Kdet=1000, budget=2pp)` mostly Kdet alone, mostly budget alone, or synergy?

Protocol: seeds `{0,1,2}`, `random_per_run` 80/20, anneal 8000. Protect cell = `(Kdet=10000, budget=0pp)`.

## h24/paper — mean speedup heatmap

| Kdet \ budget | 0pp | 1pp | 2pp | 3pp |
|---|---:|---:|---:|---:|
| 10000 | 1.00\* | 1.60\* | 3.05\* | 15.31 |
| 5000 | 2.04\* | 3.09\* | 5.87\* | 24.97 |
| 2000 | 4.84\* | 6.76\* | 12.63\* | 36.74 |
| 1000 | 8.97\* | 12.48\* | 21.88\* | 50.75 |

\* = safe (mean Δacc ≥ −3pp and ≥80% seeds pass).

## h24/paper — mean Δacc (pp)

| Kdet \ budget | 0pp | 1pp | 2pp | 3pp |
|---|---:|---:|---:|---:|
| 10000 | 0.00 | -0.84 | -1.98 | -3.38 |
| 5000 | 0.15 | -0.82 | -2.02 | -3.40 |
| 2000 | 0.17 | -1.20 | -2.04 | -3.40 |
| 1000 | 0.22 | -0.88 | -2.13 | -3.38 |

### Headline (h24/paper)

- `(1000, 0pp)` speedup=8.9688 safe=True
- `(10000, 2pp)` speedup=3.0470 safe=True
- `(1000, 2pp)` speedup=21.8836 residual_vs_independent=-5.4780 safe=True
- **Verdict:** `both_contribute_near_independent`

## h08/paper — mean speedup heatmap

| Kdet \ budget | 0pp | 1pp | 2pp | 3pp |
|---|---:|---:|---:|---:|
| 10000 | 1.00\* | 1.07\* | 1.16\* | 1.25\* |
| 5000 | 1.99\* | 2.11\* | 2.29\* | 2.47\* |
| 2000 | 4.86\* | 5.15\* | 5.62\* | 6.00\* |
| 1000 | 9.38\* | 10.08\* | 10.85\* | 11.61\* |

\* = safe (mean Δacc ≥ −3pp and ≥80% seeds pass).

## h08/paper — mean Δacc (pp)

| Kdet \ budget | 0pp | 1pp | 2pp | 3pp |
|---|---:|---:|---:|---:|
| 10000 | 0.00 | -0.89 | -1.79 | -2.66 |
| 5000 | 0.08 | -0.83 | -1.82 | -2.51 |
| 2000 | 0.16 | -0.84 | -1.75 | -2.59 |
| 1000 | 0.02 | -0.75 | -1.68 | -2.54 |

### Headline (h08/paper)

- `(1000, 0pp)` speedup=9.3779 safe=True
- `(10000, 2pp)` speedup=1.1565 safe=True
- `(1000, 2pp)` speedup=10.8471 residual_vs_independent=0.0018 safe=True
- **Verdict:** `mostly_kdet_alone`

## Safe cells

- **h24/paper**: `kdet_10000__budget_0pp`, `kdet_10000__budget_1pp`, `kdet_10000__budget_2pp`, `kdet_5000__budget_0pp`, `kdet_5000__budget_1pp`, `kdet_5000__budget_2pp`, `kdet_2000__budget_0pp`, `kdet_2000__budget_1pp`, `kdet_2000__budget_2pp`, `kdet_1000__budget_0pp`, `kdet_1000__budget_1pp`, `kdet_1000__budget_2pp`
- **h08/paper**: `kdet_10000__budget_0pp`, `kdet_10000__budget_1pp`, `kdet_10000__budget_2pp`, `kdet_10000__budget_3pp`, `kdet_5000__budget_0pp`, `kdet_5000__budget_1pp`, `kdet_5000__budget_2pp`, `kdet_5000__budget_3pp`, `kdet_2000__budget_0pp`, `kdet_2000__budget_1pp`, `kdet_2000__budget_2pp`, `kdet_2000__budget_3pp`, `kdet_1000__budget_0pp`, `kdet_1000__budget_1pp`, `kdet_1000__budget_2pp`, `kdet_1000__budget_3pp`
- **h24/trained**: `measured_kdet_28.2808__budget_0pp`, `measured_kdet_28.2808__budget_1pp`, `measured_kdet_28.2808__budget_2pp`
- **h08/trained**: `measured_kdet_28.2808__budget_0pp`, `measured_kdet_28.2808__budget_1pp`, `measured_kdet_28.2808__budget_2pp`

