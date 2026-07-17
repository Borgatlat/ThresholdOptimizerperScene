# Sequence-Order Ablations — Comparison

Question: for a fixed set of classifiers on the initial chain, how much does **order** matter after threshold retuning?

All orders for a given (scene, detector_mode) share the same `blocked_per_run` holdout. DP order is always included.

Ranking: feasible policies first, then lower holdout cost. Reference orders (`ref_*`) may change membership (not only permutation).

| scene | detector | #orders | DP order | DP cost | DP feas | best label | best order | best cost | best feas | Δcost vs DP | #feas cheaper | #near DP |
|---|---|---:|---|---:|---|---|---|---:|---|---:|---:|---:|
| a06 | paper | 29 | `K0→K3→K2→K1` | 3374.1963 | False | perm_K2>K3>K0>K1 | `K2→K3→K0→K1` | 2745.0074 | False | -629.1889 | 0 | 0 |
| a06 | trained | 6 | `K0` | 9.4578 | False | ref_K0>K3 | `K0→K3` | 8.9191 | True | -0.5386 | 1 | 1 |
| h08 | paper | 29 | `K0→K3→K2→K1` | 2342.3768 | True | perm_K2>K0>K3>K1 | `K2→K0→K3→K1` | 2296.8530 | True | -45.5238 | 5 | 9 |
| h08 | trained | 6 | `K0` | 15.9715 | True | ref_K0>K3 | `K0→K3` | 10.6172 | True | -5.3542 | 5 | 5 |
| h24 | paper | 29 | `K0→K3→K2→K1` | 704.9883 | True | swap_1_2 | `K0→K2→K3→K1` | 702.9874 | True | -2.0009 | 1 | 4 |
| h24 | trained | 6 | `K0→K3` | 8.8955 | True | dp_order | `K0→K3` | 8.8955 | True | 0.0000 | 0 | 0 |
| i29 | paper | 29 | `K0→K2→K3→K1` | 5783.9548 | True | perm_K0>K1>K2>K3 | `K0→K1→K2→K3` | 5635.3383 | True | -148.6165 | 20 | 23 |
| i29 | trained | 6 | `K0` | 15.4674 | False | ref_K0>K3 | `K0→K3` | 7.4283 | False | -8.0392 | 0 | 0 |
| s31 | paper | 29 | `K0→K3→K2→K1` | 4105.9331 | False | perm_K0>K1>K2>K3 | `K0→K1→K2→K3` | 4004.2470 | True | -101.6861 | 2 | 2 |
| s31 | trained | 6 | `K0` | 23.1079 | True | ref_K0>K3 | `K0→K3` | 11.8349 | True | -11.2731 | 3 | 3 |

## Verdict

Across 10 (scene, detector) settings:
- DP order ranked best in **1** settings
- Some other order ranked above DP in **9** settings
- Of those, **7** were feasible *and* cheaper than DP

Δcost < 0 means the ranked-best policy was cheaper than DP. When both DP and best are infeasible, treat cost wins cautiously (they may buy speed by missing the accuracy target). Many near-DP feasible orders ⇒ order is weakly identified after annealing.

