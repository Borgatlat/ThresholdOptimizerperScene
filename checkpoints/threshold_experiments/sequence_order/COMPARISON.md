# Sequence-Order Ablations — Comparison

Question: for a fixed set of classifiers on the initial chain, how much does **order** matter after threshold retuning?

All orders for a given (scene, detector_mode) share the same `blocked_per_run` holdout. DP order is always included.

| scene | detector | #orders | DP order | DP cost | best label | best order | best cost | Δcost vs DP | #cheaper | #near DP (±1%) |
|---|---|---:|---|---:|---|---|---:|---:|---:|---:|
| a06 | paper | 29 | `K0→K3→K2→K1` | 3374.1963 | perm_K2>K3>K0>K1 | `K2→K3→K0→K1` | 2745.0074 | -629.1889 | 0 | 0 |
| a06 | trained | 6 | `K0` | 9.4578 | ref_K0>K3 | `K0→K3` | 8.9191 | -0.5386 | 1 | 1 |
| h08 | paper | 29 | `K0→K3→K2→K1` | 2342.3768 | perm_K2>K0>K3>K1 | `K2→K0→K3→K1` | 2296.8530 | -45.5238 | 5 | 9 |
| h08 | trained | 6 | `K0` | 15.9715 | ref_K0>K3 | `K0→K3` | 10.6172 | -5.3542 | 5 | 5 |
| h24 | paper | 29 | `K0→K3→K2→K1` | 704.9883 | swap_1_2 | `K0→K2→K3→K1` | 702.9874 | -2.0009 | 1 | 4 |
| h24 | trained | 6 | `K0→K3` | 8.8955 | dp_order | `K0→K3` | 8.8955 | 0.0000 | 0 | 0 |
| i29 | paper | 29 | `K0→K2→K3→K1` | 5783.9548 | perm_K0>K1>K2>K3 | `K0→K1→K2→K3` | 5635.3383 | -148.6165 | 20 | 23 |
| i29 | trained | 6 | `K0` | 15.4674 | ref_K0>K3 | `K0→K3` | 7.4283 | -8.0392 | 0 | 0 |
| s31 | paper | 29 | `K0→K3→K2→K1` | 4105.9331 | perm_K0>K1>K2>K3 | `K0→K1→K2→K3` | 4004.2470 | -101.6861 | 2 | 2 |
| s31 | trained | 6 | `K0` | 23.1079 | ref_K0>K3 | `K0→K3` | 11.8349 | -11.2731 | 3 | 3 |

## Verdict

Across 10 (scene, detector) settings:
- DP order ranked best in **1** settings
- Some other order beat DP on holdout cost in **9** settings

Δcost < 0 means the best non-default ranking beat DP (cheaper). Many near-DP orders ⇒ order is weakly identified after annealing.

