# Constraint Variants — Comparison

Question: if we protect **macro** or **worst-class** accuracy (not only micro) while annealing, how do holdout accuracy, fairness, and cost change?

Same DP layout + same `blocked_per_run` holdout within each (scene, detector_mode). Skipped = collection baseline already below the requested fixed floor.

| scene | detector | variant | floor | holdout micro | macro | worst | cost (ms) | feas | Δcost vs micro_base | Δworst vs micro_base |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| h24 | paper | micro_baseline | 0.9655 | 0.9671 | 0.9659 | 0.9264 | 696.4758 | True | 0.0000 | 0.0000 |
| h24 | paper | macro_baseline | 0.9642 | 0.9671 | 0.9659 | 0.9264 | 704.9474 | True | 8.4716 | 0.0000 |
| h24 | paper | worst_baseline | 0.9364 | 0.9639 | 0.9629 | 0.9292 | 851.6854 | False | 155.2096 | 0.0027 |
| h24 | paper | micro_0.95 | 0.9500 | 0.9471 | 0.9451 | 0.8801 | 299.1161 | False | -397.3597 | -0.0463 |
| h24 | paper | macro_0.95 | 0.9500 | 0.9490 | 0.9471 | 0.8856 | 323.9599 | False | -372.5159 | -0.0409 |
| h24 | paper | worst_0.90 | 0.9000 | 0.9523 | 0.9506 | 0.8910 | 372.9631 | False | -323.5127 | -0.0354 |
| h08 | paper | micro_baseline | 0.9071 | 0.9154 | 0.9151 | 0.8862 | 2334.0031 | True | 0.0000 | 0.0000 |
| h08 | paper | macro_baseline | 0.9067 | 0.9150 | 0.9148 | 0.8855 | 2334.0009 | True | -0.0021 | -0.0007 |
| h08 | paper | worst_baseline | 0.8844 | 0.9093 | 0.9090 | 0.8669 | 2336.8755 | False | 2.8724 | -0.0193 |
| h08 | paper | micro_0.95 | | skipped | | | | | | |
| h08 | paper | macro_0.95 | | skipped | | | | | | |
| h08 | paper | worst_0.90 | | skipped | | | | | | |
| s31 | paper | micro_baseline | 0.7844 | 0.7825 | 0.7825 | 0.7614 | 4009.2630 | False | 0.0000 | 0.0000 |
| s31 | paper | macro_baseline | 0.7841 | 0.7827 | 0.7827 | 0.7557 | 4023.0089 | False | 13.7459 | -0.0057 |
| s31 | paper | worst_baseline | 0.7662 | 0.7742 | 0.7742 | 0.7614 | 3974.4960 | False | -34.7671 | 0.0000 |
| s31 | paper | micro_0.95 | | skipped | | | | | | |
| s31 | paper | macro_0.95 | | skipped | | | | | | |
| s31 | paper | worst_0.90 | | skipped | | | | | | |
| a06 | paper | micro_baseline | 0.7666 | 0.6124 | 0.6182 | 0.4232 | 3023.6956 | False | 0.0000 | 0.0000 |
| a06 | paper | macro_baseline | 0.7629 | 0.6399 | 0.6449 | 0.4401 | 3409.4486 | False | 385.7530 | 0.0169 |
| a06 | paper | worst_baseline | 0.6952 | 0.6089 | 0.6137 | 0.5024 | 2892.6856 | False | -131.0100 | 0.0792 |
| a06 | paper | micro_0.95 | | skipped | | | | | | |
| a06 | paper | macro_0.95 | | skipped | | | | | | |
| a06 | paper | worst_0.90 | | skipped | | | | | | |
| i29 | paper | micro_baseline | 0.7275 | 0.7298 | 0.7281 | 0.6565 | 5614.6297 | True | 0.0000 | 0.0000 |
| i29 | paper | macro_baseline | 0.7264 | 0.7302 | 0.7284 | 0.6561 | 5624.0785 | True | 9.4488 | -0.0004 |
| i29 | paper | worst_baseline | 0.6180 | 0.6681 | 0.6668 | 0.6165 | 4931.7452 | False | -682.8845 | -0.0400 |
| i29 | paper | micro_0.95 | | skipped | | | | | | |
| i29 | paper | macro_0.95 | | skipped | | | | | | |
| i29 | paper | worst_0.90 | | skipped | | | | | | |
| h24 | trained | micro_baseline | 0.9266 | 0.9335 | 0.9310 | 0.8474 | 8.8710 | True | 0.0000 | 0.0000 |
| h24 | trained | macro_baseline | 0.9241 | 0.9323 | 0.9299 | 0.8556 | 8.8857 | True | 0.0148 | 0.0082 |
| h24 | trained | worst_baseline | 0.8768 | 0.9406 | 0.9382 | 0.8692 | 10.9895 | False | 2.1186 | 0.0218 |
| h24 | trained | micro_0.95 | | skipped | | | | | | |
| h24 | trained | macro_0.95 | | skipped | | | | | | |
| h24 | trained | worst_0.90 | | skipped | | | | | | |
| h08 | trained | micro_baseline | 0.7249 | 0.7616 | 0.7606 | 0.6828 | 15.9439 | True | 0.0000 | 0.0000 |
| h08 | trained | macro_baseline | 0.7232 | 0.7617 | 0.7607 | 0.6848 | 15.5224 | True | -0.4215 | 0.0021 |
| h08 | trained | worst_baseline | 0.6502 | 0.7301 | 0.7291 | 0.6869 | 8.7164 | True | -7.2275 | 0.0041 |
| h08 | trained | micro_0.95 | | skipped | | | | | | |
| h08 | trained | macro_0.95 | | skipped | | | | | | |
| h08 | trained | worst_0.90 | | skipped | | | | | | |
| s31 | trained | micro_baseline | 0.4828 | 0.4968 | 0.4970 | 0.4311 | 23.2839 | True | 0.0000 | 0.0000 |
| s31 | trained | macro_baseline | 0.4833 | 0.4956 | 0.4959 | 0.4319 | 23.0630 | True | -0.2208 | 0.0008 |
| s31 | trained | worst_baseline | 0.4230 | 0.4579 | 0.4577 | 0.4206 | 15.6444 | False | -7.6394 | -0.0105 |
| s31 | trained | micro_0.95 | | skipped | | | | | | |
| s31 | trained | macro_0.95 | | skipped | | | | | | |
| s31 | trained | worst_0.90 | | skipped | | | | | | |
| a06 | trained | micro_baseline | 0.3435 | 0.3425 | 0.3409 | 0.0740 | 9.4578 | False | 0.0000 | 0.0000 |
| a06 | trained | macro_baseline | 0.3423 | 0.3457 | 0.3441 | 0.0724 | 9.9124 | True | 0.4547 | -0.0016 |
| a06 | trained | worst_baseline | 0.0746 | 0.3181 | 0.3165 | 0.0967 | 4.6403 | True | -4.8174 | 0.0228 |
| a06 | trained | micro_0.95 | | skipped | | | | | | |
| a06 | trained | macro_0.95 | | skipped | | | | | | |
| a06 | trained | worst_0.90 | | skipped | | | | | | |
| i29 | trained | micro_baseline | 0.3049 | 0.2532 | 0.2523 | 0.1632 | 15.4674 | False | 0.0000 | 0.0000 |
| i29 | trained | macro_baseline | 0.3037 | 0.2542 | 0.2534 | 0.1672 | 14.5034 | False | -0.9641 | 0.0040 |
| i29 | trained | worst_baseline | 0.2343 | 0.2497 | 0.2487 | 0.1573 | 14.0672 | False | -1.4002 | -0.0059 |
| i29 | trained | micro_0.95 | | skipped | | | | | | |
| i29 | trained | macro_0.95 | | skipped | | | | | | |
| i29 | trained | worst_0.90 | | skipped | | | | | | |

## Verdict

Among 10 settings with all three `*_baseline` variants:
- **macro_baseline** costlier than micro_baseline in **6**
- **worst_baseline** costlier than micro_baseline in **3**

Δcost > 0 vs micro_baseline means the stricter fairness floor bought equity at a runtime price. Infeasible / skipped fixed floors are reported explicitly (negative result is fine).

