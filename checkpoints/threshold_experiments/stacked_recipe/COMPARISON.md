# Stacked Recipe — Comparison

Question: does combining **order + accuracy budget (+ optional lower Kdet)** beat any single lever alone, without a major accuracy drop?

Winner column: cheapest recipe with Δacc vs `baseline_protect` ≥ −3pp.

| scene | detector | recipe | holdout acc | cost (ms) | Δacc vs base | Δcost vs base | speedup vs base | feas | winner? |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| a06 | paper | baseline_protect | 0.6750 | 3923.8950 | 0.0000 | 0.0000 | 1.0000 | False | YES |
| a06 | paper | budget_only_2pp | 0.6060 | 2965.1090 | -0.0689 | -958.7860 | 1.3234 | False |  |
| a06 | paper | budget_only_3pp | 0.6074 | 2964.5627 | -0.0676 | -959.3323 | 1.3236 | False |  |
| a06 | paper | order_only | 0.6750 | 3923.8950 | 0.0000 | 0.0000 | 1.0000 | False |  |
| a06 | paper | stacked_full_kdet1000_order_budget_2pp | 0.5999 | 303.4105 | -0.0750 | -3620.4845 | 12.9326 | False |  |
| a06 | paper | stacked_kdet1000_budget_2pp | 0.6133 | 319.8270 | -0.0616 | -3604.0681 | 12.2688 | False |  |
| a06 | paper | stacked_order_budget_2pp | 0.6060 | 2965.1090 | -0.0689 | -958.7860 | 1.3234 | False |  |
| a06 | paper | stacked_order_budget_3pp | 0.6074 | 2964.5627 | -0.0676 | -959.3323 | 1.3236 | False |  |
| a06 | trained | baseline_protect | 0.3425 | 9.4578 | 0.0000 | 0.0000 | 1.0000 | False |  |
| a06 | trained | budget_only_2pp | 0.3279 | 6.6670 | -0.0145 | -2.7908 | 1.4186 | True |  |
| a06 | trained | budget_only_3pp | 0.3178 | 4.7131 | -0.0247 | -4.7446 | 2.0067 | True | YES |
| a06 | trained | order_only | 0.3425 | 9.4578 | 0.0000 | 0.0000 | 1.0000 | False |  |
| a06 | trained | stacked_full_kdet1000_order_budget_2pp | | skipped | | | | | |
| a06 | trained | stacked_kdet1000_budget_2pp | | skipped | | | | | |
| a06 | trained | stacked_order_budget_2pp | 0.3279 | 6.6670 | -0.0145 | -2.7908 | 1.4186 | True |  |
| a06 | trained | stacked_order_budget_3pp | 0.3178 | 4.7131 | -0.0247 | -4.7446 | 2.0067 | True |  |
| h08 | paper | baseline_protect | 0.9154 | 2339.0189 | 0.0000 | 0.0000 | 1.0000 | True |  |
| h08 | paper | budget_only_2pp | 0.8996 | 2018.4785 | -0.0158 | -320.5403 | 1.1588 | True |  |
| h08 | paper | budget_only_3pp | 0.8918 | 1840.5536 | -0.0236 | -498.4653 | 1.2708 | True |  |
| h08 | paper | order_only | 0.9120 | 2359.0405 | -0.0034 | 20.0216 | 0.9915 | True |  |
| h08 | paper | stacked_full_kdet1000_order_budget_2pp | 0.8995 | 218.4732 | -0.0159 | -2120.5457 | 10.7062 | True |  |
| h08 | paper | stacked_kdet1000_budget_2pp | 0.8996 | 218.3242 | -0.0158 | -2120.6947 | 10.7135 | True | YES |
| h08 | paper | stacked_order_budget_2pp | 0.8948 | 2057.9082 | -0.0206 | -281.1107 | 1.1366 | True |  |
| h08 | paper | stacked_order_budget_3pp | 0.8887 | 1802.4928 | -0.0266 | -536.5261 | 1.2977 | True |  |
| h08 | trained | baseline_protect | 0.7619 | 15.8398 | 0.0000 | 0.0000 | 1.0000 | True |  |
| h08 | trained | budget_only_2pp | 0.7364 | 9.7439 | -0.0255 | -6.0959 | 1.6256 | True | YES |
| h08 | trained | budget_only_3pp | 0.7294 | 8.3485 | -0.0325 | -7.4913 | 1.8973 | True |  |
| h08 | trained | order_only | 0.7619 | 15.8398 | 0.0000 | 0.0000 | 1.0000 | True |  |
| h08 | trained | stacked_full_kdet1000_order_budget_2pp | | skipped | | | | | |
| h08 | trained | stacked_kdet1000_budget_2pp | | skipped | | | | | |
| h08 | trained | stacked_order_budget_2pp | 0.7364 | 9.7439 | -0.0255 | -6.0959 | 1.6256 | True |  |
| h08 | trained | stacked_order_budget_3pp | 0.7294 | 8.3485 | -0.0325 | -7.4913 | 1.8973 | True |  |
| h24 | paper | baseline_protect | 0.9665 | 718.4370 | 0.0000 | 0.0000 | 1.0000 | True |  |
| h24 | paper | budget_only_2pp | 0.9432 | 201.3892 | -0.0232 | -517.0478 | 3.5674 | False |  |
| h24 | paper | budget_only_3pp | 0.9361 | 79.3740 | -0.0303 | -639.0630 | 9.0513 | True |  |
| h24 | paper | order_only | 0.9677 | 705.8831 | 0.0013 | -12.5539 | 1.0178 | True |  |
| h24 | paper | stacked_full_kdet1000_order_budget_2pp | 0.9458 | 40.5055 | -0.0206 | -677.9315 | 17.7368 | True |  |
| h24 | paper | stacked_kdet1000_budget_2pp | 0.9458 | 37.1087 | -0.0206 | -681.3283 | 19.3603 | True | YES |
| h24 | paper | stacked_order_budget_2pp | 0.9426 | 226.7764 | -0.0239 | -491.6606 | 3.1680 | True |  |
| h24 | paper | stacked_order_budget_3pp | 0.9368 | 193.6919 | -0.0297 | -524.7451 | 3.7092 | True |  |
| h24 | trained | baseline_protect | 0.9323 | 9.4511 | 0.0000 | 0.0000 | 1.0000 | True |  |
| h24 | trained | budget_only_2pp | 0.9148 | 6.6497 | -0.0174 | -2.8014 | 1.4213 | True |  |
| h24 | trained | budget_only_3pp | 0.9065 | 6.1345 | -0.0258 | -3.3166 | 1.5406 | True | YES |
| h24 | trained | order_only | 0.9323 | 9.4511 | 0.0000 | 0.0000 | 1.0000 | True |  |
| h24 | trained | stacked_full_kdet1000_order_budget_2pp | | skipped | | | | | |
| h24 | trained | stacked_kdet1000_budget_2pp | | skipped | | | | | |
| h24 | trained | stacked_order_budget_2pp | 0.9148 | 6.6497 | -0.0174 | -2.8014 | 1.4213 | True |  |
| h24 | trained | stacked_order_budget_3pp | 0.9065 | 6.1345 | -0.0258 | -3.3166 | 1.5406 | True |  |
| i29 | paper | baseline_protect | 0.7465 | 5813.2885 | 0.0000 | 0.0000 | 1.0000 | True |  |
| i29 | paper | budget_only_2pp | 0.7061 | 5309.6412 | -0.0403 | -503.6473 | 1.0949 | False |  |
| i29 | paper | budget_only_3pp | 0.7055 | 5337.4678 | -0.0410 | -475.8207 | 1.0891 | True |  |
| i29 | paper | order_only | 0.7454 | 5838.1561 | -0.0010 | 24.8676 | 0.9957 | True |  |
| i29 | paper | stacked_full_kdet1000_order_budget_2pp | 0.7190 | 559.0196 | -0.0275 | -5254.2688 | 10.3991 | True | YES |
| i29 | paper | stacked_kdet1000_budget_2pp | 0.7100 | 552.7444 | -0.0364 | -5260.5441 | 10.5171 | False |  |
| i29 | paper | stacked_order_budget_2pp | 0.7136 | 5375.6902 | -0.0329 | -437.5983 | 1.0814 | True |  |
| i29 | paper | stacked_order_budget_3pp | 0.7084 | 5301.8406 | -0.0381 | -511.4478 | 1.0965 | True |  |
| i29 | trained | baseline_protect | 0.2532 | 15.4674 | 0.0000 | 0.0000 | 1.0000 | False |  |
| i29 | trained | budget_only_2pp | 0.2526 | 3.4285 | -0.0006 | -12.0390 | 4.5115 | False | YES |
| i29 | trained | budget_only_3pp | 0.2526 | 3.4285 | -0.0006 | -12.0390 | 4.5115 | False |  |
| i29 | trained | order_only | 0.2532 | 15.4674 | 0.0000 | 0.0000 | 1.0000 | False |  |
| i29 | trained | stacked_full_kdet1000_order_budget_2pp | | skipped | | | | | |
| i29 | trained | stacked_kdet1000_budget_2pp | | skipped | | | | | |
| i29 | trained | stacked_order_budget_2pp | 0.2526 | 3.4285 | -0.0006 | -12.0390 | 4.5115 | False |  |
| i29 | trained | stacked_order_budget_3pp | 0.2526 | 3.4285 | -0.0006 | -12.0390 | 4.5115 | False |  |
| s31 | paper | baseline_protect | 0.7812 | 4043.2902 | 0.0000 | 0.0000 | 1.0000 | False |  |
| s31 | paper | budget_only_2pp | 0.7663 | 3759.0214 | -0.0149 | -284.2688 | 1.0756 | True |  |
| s31 | paper | budget_only_3pp | 0.7577 | 3666.0576 | -0.0234 | -377.2326 | 1.1029 | True |  |
| s31 | paper | order_only | 0.7776 | 4079.9473 | -0.0036 | 36.6571 | 0.9910 | False |  |
| s31 | paper | stacked_full_kdet1000_order_budget_2pp | 0.7607 | 389.2913 | -0.0204 | -3653.9989 | 10.3863 | True | YES |
| s31 | paper | stacked_kdet1000_budget_2pp | 0.7657 | 393.6550 | -0.0155 | -3649.6352 | 10.2712 | True |  |
| s31 | paper | stacked_order_budget_2pp | 0.7677 | 3789.5304 | -0.0135 | -253.7598 | 1.0670 | True |  |
| s31 | paper | stacked_order_budget_3pp | 0.7577 | 3678.1635 | -0.0234 | -365.1268 | 1.0993 | True |  |
| s31 | trained | baseline_protect | 0.4976 | 23.2013 | 0.0000 | 0.0000 | 1.0000 | True |  |
| s31 | trained | budget_only_2pp | 0.4804 | 17.8147 | -0.0173 | -5.3866 | 1.3024 | True |  |
| s31 | trained | budget_only_3pp | 0.4696 | 15.3185 | -0.0280 | -7.8827 | 1.5146 | True | YES |
| s31 | trained | order_only | 0.4976 | 23.2013 | 0.0000 | 0.0000 | 1.0000 | True |  |
| s31 | trained | stacked_full_kdet1000_order_budget_2pp | | skipped | | | | | |
| s31 | trained | stacked_kdet1000_budget_2pp | | skipped | | | | | |
| s31 | trained | stacked_order_budget_2pp | 0.4804 | 17.8147 | -0.0173 | -5.3866 | 1.3024 | True |  |
| s31 | trained | stacked_order_budget_3pp | 0.4696 | 15.3185 | -0.0280 | -7.8827 | 1.5146 | True |  |

## Verdict

- **a06/paper** winner: `baseline_protect` (best single `order_only`, best stacked `None`, stacking_helps=None)
- **a06/trained** winner: `budget_only_3pp` (best single `budget_only_3pp`, best stacked `stacked_order_budget_3pp`, stacking_helps=False)
- **h08/paper** winner: `stacked_kdet1000_budget_2pp` (best single `budget_only_3pp`, best stacked `stacked_kdet1000_budget_2pp`, stacking_helps=True)
- **h08/trained** winner: `budget_only_2pp` (best single `budget_only_2pp`, best stacked `stacked_order_budget_2pp`, stacking_helps=False)
- **h24/paper** winner: `stacked_kdet1000_budget_2pp` (best single `budget_only_2pp`, best stacked `stacked_kdet1000_budget_2pp`, stacking_helps=True)
- **h24/trained** winner: `budget_only_3pp` (best single `budget_only_3pp`, best stacked `stacked_order_budget_3pp`, stacking_helps=False)
- **i29/paper** winner: `stacked_full_kdet1000_order_budget_2pp` (best single `order_only`, best stacked `stacked_full_kdet1000_order_budget_2pp`, stacking_helps=True)
- **i29/trained** winner: `budget_only_2pp` (best single `budget_only_2pp`, best stacked `stacked_order_budget_2pp`, stacking_helps=False)
- **s31/paper** winner: `stacked_full_kdet1000_order_budget_2pp` (best single `budget_only_3pp`, best stacked `stacked_full_kdet1000_order_budget_2pp`, stacking_helps=True)
- **s31/trained** winner: `budget_only_3pp` (best single `budget_only_3pp`, best stacked `stacked_order_budget_3pp`, stacking_helps=False)

Stacked recipe won (within −3pp acc) in **4/10** (scene, detector) settings.
Stacking beat the best single lever (same −3pp gate) in **4/9** comparable settings. If that fraction is low, stacking did **not** help beyond the best individual lever on that split — a valid negative result.

