# Alternating Structure ↔ Thresholds — Comparison

Question: after thresholds change, does re-synthesizing the cascade (alternating) beat one-shot (DP once → thresholds once)?

All methods for a given (scene, detector_mode) share the **same** `blocked_per_run` holdout split. Negative result is reported if alternating does not improve.

| scene | detector | method | holdout acc | cost (ms) | speedup | feasible | Δacc vs one-shot | Δcost vs one-shot (ms) | layout |
|---|---|---|---:|---:|---:|---|---:|---:|---|
| a06 | paper | alternating_n2 | 0.5919 | 2792.2270 | 1.3583 | False | -0.0831 | -1131.6680 | `K2→K3→K0→detector` |
| a06 | paper | alternating_n3 | 0.5940 | 2809.1773 | 1.3999 | False | -0.0810 | -1114.7177 | `K0→K3→detector` |
| a06 | paper | one_shot | 0.6750 | 3923.8950 | 0.9616 | False |  |  | `K0→K3→K2→K1→detector` |
| a06 | trained | alternating_n2 | 0.3425 | 9.4578 | 2.8874 | False | 0.0000 | 0.0000 | `K0→detector` |
| a06 | trained | alternating_n3 | 0.3425 | 9.4578 | 2.8874 | False | 0.0000 | 0.0000 | `K0→detector` |
| a06 | trained | one_shot | 0.3425 | 9.4578 | 2.8866 | False |  |  | `K0→detector` |
| h08 | paper | alternating_n2 | 0.9154 | 2337.0319 | 1.3601 | True | 0.0000 | -1.9870 | `K3→K1→K2→detector` |
| h08 | paper | alternating_n3 | 0.9150 | 2333.6773 | 1.3621 | True | -0.0003 | -5.3416 | `K3→K1→K2→detector` |
| h08 | paper | one_shot | 0.9154 | 2339.0189 | 1.2567 | True |  |  | `K0→K3→K2→K1→detector` |
| h08 | trained | alternating_n2 | 0.7624 | 11.4029 | 2.0401 | True | 0.0005 | -4.4369 | `K0→K3→detector` |
| h08 | trained | alternating_n3 | 0.7624 | 11.4029 | 2.0401 | True | 0.0005 | -4.4369 | `K0→K3→detector` |
| h08 | trained | one_shot | 0.7619 | 15.8398 | 1.3012 | True |  |  | `K0→detector` |
| h24 | paper | alternating_n2 | 0.9671 | 710.7046 | 1.9087 | True | 0.0006 | -7.7324 | `K3→K0→K1→K2→detector` |
| h24 | paper | alternating_n3 | 0.9671 | 710.6819 | 1.9087 | True | 0.0006 | -7.7551 | `K3→K0→K2→K1→detector` |
| h24 | paper | one_shot | 0.9665 | 718.4370 | 1.7660 | True |  |  | `K0→K3→K2→K1→detector` |
| h24 | trained | alternating_n2 | 0.9335 | 8.8710 | 1.4567 | True | 0.0013 | -0.5802 | `K0→K3→detector` |
| h24 | trained | alternating_n3 | 0.9335 | 8.8710 | 1.4567 | True | 0.0013 | -0.5802 | `K0→K3→detector` |
| h24 | trained | one_shot | 0.9323 | 9.4511 | 1.3672 | True |  |  | `K0→K3→detector` |
| i29 | paper | alternating_n2 | 0.7303 | 5615.5011 | 1.1495 | True | -0.0161 | -197.7874 | `K0→K3→K2→K1→detector` |
| i29 | paper | alternating_n3 | 0.7298 | 5605.9583 | 1.3701 | True | -0.0166 | -207.3302 | `K3→K1→detector` |
| i29 | paper | one_shot | 0.7465 | 5813.2885 | 1.0272 | True |  |  | `K0→K2→K3→K1→detector` |
| i29 | trained | alternating_n2 | 0.2610 | 9.5678 | 2.7201 | False | 0.0078 | -5.8996 | `K0→detector` |
| i29 | trained | alternating_n3 | 0.2610 | 9.5678 | 2.7201 | False | 0.0078 | -5.8996 | `K0→detector` |
| i29 | trained | one_shot | 0.2532 | 15.4674 | 1.6263 | False |  |  | `K0→detector` |
| s31 | paper | alternating_n2 | 0.7831 | 4025.0098 | 1.2431 | False | 0.0020 | -18.2804 | `K3→K2→K1→detector` |
| s31 | paper | alternating_n3 | 0.7821 | 4029.7713 | 1.2487 | False | 0.0010 | -13.5190 | `K3→K1→K2→detector` |
| s31 | paper | one_shot | 0.7812 | 4043.2902 | 1.1143 | False |  |  | `K0→K3→K2→K1→detector` |
| s31 | trained | alternating_n2 | 0.4946 | 21.2818 | 1.1473 | True | -0.0030 | -1.9195 | `K0→detector` |
| s31 | trained | alternating_n3 | 0.4827 | 10.6957 | 2.4979 | False | -0.0149 | -12.5055 | `K0→K3→detector` |
| s31 | trained | one_shot | 0.4976 | 23.2013 | 1.0527 | True |  |  | `K0→detector` |

## Verdict

Among 20 alternating runs (N=2 and N=3, both detector modes / scenes):
- **18** lowered holdout expected cost vs one-shot
- **10** raised holdout accuracy vs one-shot

Δcost < 0 means alternating is cheaper; Δacc > 0 means more accurate.

