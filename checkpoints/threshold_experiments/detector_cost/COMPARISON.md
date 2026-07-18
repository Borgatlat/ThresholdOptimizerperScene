# Detector-Cost Sensitivity — Comparison

Question: as assumed paper-Kdet cost changes, how do **DP layout**, holdout accuracy, and expected cost change after threshold retuning?

All paper costs for a scene share the same `blocked_per_run` holdout. `trained_ref` uses the measured Kdet cost (structure + replay).

| scene | mode | Kdet cost (ms) | chain len | layout | holdout acc | opt cost (ms) | speedup | feasible |
|---|---|---:|---:|---|---:|---:|---:|---|
| h24 | paper | 100.0000 | 3 | `K0→K3→K2→detector` | 0.9665 | 18.2770 | 1.2372 | True |
| h24 | paper | 250.0000 | 3 | `K0→K3→K2→detector` | 0.9665 | 29.1157 | 1.4314 | True |
| h24 | paper | 500.0000 | 3 | `K0→K3→K2→detector` | 0.9671 | 46.3618 | 1.5843 | True |
| h24 | paper | 1000.0000 | 3 | `K0→K3→K2→detector` | 0.9671 | 81.2005 | 1.6872 | True |
| h24 | paper | 2500.0000 | 4 | `K0→K3→K2→K1→detector` | 0.9671 | 183.5726 | 1.7717 | True |
| h24 | paper | 5000.0000 | 4 | `K0→K3→K2→K1→detector` | 0.9671 | 354.5403 | 1.8045 | True |
| h24 | paper | 10000.0000 | 4 | `K0→K3→K2→K1→detector` | 0.9671 | 696.4758 | 1.8217 | True |
| h24 | trained | 28.2808 | 2 | `K0→K3→detector` | 0.9342 | 9.3718 | 1.3788 | True |
| h08 | paper | 100.0000 | 3 | `K0→K3→K2→detector` | 0.9135 | 37.8471 | 1.1309 | True |
| h08 | paper | 250.0000 | 3 | `K0→K3→K2→detector` | 0.9155 | 74.0788 | 1.1705 | True |
| h08 | paper | 500.0000 | 3 | `K0→K3→K2→detector` | 0.9159 | 133.5139 | 1.1976 | True |
| h08 | paper | 1000.0000 | 3 | `K0→K3→K2→detector` | 0.9161 | 251.2002 | 1.2192 | True |
| h08 | paper | 2500.0000 | 4 | `K0→K3→K2→K1→detector` | 0.9147 | 597.6176 | 1.2492 | True |
| h08 | paper | 5000.0000 | 4 | `K0→K3→K2→K1→detector` | 0.9152 | 1177.8343 | 1.2545 | True |
| h08 | paper | 10000.0000 | 4 | `K0→K3→K2→K1→detector` | 0.9154 | 2339.0167 | 1.2567 | True |
| h08 | trained | 28.2808 | 1 | `K0→detector` | 0.7619 | 15.8398 | 1.3012 | True |
| s31 | paper | 100.0000 | 3 | `K0→K3→K2→detector` | 0.7833 | 57.5376 | 1.0574 | False |
| s31 | paper | 250.0000 | 3 | `K0→K3→K2→detector` | 0.7833 | 118.3216 | 1.0841 | False |
| s31 | paper | 500.0000 | 3 | `K0→K3→K2→detector` | 0.7843 | 219.2658 | 1.0976 | False |
| s31 | paper | 1000.0000 | 3 | `K0→K3→K2→detector` | 0.7861 | 427.9753 | 1.0876 | True |
| s31 | paper | 2500.0000 | 4 | `K0→K3→K2→K1→detector` | 0.7835 | 1022.9390 | 1.1152 | False |
| s31 | paper | 5000.0000 | 4 | `K0→K3→K2→K1→detector` | 0.7833 | 2023.9349 | 1.1178 | False |
| s31 | paper | 10000.0000 | 4 | `K0→K3→K2→K1→detector` | 0.7804 | 4014.9138 | 1.1221 | False |
| s31 | trained | 28.2808 | 1 | `K0→detector` | 0.4956 | 22.9008 | 1.0665 | True |
| a06 | paper | 100.0000 | 3 | `K0→K3→K2→detector` | 0.6097 | 45.0748 | 1.2372 | False |
| a06 | paper | 250.0000 | 3 | `K0→K3→K2→detector` | 0.6277 | 96.3443 | 1.1660 | False |
| a06 | paper | 500.0000 | 4 | `K0→K3→K2→K1→detector` | 0.6080 | 166.6530 | 1.2436 | False |
| a06 | paper | 1000.0000 | 4 | `K0→K3→K2→K1→detector` | 0.6685 | 394.8855 | 1.0002 | False |
| a06 | paper | 2500.0000 | 4 | `K0→K3→K2→K1→detector` | 0.6786 | 999.0758 | 0.9589 | False |
| a06 | paper | 5000.0000 | 4 | `K0→K3→K2→K1→detector` | 0.6746 | 1929.7708 | 0.9827 | False |
| a06 | paper | 10000.0000 | 4 | `K0→K3→K2→K1→detector` | 0.6642 | 3713.2092 | 1.0162 | False |
| a06 | trained | 28.2808 | 1 | `K0→detector` | 0.3425 | 9.4578 | 2.8866 | False |
| i29 | paper | 100.0000 | 3 | `K0→K2→K3→detector` | 0.7431 | 74.4614 | 1.0252 | True |
| i29 | paper | 250.0000 | 3 | `K0→K2→K3→detector` | 0.7404 | 161.1744 | 1.0283 | True |
| i29 | paper | 500.0000 | 3 | `K0→K2→K3→detector` | 0.7495 | 308.7824 | 1.0192 | True |
| i29 | paper | 1000.0000 | 3 | `K0→K2→K3→detector` | 0.7366 | 588.8509 | 1.0405 | True |
| i29 | paper | 2500.0000 | 4 | `K0→K2→K3→K1→detector` | 0.7476 | 1478.4174 | 1.0206 | True |
| i29 | paper | 5000.0000 | 4 | `K0→K2→K3→K1→detector` | 0.7351 | 2857.2407 | 1.0487 | True |
| i29 | paper | 10000.0000 | 4 | `K0→K2→K3→K1→detector` | 0.7323 | 5638.7813 | 1.0590 | True |
| i29 | trained | 28.2808 | 1 | `K0→detector` | 0.2402 | 24.8646 | 1.0117 | False |

## Verdict

On **h24/paper**, chain length (excl. detector) across the sweep: [3, 3, 3, 3, 4, 4, 4] for Kdet costs [100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0].
- Chain length is **non-decreasing** in Kdet cost.
Higher assumed Kdet cost usually deepens the cascade and raises end-to-end expected cost even after threshold retune — unless annealing finds aggressive early accepts.

