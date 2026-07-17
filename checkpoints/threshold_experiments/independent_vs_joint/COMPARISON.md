# Independent vs Joint Thresholds — Comparison

Question: does **joint** end-to-end annealing beat calibrating each Ki **independently** (precision / P(IDK)), on the same DP layout + holdout?

| scene | detector | method | holdout acc | cost (ms) | speedup vs collect | feasible | Δcost vs joint | Δacc vs joint |
|---|---|---|---:|---:|---:|---|---:|---:|
| a06 | paper | collection | 0.6464 | 3773.3912 | 1.0000 | False | -1084.2609 | -0.0965 |
| a06 | paper | indep_precision | 0.9625 | 8613.3491 | 0.4381 | True | 3755.6970 | 0.2196 |
| a06 | paper | indep_precision_match | 0.6453 | 3757.9672 | 1.0041 | False | -1099.6849 | -0.0976 |
| a06 | paper | indep_p_idk_match | 0.6464 | 3773.3912 | 1.0000 | False | -1084.2609 | -0.0965 |
| a06 | paper | indep_p_idk_fixed | 0.3409 | 9.7003 | 388.9982 | False | -4847.9518 | -0.4020 |
| a06 | paper | joint_anneal | 0.7429 | 4857.6521 | 0.7768 | False | 0.0000 | 0.0000 |
| a06 | trained | collection | 0.3292 | 27.3012 | 1.0000 | False | 17.8435 | -0.0132 |
| a06 | trained | indep_precision | 0.3354 | 30.2478 | 0.9026 | False | 20.7900 | -0.0071 |
| a06 | trained | indep_precision_match | 0.3287 | 27.1297 | 1.0063 | False | 17.6719 | -0.0138 |
| a06 | trained | indep_p_idk_match | 0.3292 | 27.3012 | 1.0000 | False | 17.8435 | -0.0132 |
| a06 | trained | indep_p_idk_fixed | 0.3193 | 8.8357 | 3.0899 | False | -0.6221 | -0.0232 |
| a06 | trained | joint_anneal | 0.3425 | 9.4578 | 2.8866 | False | 0.0000 | 0.0000 |
| h08 | paper | collection | 0.9157 | 2939.5043 | 1.0000 | True | 587.7443 | 0.0020 |
| h08 | paper | indep_precision | 0.9149 | 2594.3902 | 1.1330 | True | 242.6302 | 0.0012 |
| h08 | paper | indep_precision_match | 0.9157 | 2936.1373 | 1.0011 | True | 584.3773 | 0.0020 |
| h08 | paper | indep_p_idk_match | 0.9157 | 2939.5043 | 1.0000 | True | 587.7443 | 0.0020 |
| h08 | paper | indep_p_idk_fixed | 0.7007 | 46.2702 | 63.5292 | False | -2305.4899 | -0.2130 |
| h08 | paper | joint_anneal | 0.9137 | 2351.7600 | 1.2499 | True | 0.0000 | 0.0000 |
| h08 | trained | collection | 0.7743 | 20.6108 | 1.0000 | True | 4.6293 | 0.0117 |
| h08 | trained | indep_precision | 0.7782 | 24.3592 | 0.8461 | True | 8.3777 | 0.0156 |
| h08 | trained | indep_precision_match | 0.7743 | 20.5845 | 1.0013 | True | 4.6030 | 0.0117 |
| h08 | trained | indep_p_idk_match | 0.7743 | 20.6108 | 1.0000 | True | 4.6293 | 0.0117 |
| h08 | trained | indep_p_idk_fixed | 0.7007 | 7.5634 | 2.7251 | False | -8.4181 | -0.0618 |
| h08 | trained | joint_anneal | 0.7626 | 15.9815 | 1.2897 | True | 0.0000 | 0.0000 |
| h24 | paper | collection | 0.9690 | 1268.7928 | 1.0000 | True | 571.1887 | 0.0019 |
| h24 | paper | indep_precision | 0.8910 | 6.3098 | 201.0818 | False | -691.2942 | -0.0761 |
| h24 | paper | indep_precision_match | 0.9690 | 1262.3436 | 1.0051 | True | 564.7395 | 0.0019 |
| h24 | paper | indep_p_idk_match | 0.9690 | 1268.7928 | 1.0000 | True | 571.1887 | 0.0019 |
| h24 | paper | indep_p_idk_fixed | 0.9135 | 180.7187 | 7.0208 | False | -516.8854 | -0.0535 |
| h24 | paper | joint_anneal | 0.9671 | 697.6041 | 1.8188 | True | 0.0000 | 0.0000 |
| h24 | trained | collection | 0.9445 | 12.9219 | 1.0000 | True | 4.0934 | 0.0123 |
| h24 | trained | indep_precision | 0.8910 | 6.3098 | 2.0479 | False | -2.5187 | -0.0413 |
| h24 | trained | indep_precision_match | 0.9445 | 12.9219 | 1.0000 | True | 4.0934 | 0.0123 |
| h24 | trained | indep_p_idk_match | 0.9445 | 12.9219 | 1.0000 | True | 4.0934 | 0.0123 |
| h24 | trained | indep_p_idk_fixed | 0.9135 | 7.2812 | 1.7747 | False | -1.5473 | -0.0187 |
| h24 | trained | joint_anneal | 0.9323 | 8.8285 | 1.4637 | True | 0.0000 | 0.0000 |
| i29 | paper | collection | 0.7398 | 5971.2641 | 1.0000 | True | 232.3403 | 0.0008 |
| i29 | paper | indep_precision | 0.9967 | 9703.1029 | 0.6154 | True | 3964.1791 | 0.2576 |
| i29 | paper | indep_precision_match | 0.7357 | 5903.0393 | 1.0116 | True | 164.1155 | -0.0034 |
| i29 | paper | indep_p_idk_match | 0.7398 | 5971.2641 | 1.0000 | True | 232.3403 | 0.0008 |
| i29 | paper | indep_p_idk_fixed | 0.2512 | 35.3799 | 168.7754 | False | -5703.5439 | -0.4879 |
| i29 | paper | joint_anneal | 0.7391 | 5738.9238 | 1.0405 | True | 0.0000 | 0.0000 |
| i29 | trained | collection | 0.2396 | 25.1547 | 1.0000 | False | 9.6873 | -0.0136 |
| i29 | trained | indep_precision | 0.2439 | 30.2755 | 0.8309 | False | 14.8080 | -0.0093 |
| i29 | trained | indep_precision_match | 0.2386 | 24.7834 | 1.0150 | False | 9.3159 | -0.0145 |
| i29 | trained | indep_p_idk_match | 0.2396 | 25.1547 | 1.0000 | False | 9.6873 | -0.0136 |
| i29 | trained | indep_p_idk_fixed | 0.2486 | 8.1975 | 3.0686 | False | -7.2699 | -0.0045 |
| i29 | trained | joint_anneal | 0.2532 | 15.4674 | 1.6263 | False | 0.0000 | 0.0000 |
| s31 | paper | collection | 0.7867 | 4505.3149 | 1.0000 | True | 504.0180 | 0.0050 |
| s31 | paper | indep_precision | 0.9760 | 7963.4899 | 0.5657 | True | 3962.1930 | 0.1942 |
| s31 | paper | indep_precision_match | 0.7867 | 4505.3140 | 1.0000 | True | 504.0172 | 0.0050 |
| s31 | paper | indep_p_idk_match | 0.7867 | 4505.3149 | 1.0000 | True | 504.0180 | 0.0050 |
| s31 | paper | indep_p_idk_fixed | 0.4373 | 49.3313 | 91.3277 | False | -3951.9655 | -0.3444 |
| s31 | paper | joint_anneal | 0.7817 | 4001.2969 | 1.1260 | False | 0.0000 | 0.0000 |
| s31 | trained | collection | 0.4948 | 24.4237 | 1.0000 | True | 1.4030 | -0.0002 |
| s31 | trained | indep_precision | 0.5103 | 30.1714 | 0.8095 | True | 7.1506 | 0.0153 |
| s31 | trained | indep_precision_match | 0.4948 | 24.4228 | 1.0000 | True | 1.4021 | -0.0002 |
| s31 | trained | indep_p_idk_match | 0.4948 | 24.4237 | 1.0000 | True | 1.4030 | -0.0002 |
| s31 | trained | indep_p_idk_fixed | 0.4304 | 7.2155 | 3.3849 | False | -15.8052 | -0.0647 |
| s31 | trained | joint_anneal | 0.4950 | 23.0207 | 1.0609 | True | 0.0000 | 0.0000 |

## Verdict

Across 10 (scene, detector) settings (feasible policies only):
- Joint is cheapest feasible (or sole feasible) in **6** settings
- Some *feasible* independent method beats joint on cost in **0** settings

Δcost vs joint < 0 means that method is cheaper than joint anneal. Ignore infeasible cheap runs (they miss the accuracy target). Independent methods never see cascade cost while choosing thresholds.

