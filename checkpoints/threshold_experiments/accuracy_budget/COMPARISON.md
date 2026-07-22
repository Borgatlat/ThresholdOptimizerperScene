# Accuracy-Budget Pareto — Comparison

Question: how much **speed** do we buy by allowing a small, controlled drop in the micro-accuracy floor below the collection baseline?

Same DP layout + same `blocked_per_run` holdout within each (scene, detector_mode). `budget_0pp` = protect baseline (current default).

| scene | detector | budget | floor | holdout acc | cost (ms) | speedup vs collect | feas | Δcost vs 0pp | Δacc vs 0pp | speedup vs 0pp |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| h24 | paper | budget_0pp | 0.9655 | 0.9671 | 696.4758 | 1.8217 | True | 0.0000 | 0.0000 | 1.0000 |
| h24 | paper | budget_0p5pp | 0.9605 | 0.9594 | 531.0338 | 2.3893 | False | -165.4420 | -0.0077 | 1.3115 |
| h24 | paper | budget_1pp | 0.9555 | 0.9542 | 460.1348 | 2.7574 | False | -236.3411 | -0.0129 | 1.5136 |
| h24 | paper | budget_2pp | 0.9455 | 0.9445 | 260.3104 | 4.8742 | False | -436.1654 | -0.0226 | 2.6756 |
| h24 | paper | budget_3pp | 0.9355 | 0.9368 | 84.8975 | 14.9450 | True | -611.5783 | -0.0303 | 8.2037 |
| h08 | paper | budget_0pp | 0.9071 | 0.9152 | 2340.6986 | 1.2558 | True | 0.0000 | 0.0000 | 1.0000 |
| h08 | paper | budget_0p5pp | 0.9021 | 0.9115 | 2271.9815 | 1.2938 | True | -68.7171 | -0.0037 | 1.0302 |
| h08 | paper | budget_1pp | 0.8971 | 0.9075 | 2169.4985 | 1.3549 | True | -171.2001 | -0.0077 | 1.0789 |
| h08 | paper | budget_2pp | 0.8871 | 0.8998 | 2021.8407 | 1.4539 | True | -318.8579 | -0.0154 | 1.1577 |
| h08 | paper | budget_3pp | 0.8771 | 0.8914 | 1825.5012 | 1.6102 | True | -515.1974 | -0.0238 | 1.2822 |
| s31 | paper | budget_0pp | 0.7844 | 0.7823 | 4021.2626 | 1.1204 | False | 0.0000 | 0.0000 | 1.0000 |
| s31 | paper | budget_0p5pp | 0.7794 | 0.7798 | 4007.4291 | 1.1242 | True | -13.8335 | -0.0026 | 1.0035 |
| s31 | paper | budget_1pp | 0.7744 | 0.7744 | 3915.2291 | 1.1507 | True | -106.0336 | -0.0079 | 1.0271 |
| s31 | paper | budget_2pp | 0.7644 | 0.7661 | 3780.7815 | 1.1916 | True | -240.4811 | -0.0163 | 1.0636 |
| s31 | paper | budget_3pp | 0.7544 | 0.7560 | 3617.7817 | 1.2453 | True | -403.4810 | -0.0264 | 1.1115 |
| a06 | paper | budget_0pp | 0.7666 | 0.6656 | 3751.9113 | 1.0057 | False | 0.0000 | 0.0000 | 1.0000 |
| a06 | paper | budget_0p5pp | 0.7616 | 0.6313 | 3326.3398 | 1.1344 | False | -425.5716 | -0.0343 | 1.1279 |
| a06 | paper | budget_1pp | 0.7566 | 0.6411 | 3405.9720 | 1.1079 | False | -345.9393 | -0.0245 | 1.1016 |
| a06 | paper | budget_2pp | 0.7466 | 0.6639 | 3707.3735 | 1.0178 | False | -44.5378 | -0.0017 | 1.0120 |
| a06 | paper | budget_3pp | 0.7366 | 0.6746 | 3852.2099 | 0.9795 | False | 100.2985 | 0.0090 | 0.9740 |
| i29 | paper | budget_0pp | 0.7275 | 0.7306 | 5618.1408 | 1.0629 | True | 0.0000 | 0.0000 | 1.0000 |
| i29 | paper | budget_0p5pp | 0.7225 | 0.7250 | 5562.6980 | 1.0734 | True | -55.4428 | -0.0056 | 1.0100 |
| i29 | paper | budget_1pp | 0.7175 | 0.7311 | 5676.6861 | 1.0519 | True | 58.5453 | 0.0005 | 0.9897 |
| i29 | paper | budget_2pp | 0.7075 | 0.7207 | 5522.6917 | 1.0812 | True | -95.4490 | -0.0099 | 1.0173 |
| i29 | paper | budget_3pp | 0.6975 | 0.6963 | 5190.7087 | 1.1504 | False | -427.4321 | -0.0343 | 1.0823 |
| h24 | trained | budget_0pp | 0.9266 | 0.9310 | 9.2662 | 1.3945 | True | 0.0000 | 0.0000 | 1.0000 |
| h24 | trained | budget_0p5pp | 0.9216 | 0.9290 | 8.1828 | 1.5791 | True | -1.0834 | -0.0019 | 1.1324 |
| h24 | trained | budget_1pp | 0.9166 | 0.9239 | 7.4889 | 1.7255 | True | -1.7773 | -0.0071 | 1.2373 |
| h24 | trained | budget_2pp | 0.9066 | 0.9161 | 6.7611 | 1.9112 | True | -2.5051 | -0.0148 | 1.3705 |
| h24 | trained | budget_3pp | 0.8966 | 0.9065 | 6.1345 | 2.1064 | True | -3.1317 | -0.0245 | 1.5105 |
| h08 | trained | budget_0pp | 0.7249 | 0.7619 | 15.8398 | 1.3012 | True | 0.0000 | 0.0000 | 1.0000 |
| h08 | trained | budget_0p5pp | 0.7199 | 0.7549 | 13.8720 | 1.4858 | True | -1.9678 | -0.0070 | 1.1419 |
| h08 | trained | budget_1pp | 0.7149 | 0.7482 | 12.0665 | 1.7081 | True | -3.7733 | -0.0137 | 1.3127 |
| h08 | trained | budget_2pp | 0.7049 | 0.7364 | 9.7439 | 2.1153 | True | -6.0959 | -0.0255 | 1.6256 |
| h08 | trained | budget_3pp | 0.6949 | 0.7294 | 8.3485 | 2.4688 | True | -7.4913 | -0.0325 | 1.8973 |
| s31 | trained | budget_0pp | 0.4828 | 0.4962 | 23.1090 | 1.0569 | True | 0.0000 | 0.0000 | 1.0000 |
| s31 | trained | budget_0p5pp | 0.4778 | 0.4954 | 21.7040 | 1.1253 | True | -1.4051 | -0.0008 | 1.0647 |
| s31 | trained | budget_1pp | 0.4728 | 0.4859 | 19.8886 | 1.2280 | True | -3.2204 | -0.0103 | 1.1619 |
| s31 | trained | budget_2pp | 0.4628 | 0.4808 | 18.2239 | 1.3402 | True | -4.8852 | -0.0155 | 1.2681 |
| s31 | trained | budget_3pp | 0.4528 | 0.4639 | 15.3619 | 1.5899 | True | -7.7471 | -0.0323 | 1.5043 |
| a06 | trained | budget_0pp | 0.3435 | 0.3425 | 9.4578 | 2.8866 | False | 0.0000 | 0.0000 | 1.0000 |
| a06 | trained | budget_0p5pp | 0.3385 | 0.3382 | 8.5740 | 3.1842 | False | -0.8837 | -0.0042 | 1.1031 |
| a06 | trained | budget_1pp | 0.3335 | 0.3350 | 8.0937 | 3.3731 | True | -1.3640 | -0.0075 | 1.1685 |
| a06 | trained | budget_2pp | 0.3235 | 0.3279 | 6.6670 | 4.0950 | True | -2.7908 | -0.0145 | 1.4186 |
| a06 | trained | budget_3pp | 0.3135 | 0.3178 | 4.7131 | 5.7926 | True | -4.7446 | -0.0247 | 2.0067 |
| i29 | trained | budget_0pp | 0.3049 | 0.2532 | 15.4674 | 1.6263 | False | 0.0000 | 0.0000 | 1.0000 |
| i29 | trained | budget_0p5pp | 0.2999 | 0.2586 | 7.1351 | 3.5255 | False | -8.3323 | 0.0055 | 2.1678 |
| i29 | trained | budget_1pp | 0.2949 | 0.2568 | 4.8972 | 5.1365 | False | -10.5702 | 0.0037 | 3.1584 |
| i29 | trained | budget_2pp | 0.2849 | 0.2526 | 3.4285 | 7.3370 | False | -12.0390 | -0.0006 | 4.5115 |
| i29 | trained | budget_3pp | 0.2749 | 0.2526 | 3.4285 | 7.3370 | False | -12.0390 | -0.0006 | 4.5115 |

## Verdict

At a **1pp** accuracy budget (paper Kdet), median speedup vs `budget_0pp` across scenes: **1.08×** (n=5).
Negative Δcost vs 0pp means the relaxed floor found a cheaper policy. Check Δacc: a large unexpected accuracy drop means the budget was spent aggressively; a small drop with big speedup is the sweet spot.

