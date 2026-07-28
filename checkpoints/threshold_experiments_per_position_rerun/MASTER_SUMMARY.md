# Threshold Optimizer Experiment Summary

Scene-switching / per-scene classifier training was **not** run.

| suite | run | holdout acc | holdout cost (ms) | speedup vs baseline | feasible |
|---|---|---:|---:|---:|---|
| layouts | dp_optimal | 0.9716 | 832.6563 | 1.5238 | True |
| layouts | global_only | 0.9755 | 969.3845 | 1.4037 | True |
| layouts | single_global | 0.9852 | 1561.0627 | 1.0000 | True |
| layouts | hierarchy_classic | 0.9871 | 4267.9469 | 1.0272 | True |
| layouts | three_linear | 0.9890 | 1917.8680 | 4.1004 | True |
| layouts | three_global | 0.9755 | 969.3845 | 1.4037 | True |
| layouts | k0_k2_k3_hierarchy | 0.9729 | 892.9212 | 1.4926 | True |
| layouts | both_identifiers | 0.9748 | 979.9983 | 2.2564 | True |
| targets | paper_baseline | 0.9716 | 832.6563 | 1.5238 | True |
| targets | paper_acc_0.90 | 0.9077 | 6.6073 | 192.0297 | True |
| targets | paper_acc_0.95 | 0.9516 | 389.0978 | 3.2609 | True |
| targets | paper_acc_0.98 | 0.9845 | 1426.6153 | 0.8894 | True |
| targets | trained_baseline | 0.9323 | 8.8200 | 1.4651 | True |
| targets | trained_acc_0.90 | 0.9110 | 6.2446 | 2.0693 | True |
| targets | trained_acc_0.95 | 0.9400 | 13.5645 | 0.9526 | False |
| targets | trained_acc_0.98 | 0.9400 | 13.5645 | 0.9526 | False |
| scenes_trained | h24 | 0.9323 | 8.8200 | 1.4651 | True |
| scenes_trained | h08 | 0.7619 | 15.8398 | 1.3012 | True |
| scenes_trained | s31 | 0.4929 | 23.1909 | 1.0532 | True |
| scenes_trained | a06 | 0.3425 | 9.4578 | 2.8866 | False |
| scenes_trained | i29 | 0.2532 | 15.4674 | 1.6263 | False |
| transfer_zero_shot | h24 | 0.9667 | 858.9426 |  | None |
| transfer_zero_shot | h08 | 0.9048 | 2731.0344 |  | None |
| transfer_zero_shot | s31 | 0.7772 | 3880.3132 |  | None |
| transfer_zero_shot | a06 | 0.6904 | 3823.3854 |  | None |
| transfer_zero_shot | i29 | 0.7118 | 4940.3333 |  | None |
| transfer_retune | h24 | 0.9716 | 832.6563 | 1.5238 | None |
| transfer_retune | h08 | 0.9159 | 2370.9109 | 1.2398 | None |
| transfer_retune | s31 | 0.7851 | 4116.5109 | 1.0964 | None |
| transfer_retune | a06 | 0.7138 | 4456.4265 | 0.8464 | None |
| transfer_retune | i29 | 0.7378 | 5732.1404 | 1.0418 | None |
| search_settings | q10_blocked | 0.9658 | 846.8404 | 1.4983 | True |
| search_settings | q25_blocked | 0.9748 | 992.9388 | 1.2778 | True |
| search_settings | q50_blocked | 0.9716 | 832.6563 | 1.5238 | True |
| search_settings | q100_blocked | 0.9677 | 711.2906 | 1.7838 | True |
| search_settings | q50_random | 0.9555 | 870.7184 | 1.5985 | False |
| layouts_by_scene | h24__dp_optimal | 0.9716 | 832.6563 | 1.5238 | True |
| layouts_by_scene | h24__global_only | 0.9755 | 969.3845 | 1.4037 | True |
| layouts_by_scene | h24__hierarchy_classic | 0.9871 | 4267.9469 | 1.0272 | True |
| layouts_by_scene | h24__three_linear | 0.9890 | 1917.8680 | 4.1004 | True |
| layouts_by_scene | h08__dp_optimal | 0.9142 | 2438.2189 | 1.2056 | True |
| layouts_by_scene | h08__global_only | 0.9254 | 2699.4621 | 1.1809 | True |
| layouts_by_scene | h08__hierarchy_classic | 0.9657 | 6542.7393 | 0.9916 | True |
| layouts_by_scene | h08__three_linear | 0.9745 | 4201.9746 | 1.8526 | True |
| layouts_by_scene | s31__dp_optimal | 0.7881 | 4316.6270 | 1.0437 | True |
| layouts_by_scene | s31__global_only | 0.8163 | 4609.8219 | 1.0933 | False |
| layouts_by_scene | s31__hierarchy_classic | 0.8940 | 7431.0749 | 1.0393 | True |
| layouts_by_scene | s31__three_linear | 0.8883 | 5893.6104 | 1.2998 | False |
| layouts_by_scene | a06__dp_optimal | 0.7109 | 4348.1518 | 0.8678 | False |
| layouts_by_scene | a06__global_only | 0.6300 | 3348.1907 | 1.3147 | False |
| layouts_by_scene | a06__hierarchy_classic | 0.9297 | 8796.2954 | 0.9627 | False |
| layouts_by_scene | a06__three_linear | 0.8440 | 6517.3221 | 1.2179 | False |
| layouts_by_scene | i29__dp_optimal | 0.7392 | 5682.9349 | 1.0507 | True |
| layouts_by_scene | i29__global_only | 0.7673 | 6077.9329 | 1.0558 | True |
| layouts_by_scene | i29__hierarchy_classic | 0.8897 | 8038.2031 | 0.9960 | True |
| layouts_by_scene | i29__three_linear | 0.9154 | 8035.4165 | 1.0234 | True |
