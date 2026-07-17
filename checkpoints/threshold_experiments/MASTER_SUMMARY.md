# Threshold Optimizer Experiment Summary

Scene-switching / per-scene classifier training was **not** run.

| suite | run | holdout acc | holdout cost (ms) | speedup vs baseline | feasible |
|---|---|---:|---:|---:|---|
| layouts | dp_optimal | 0.9671 | 697.6041 | 1.8188 | True |
| layouts | global_only | 0.9755 | 969.3845 | 1.4037 | True |
| layouts | single_global | 0.9852 | 1561.0627 | 1.0000 | True |
| layouts | hierarchy_classic | 0.9871 | 4267.9469 | 1.0272 | True |
| layouts | three_linear | 0.9890 | 1917.8680 | 4.1004 | True |
| layouts | three_global | 0.9755 | 969.3845 | 1.4037 | True |
| layouts | k0_k2_k3_hierarchy | 0.9684 | 733.0363 | 1.8182 | True |
| layouts | both_identifiers | 0.9755 | 986.4690 | 2.2416 | True |
| targets | paper_baseline | 0.9671 | 697.6041 | 1.8188 | True |
| targets | paper_acc_0.90 | 0.9084 | 6.2736 | 202.2443 | True |
| targets | paper_acc_0.95 | 0.9471 | 305.3175 | 4.1557 | False |
| targets | paper_acc_0.98 | 0.9845 | 1478.1919 | 0.8583 | True |
| targets | trained_baseline | 0.9323 | 8.8285 | 1.4637 | True |
| targets | trained_acc_0.90 | 0.9090 | 6.2736 | 2.0597 | True |
| targets | trained_acc_0.95 | 0.9394 | 15.2205 | 0.8490 | False |
| targets | trained_acc_0.98 | 0.9394 | 15.2205 | 0.8490 | False |
| scenes_trained | h24 | 0.9323 | 8.8285 | 1.4637 | True |
| scenes_trained | h08 | 0.7626 | 15.9815 | 1.2897 | True |
| scenes_trained | s31 | 0.4950 | 23.0207 | 1.0609 | True |
| scenes_trained | a06 | 0.3425 | 9.4578 | 2.8866 | False |
| scenes_trained | i29 | 0.2532 | 15.4674 | 1.6263 | False |
| transfer_zero_shot | h24 | 0.9658 | 763.9786 |  | None |
| transfer_zero_shot | h08 | 0.8888 | 2402.1695 |  | None |
| transfer_zero_shot | s31 | 0.7424 | 3347.8290 |  | None |
| transfer_zero_shot | a06 | 0.6653 | 3513.6382 |  | None |
| transfer_zero_shot | i29 | 0.6579 | 4251.2862 |  | None |
| transfer_retune | h24 | 0.9671 | 697.6041 | 1.8188 | None |
| transfer_retune | h08 | 0.9145 | 2355.6780 | 1.2478 | None |
| transfer_retune | s31 | 0.7881 | 4134.5993 | 1.0916 | None |
| transfer_retune | a06 | 0.6769 | 3925.0751 | 0.9610 | None |
| transfer_retune | i29 | 0.7317 | 5676.5425 | 1.0520 | None |
| search_settings | q10_blocked | 0.9677 | 854.8412 | 1.4842 | True |
| search_settings | q25_blocked | 0.9665 | 718.5180 | 1.7658 | True |
| search_settings | q50_blocked | 0.9671 | 697.6041 | 1.8188 | True |
| search_settings | q100_blocked | 0.9671 | 705.2646 | 1.7990 | True |
| search_settings | q50_random | 0.9574 | 834.7057 | 1.6674 | False |
| layouts_by_scene | h24__dp_optimal | 0.9671 | 697.6041 | 1.8188 | True |
| layouts_by_scene | h24__global_only | 0.9755 | 969.3845 | 1.4037 | True |
| layouts_by_scene | h24__hierarchy_classic | 0.9871 | 4267.9469 | 1.0272 | True |
| layouts_by_scene | h24__three_linear | 0.9890 | 1917.8680 | 4.1004 | True |
| layouts_by_scene | h08__dp_optimal | 0.9137 | 2351.7600 | 1.2499 | True |
| layouts_by_scene | h08__global_only | 0.9254 | 2699.4621 | 1.1809 | True |
| layouts_by_scene | h08__hierarchy_classic | 0.9657 | 6542.7393 | 0.9916 | True |
| layouts_by_scene | h08__three_linear | 0.9745 | 4201.9746 | 1.8526 | True |
| layouts_by_scene | s31__dp_optimal | 0.7817 | 4001.2969 | 1.1260 | False |
| layouts_by_scene | s31__global_only | 0.8163 | 4609.8219 | 1.0933 | False |
| layouts_by_scene | s31__hierarchy_classic | 0.8940 | 7431.0749 | 1.0393 | True |
| layouts_by_scene | s31__three_linear | 0.8883 | 5893.6104 | 1.2998 | False |
| layouts_by_scene | a06__dp_optimal | 0.7429 | 4857.6521 | 0.7768 | False |
| layouts_by_scene | a06__global_only | 0.6300 | 3348.1907 | 1.3147 | False |
| layouts_by_scene | a06__hierarchy_classic | 0.9297 | 8796.2954 | 0.9627 | False |
| layouts_by_scene | a06__three_linear | 0.8440 | 6517.3221 | 1.2179 | False |
| layouts_by_scene | i29__dp_optimal | 0.7391 | 5738.9238 | 1.0405 | True |
| layouts_by_scene | i29__global_only | 0.7673 | 6077.9329 | 1.0558 | True |
| layouts_by_scene | i29__hierarchy_classic | 0.8897 | 8038.2031 | 0.9960 | True |
| layouts_by_scene | i29__three_linear | 0.9154 | 8035.4165 | 1.0234 | True |
