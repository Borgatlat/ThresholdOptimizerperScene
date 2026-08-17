# Per-Scene Threshold Bank Results

Oracle thresholds for each scene. No scene-switching was run.

| experiment | scene | holdout acc | cost (ms) | speedup | feasible | layout |
|---|---|---:|---:|---:|---|---|
| per_scene_structure__paper | h24 | 0.9671 | 697.6041 | 1.8188 | True | `K0→K3→K2→K1→detector` |
| per_scene_structure__paper | h08 | 0.9137 | 2351.7600 | 1.2499 | True | `K0→K3→K2→K1→detector` |
| per_scene_structure__paper | s31 | 0.7817 | 4001.2969 | 1.1260 | False | `K0→K3→K2→K1→detector` |
| per_scene_structure__paper | a06 | 0.7429 | 4857.6521 | 0.7768 | False | `K0→K3→K2→K1→detector` |
| per_scene_structure__paper | i29 | 0.7391 | 5738.9238 | 1.0405 | True | `K0→K2→K3→K1→detector` |
| shared_h24_structure__paper | h24 | 0.9671 | 697.6041 | 1.8188 | True | `K0→K3→K2→K1→detector` |
| shared_h24_structure__paper | h08 | 0.9145 | 2355.6780 | 1.2478 | True | `K0→K3→K2→K1→detector` |
| shared_h24_structure__paper | s31 | 0.7881 | 4134.5993 | 1.0916 | True | `K0→K3→K2→K1→detector` |
| shared_h24_structure__paper | a06 | 0.6769 | 3925.0751 | 0.9610 | False | `K0→K3→K2→K1→detector` |
| shared_h24_structure__paper | i29 | 0.7317 | 5676.5425 | 1.0520 | True | `K0→K3→K2→K1→detector` |
| per_scene_structure__trained | h24 | 0.9323 | 8.8285 | 1.4637 | True | `K0→K3→detector` |
| per_scene_structure__trained | h08 | 0.7626 | 15.9815 | 1.2897 | True | `K0→detector` |
| per_scene_structure__trained | s31 | 0.4950 | 23.0207 | 1.0609 | True | `K0→detector` |
| per_scene_structure__trained | a06 | 0.3425 | 9.4578 | 2.8866 | False | `K0→detector` |
| per_scene_structure__trained | i29 | 0.2532 | 15.4674 | 1.6263 | False | `K0→detector` |
| shared_h24_structure__trained | h24 | 0.9323 | 8.8285 | 1.4637 | True | `K0→K3→detector` |
| shared_h24_structure__trained | h08 | 0.7634 | 10.6538 | 1.9490 | True | `K0→K3→detector` |
| shared_h24_structure__trained | s31 | 0.4927 | 11.5950 | 2.3042 | False | `K0→K3→detector` |
| shared_h24_structure__trained | a06 | 0.4095 | 12.3990 | 2.0309 | False | `K0→K3→detector` |
| shared_h24_structure__trained | i29 | 0.2613 | 5.9612 | 5.2756 | False | `K0→K3→detector` |
