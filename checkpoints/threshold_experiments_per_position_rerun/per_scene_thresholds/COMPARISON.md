# Per-Scene Threshold Bank Results

Oracle thresholds for each scene. No scene-switching was run.

| experiment | scene | holdout acc | cost (ms) | speedup | feasible | layout |
|---|---|---:|---:|---:|---|---|
| per_scene_structure__paper | a06 | 0.7109 | 4348.1518 | 0.8678 | False | `K0 → K3 → K2 → K1 → detector` |
| per_scene_structure__paper | h08 | 0.9142 | 2438.2189 | 1.2056 | True | `K0 → K3 → K2 → K1 → detector` |
| per_scene_structure__paper | h24 | 0.9716 | 832.6563 | 1.5238 | True | `K0 → K3 → K2 → K1 → detector` |
| per_scene_structure__paper | i29 | 0.7392 | 5682.9349 | 1.0507 | True | `K0 → K2 → K3 → K1 → detector` |
| per_scene_structure__paper | s31 | 0.7881 | 4316.6270 | 1.0437 | True | `K0 → K3 → K2 → K1 → detector` |
| shared_h24_structure__paper | a06 | 0.7138 | 4456.4265 | 0.8464 | False | `K0 → K3 → K2 → K1 → detector` |
| shared_h24_structure__paper | h08 | 0.9159 | 2370.9109 | 1.2398 | True | `K0 → K3 → K2 → K1 → detector` |
| shared_h24_structure__paper | h24 | 0.9716 | 832.6563 | 1.5238 | True | `K0 → K3 → K2 → K1 → detector` |
| shared_h24_structure__paper | i29 | 0.7378 | 5732.1404 | 1.0418 | True | `K0 → K3 → K2 → K1 → detector` |
| shared_h24_structure__paper | s31 | 0.7851 | 4116.5109 | 1.0964 | False | `K0 → K3 → K2 → K1 → detector` |
| per_scene_structure__trained | a06 | 0.3425 | 9.4578 | 2.8866 | False | `K0 → detector` |
| per_scene_structure__trained | h08 | 0.7619 | 15.8398 | 1.3012 | True | `K0 → detector` |
| per_scene_structure__trained | h24 | 0.9323 | 8.8200 | 1.4651 | True | `K0 → K3 → detector` |
| per_scene_structure__trained | i29 | 0.2532 | 15.4674 | 1.6263 | False | `K0 → detector` |
| per_scene_structure__trained | s31 | 0.4929 | 23.1909 | 1.0532 | True | `K0 → detector` |
| shared_h24_structure__trained | a06 | 0.4110 | 11.8698 | 2.1214 | False | `K0 → K3 → detector` |
| shared_h24_structure__trained | h08 | 0.7634 | 10.6437 | 1.9509 | True | `K0 → K3 → detector` |
| shared_h24_structure__trained | h24 | 0.9323 | 8.8200 | 1.4651 | True | `K0 → K3 → detector` |
| shared_h24_structure__trained | i29 | 0.2613 | 5.9612 | 5.2756 | False | `K0 → K3 → detector` |
| shared_h24_structure__trained | s31 | 0.4937 | 11.9047 | 2.2443 | False | `K0 → K3 → detector` |
