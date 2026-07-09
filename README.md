# ThresholdOptimizerperScene

Per-scene IDK cascade threshold optimization on the [M3N-VC](https://github.com/UMBC-VEECO/M3N-VC) dataset. Frozen **h24**-trained Ki classifiers (K0–K6 + Kdet) are evaluated zero-shot on each scene; `run_all_scenes.py` preprocesses raw parquet into spectrogram arrays and collects empirical per-segment outcomes for the hierarchy optimizer.

## Prerequisites

- Python 3.10+
- PyTorch (CPU or CUDA)
- M3N-VC scene downloads (see below)
- Trained checkpoints under `checkpoints/` (included in this repo)

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify checkpoints and (optional) raw scene data
python verify_setup.py

# 3. Place M3N-VC scenes under datasets/ (any layout below works)
#    datasets/h24/h24/          <- nested (M3N-VC default)
#    datasets/h08/              <- flat
#    datasets/s31/s31/

# 4. Run all scenes (process raw parquet + empirical outcomes)
python run_all_scenes.py

# Or one scene, skip preprocessing if arrays already exist:
python run_all_scenes.py --scenes h08 --skip-process
```

## M3N-VC data layout

Each scene needs `*_mic.parquet`, `*_geo.parquet`, `run_ids.parquet`, and `sensor_location.parquet`. Supported paths (auto-detected):

| Layout | Example |
|--------|---------|
| Nested | `datasets/h24/h24/run0_rs1_mic.parquet` |
| Flat | `datasets/h08/run0_rs1_mic.parquet` |
| Custom | `python run_all_scenes.py --data-root /path/to/M3N-VC` |

Scenes: **h24**, **h08**, **s31**, **a06**, **i29**, **i22**.

Download from the M3N-VC release and unzip so scene folders sit under `datasets/`.

## Outputs

| Step | Output |
|------|--------|
| `process_data` | `datasets/processed/<scene>_paired_{mic,geo}.npy`, `<scene>_metadata.parquet` |
| `empirical_outcomes` | `checkpoints/empirical_outcomes_<scene>.pkl` (h24 → `empirical_outcomes.pkl`) |

## Individual commands

```bash
python process_data.py --scene h08
python empirical_outcomes.py --scene h08
python diagnose_scene.py
```

## Fixed-layout threshold optimization

`threshold_optimizer.py` replays the current hierarchy against cached raw
confidence/prediction outputs. It does not run the Ki models again. The
default target accuracy is **0.95** and the default benchmark uses a compact
five-state grid per active model, so exhaustive search remains a quick,
reproducible baseline.

```bash
# Compare exact Cartesian grid search with simulated annealing + coordinate descent.
python threshold_optimizer.py --method benchmark

# Run either optimizer independently.
python threshold_optimizer.py --method exhaustive --target-accuracy 0.95
python threshold_optimizer.py --method anneal --target-accuracy 0.95 --iterations 5000

# Use a scene-specific empirical-output table.
python threshold_optimizer.py --outcomes checkpoints/empirical_outcomes_h08.pkl --method anneal

# Compare against the imperfect logged Kdet instead of the paper's
# always-correct fallback assumption.
python threshold_optimizer.py --method benchmark --detector-mode trained
```

For a finite empirical table, a policy only changes when a threshold crosses
an observed confidence. `--all-observed-thresholds` therefore exposes the
exact one-model breakpoints, but their Cartesian product grows exponentially;
exhaustive mode intentionally rejects oversized searches. Increase
`--quantile-points` for a denser bounded grid, or use annealing for a large
grid.

## Troubleshooting

- **Missing checkpoints**: run `python verify_setup.py`; weights live in `checkpoints/K*.pt`.
- **Scene skipped (data not found)**: download that scene from M3N-VC and place under `datasets/<scene>/`.
- **i22 multi-target runs**: segments without a single-vehicle label in `run_ids.parquet` are skipped by design.
- **Memory on large scenes**: processing is file-by-file; use `--scenes` to run one scene at a time.
