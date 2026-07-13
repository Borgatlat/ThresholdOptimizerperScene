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
reproducible baseline. By default it uses the logged, imperfect **trained
Kdet** predictions and Kdet cost; the paper's perfect 10,000 ms fallback is
available only through `--detector-mode paper`.

```bash
# Compare exact Cartesian grid search with simulated annealing + coordinate descent.
python threshold_optimizer.py --method benchmark

# Run either optimizer independently.
python threshold_optimizer.py --method exhaustive --target-accuracy 0.95
python threshold_optimizer.py --method anneal --target-accuracy 0.95 --iterations 5000

# Tune on 80% of every run, then report the frozen layout/policy on the
# final 20% of every run, save the real-Kdet policy. This is the recommended
# overfitting and deployment check.
python threshold_optimizer.py --method anneal --holdout-fraction 0.20 \
  --iterations 10000 --output checkpoints/threshold_optimizer_trained_metrics.json

# Use a scene-specific empirical-output table.
python threshold_optimizer.py --outcomes checkpoints/empirical_outcomes_h08.pkl --method anneal

# Reproduce the paper's synthetic perfect-fallback assumption only when needed.
python threshold_optimizer.py --method benchmark --detector-mode paper
```

For a finite empirical table, a policy only changes when a threshold crosses
an observed confidence. `--all-observed-thresholds` therefore exposes the
exact one-model breakpoints, but their Cartesian product grows exponentially;
exhaustive mode intentionally rejects oversized searches. Increase
`--quantile-points` for a denser bounded grid, or use annealing for a large
grid.

With `--holdout-fraction`, the hierarchy is synthesized and thresholds are
selected from the validation partition only, then the frozen policy is
replayed on the holdout partition. `blocked_per_run` is the default split: it
uses each run's final contiguous segment block for holdout rather than mixing
nearby windows randomly. The current h24 table has one class per run, so this
keeps all classes in both partitions; a whole-run holdout would not.

Every final baseline, optimized, and holdout report includes
`per_class_accuracy`, `macro_accuracy`, and `worst_class_accuracy`. A class
with no evaluated samples is reported with `accuracy: null` rather than being
silently included as correct or incorrect.

## Paper-Kdet Per-Scene Experiments

`optimize_all_scenes.py` runs independent threshold experiments for every
cached scene outcome file. It uses the paper's perfect, 10,000 ms `Kdet`
assumption and sets each scene's target to that scene's baseline **validation**
accuracy. The baseline policy is part of every threshold grid, so each run has
at least one validation-feasible starting policy. Results are written to
`checkpoints/paper_kdet_baseline_target/` rather than the trained-Kdet reports.

```bash
# Run every available cached scene. Missing outcomes, such as i22, are skipped.
python optimize_all_scenes.py

# Run a selected subset or use a denser threshold grid.
python optimize_all_scenes.py --scenes a06 h08 s31 --quantile-points 25
```

Generate two figures per completed scene, plus a machine-readable collection
of the plotted values:

```bash
python plot_paper_kdet_results.py
```

Figures are saved in `checkpoints/figures/paper_kdet_baseline_target/` and
the values are saved in `checkpoints/paper_kdet_baseline_target/plot_data.json`.

## Live Runtime Benchmark

`live_cascade_benchmark.py` loads the frozen layout and thresholds from the
real-Kdet optimization report, then runs the actual Ki models on this machine.
It compares the optimized policy against the saved baseline policy on the same
inputs, alternates their order to reduce warm-cache bias, and records final
prediction accuracy against the ground-truth class. It refuses a paper-mode
metrics file so this comparison cannot silently mix two different fallbacks.

```bash
# Reproduce the saved holdout partition, then time 250 live cascade executions.
python live_cascade_benchmark.py --timed-samples 250 \
  --output checkpoints/live_cascade_benchmark.json

# Benchmark a random 250-sample subset of every processed h24 input.
python live_cascade_benchmark.py --scene h24 --partition all --timed-samples 250

# Use the full holdout partition for both live accuracy and timing.
python live_cascade_benchmark.py --timed-samples 0
```

The report contains `avg_ms`, `median_ms`, `p95_ms`, `p99_ms`, `wcet_ms`
(the largest measured latency), `min_ms`, and `std_ms` for both policies.
It also contains `accuracy`, `macro_accuracy`, `worst_class_accuracy`, and
`per_class_accuracy` from live predictions. The accuracy count includes the
untimed warmup samples; `--timed-samples 0` loads the complete selected
partition. Timing starts after input loading and host-to-device transfer,
matching the existing per-Ki profiler.

## Troubleshooting

- **Missing checkpoints**: run `python verify_setup.py`; weights live in `checkpoints/K*.pt`.
- **Scene skipped (data not found)**: download that scene from M3N-VC and place under `datasets/<scene>/`.
- **i22 multi-target runs**: i22 currently has no single-vehicle runs. The
  single-label Ki cascade cannot process it; it needs a multi-label cascade
  and evaluator rather than an arbitrary choice of one vehicle per segment.
- **Memory on large scenes**: processing is file-by-file; use `--scenes` to run one scene at a time.

## Algorithm for Threshold Optimization

The threshold optimizer can use either:

- **exhaustive search** over every combination of the allowed thresholds. If
  every one of `n` models has roughly `q` threshold values, this is
  `O(q^n)` evaluations.
- **simulated annealing with coordinate descent**, which evaluates a limited
  number of random proposals for `t` iterations, then greedily polishes the
  best policy it found.

### Terms for the Optimizer

- **Cached confidence score**: the maximum softmax probability produced by a
  model for one saved sample. These scores are collected once in
  `empirical_outcomes.pkl`; threshold tuning does not rerun the models.

- **Quantile points**: the number of equally spaced *percentiles* sampled
  from a model's cached confidence distribution. For example, four points are
  `0%, 33.3%, 66.7%, 100%`, not confidence values uniformly spaced from zero
  to one. A quantile is not an accuracy or recall value; it is a way of
  choosing thresholds where confidence values actually occur.

- **Threshold grid**: the allowed confidence thresholds for one model. It
  contains the selected confidence quantiles, the model's current threshold,
  `0.0` (accept every cached sample), and a value just above the maximum
  confidence (reject every cached sample). Between two adjacent cached
  confidence values, changing the threshold cannot change any cached route,
  so a continuous search would mostly repeat equivalent policies.

- **Policy**: one chosen threshold from every active model's grid. Replaying
  a policy gives end-to-end accuracy, expected runtime, routes, and per-class
  metrics.

- **Policy key**: the hard final ranking rule used by both optimizers. A
  policy that meets the target accuracy always beats one that misses it. Among
  feasible policies, lower expected runtime wins; accuracy breaks an exact
  runtime tie. If no policy is feasible, the smallest accuracy shortfall wins,
  then lower runtime breaks a tie.

### Exhaustive Search

This is the brute-force baseline. It evaluates every Cartesian product of the
threshold grids, then selects the policy with the best policy key. It is exact
for that discrete grid, but becomes impractical once many models or many
thresholds are used.

```text
best_policy = None

for policy in every_combination(threshold_grids):
    metrics = replay_cached_outcomes(policy)

    if policy_key(metrics) is better than policy_key(best_policy):
        best_policy = metrics

return best_policy
```

### Simulated Annealer

This is a probabilistic search over the same threshold grids. The annealing
schedule decays exponentially. Early in the search, a proposal can move many
grid positions and worse-energy proposals are sometimes accepted. Later, steps
become smaller and worse proposals become unlikely.

The proposal energy is:

$$
E = \mathrm{cost} + \mathrm{penalty}\;\max(0, \mathrm{accuracy}_{target} - \mathrm{accuracy}_{current})
$$

The second term is a ReLU-style penalty for missing the target accuracy. It
gives the annealer a stronger signal that 94.9% is preferable to 70.0% when
both policies miss a 95% target. This energy guides exploration; the policy
key above still decides the final winner.

At each iteration, the annealer chooses one model:

- **80% chance**: move that model's threshold index by a random local step.
  The maximum step decreases as the search progresses.
- **20% chance**: jump to a random threshold index in that model's grid.

```text
current = grid value nearest to each model's current threshold
current_metrics = replay_cached_outcomes(current)
best = current
best_metrics = current_metrics

for iteration in range(n_iterations):
    progress = iteration / (n_iterations - 1)
    temperature = exponential_decay(start_temperature, end_temperature, progress)

    model = random active model
    proposal = copy(current)

    if random() < 0.8:
        proposal[model] = clamp(current[model] + random_local_step(progress))
    else:
        proposal[model] = random index from that model's grid

    proposal_metrics = replay_cached_outcomes(proposal)
    delta = energy(proposal_metrics) - energy(current_metrics)

    if delta <= 0 or random() < exp(-delta / temperature):
        current = proposal
        current_metrics = proposal_metrics

    if policy_key(proposal_metrics) is better than policy_key(best_metrics):
        best = proposal
        best_metrics = proposal_metrics

return coordinate_descent(best)
```

### Coordinate Descent Polish

Coordinate descent is the greedy finishing step. It holds every threshold
fixed except one, tries every value in that model's grid, and keeps an
improvement according to the policy key. It repeats full passes until no model
improves or the maximum number of passes is reached.

```text
for pass in range(max_passes):
    changed = false

    for model in active models:
        try every threshold for model while holding all other thresholds fixed
        keep the best value if it improves the policy key
        changed = changed or an improvement was kept

    if not changed:
        break

return current policy
```

# Notes and Limitations
- Performance differs quite a bit. It performs worse when we use a real deterministic classifier compared to the 10000ms one referenced in the paper
- Accuracy differs between validation and holdout set.
