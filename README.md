# Hierarchical Cascade Optimizer

Dataset-neutral joint layout and occurrence-specific threshold optimization
for depth-one hierarchical cascades. The reusable root modules consume a
hierarchy profile, validation/test sets, and pretrained classifier adapters.
They cache joint empirical outcomes, construct router/group modules
dynamically, and optimize layouts and thresholds. The original
[M3N-VC](https://github.com/UMBC-VEECO/M3N-VC) experiments now live under
`experiments/m3n_vc/`.

## Repository architecture

| Layer | Files |
|---|---|
| Dataset profile and empirical cache | `cascade_profile.py`, `empirical_outcomes.py` |
| Generic hierarchy and threshold replay | `hierarchy_optimizer.py`, `threshold_optimizer.py` |
| Dynamic layout and memetic search | `layout_search.py`, `joint_optimize_hierarchy_ga.py` |
| End-to-end API | `optimization_pipeline.py` |
| Standard results and packet-only figures | `result_packets.py`, `plot_result_packets.py` |
| M3N-VC collection, training, and experiments | `experiments/m3n_vc/` |

New datasets use `ensure_and_optimize_joint()` from
`optimization_pipeline.py`. A profile declares leaf classes, superclass
groups, and router outputs. For example, two CIFAR-100 intermediate
classifiers and 20 superclasses create 40 independent branch modules
automatically. Intermediate depth is intentionally limited to one.

Figures consume only `cascade-result/v1` JSON packets. Each packet contains
the layout, thresholds, validation/test accuracy, expected cost, terminal
routes, sample counts, target, dataset fingerprint, and provenance.

```bash
# Convert completed historical h24 reports to the standard packet schema.
python -m experiments.m3n_vc.export_h24_result_packets

# Regenerate h24 comparison cost/routing figures from packets only.
python -m experiments.m3n_vc.plot_h24_method_comparison

# Plot any set of result packets from any dataset.
python plot_result_packets.py result_a.json result_b.json --output-dir figures
```

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
python -m experiments.m3n_vc.verify_setup

# 3. Place M3N-VC scenes under datasets/ (any layout below works)
#    datasets/h24/h24/          <- nested (M3N-VC default)
#    datasets/h08/              <- flat
#    datasets/s31/s31/

# 4. Run all scenes (process raw parquet + empirical outcomes)
python -m experiments.m3n_vc.run_all_scenes

# Or one scene, skip preprocessing if arrays already exist:
python -m experiments.m3n_vc.run_all_scenes --scenes h08 --skip-process
```

## M3N-VC data layout

Each scene needs `*_mic.parquet`, `*_geo.parquet`, `run_ids.parquet`, and `sensor_location.parquet`. Supported paths (auto-detected):

| Layout | Example |
|--------|---------|
| Nested | `datasets/h24/h24/run0_rs1_mic.parquet` |
| Flat | `datasets/h08/run0_rs1_mic.parquet` |
| Custom | `python -m experiments.m3n_vc.run_all_scenes --data-root /path/to/M3N-VC` |

Scenes: **h24**, **h08**, **s31**, **a06**, **i29**, **i22**.

Download from the M3N-VC release and unzip so scene folders sit under `datasets/`.

## Outputs

| Step | Output |
|------|--------|
| `process_data` | `datasets/processed/<scene>_paired_{mic,geo}.npy`, `<scene>_metadata.parquet` |
| `empirical_outcomes` | `checkpoints/empirical_outcomes_<scene>.pkl` (h24 → `empirical_outcomes.pkl`) |

## Individual commands

```bash
python -m experiments.m3n_vc.process_data --scene h08
python -m experiments.m3n_vc.collect_empirical_outcomes --scene h08
python -m experiments.m3n_vc.diagnose_scene
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

## Joint hierarchy and threshold optimization

`experiments/m3n_vc/joint_optimize_hierarchy_ga.py` approximates the completed K1-free brute
force with a constrained memetic genetic algorithm. The GA evolves a legal
initial chain plus K0's coupe and SUV branches. Every new non-detector-only
topology receives the same independent threshold optimization as the
exhaustive experiment: 8,000 simulated-annealing steps, 50 confidence
quantiles, inner seed 0, and the coordinate-descent polish. The detector-only
topology is scored directly in both methods. Validation selects the winner;
holdout is evaluated once only after the layout and thresholds have been
frozen.

```bash
# Inspect the search budget and measured runtime estimate.
python -m experiments.m3n_vc.joint_optimize_hierarchy_ga --dry-run

# Default: at most 512 unique layouts (9.23% of 5,545), about 26 minutes
# sequential at the measured exhaustive-run rate.
python -m experiments.m3n_vc.joint_optimize_hierarchy_ga

# Optional outer annealing: linearly move from diverse exploration toward
# stronger selection/refinement as unique_evaluations / 512 increases.
# This writes to checkpoints/joint_ga_annealed_k1_free_h24 by default.
python -m experiments.m3n_vc.joint_optimize_hierarchy_ga --annealed-outer-schedule

# Parallelize the independent inner optimizations within each generation.
python -m experiments.m3n_vc.joint_optimize_hierarchy_ga --workers 8

# Repeat only the stochastic outer search; keep split and inner seeds fixed.
python -m experiments.m3n_vc.joint_optimize_hierarchy_ga --outer-seed 1 \
  --output-dir checkpoints/joint_ga_k1_free_h24_seed1
```

The run automatically resumes from `checkpoint.json` and caches each layout
in `evaluations.jsonl`. Its final `summary.json` includes best-so-far history,
a validation cost/accuracy Pareto archive, winner-only holdout metrics, and—if
the exhaustive files are present—a post-search optimality gap, exact rank, and
an equal-budget uniform-random control. Exhaustive results are never read by
the search itself. See [JOINT_OPTIMIZATION_RESEARCH.md](experiments/m3n_vc/JOINT_OPTIMIZATION_RESEARCH.md)
for the method comparison, prior work, budget rationale, and oracle-replay
results.

Outer annealing does not change a candidate's fitness calculation: the inner
8,000-step threshold anneal, 50 quantiles, seed, split, target, hard accuracy
constraint, coordinate polish, evaluation budget, and restart rule remain the
same. Only GA population controls are interpolated, and each applied value is
stored in the generation history.

The completed exhaustive validation table can replay both outer schedules
over many paired seeds without repeating the expensive inner anneals:

```bash
python -m experiments.m3n_vc.benchmark_ga_outer_schedules --runs 1000 --no-output
```

This is an outer-search diagnostic only; it deliberately excludes holdout and
does not replace a final independent checkpoint run.

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
python -m experiments.m3n_vc.optimize_all_scenes

# Run a selected subset or use a denser threshold grid.
python -m experiments.m3n_vc.optimize_all_scenes --scenes a06 h08 s31 --quantile-points 25
```

Generate two figures per completed scene, plus a machine-readable collection
of the plotted values:

```bash
python -m experiments.m3n_vc.plot_paper_kdet_results
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
python -m experiments.m3n_vc.live_cascade_benchmark --timed-samples 250 \
  --output checkpoints/live_cascade_benchmark.json

# Benchmark a random 250-sample subset of every processed h24 input.
python -m experiments.m3n_vc.live_cascade_benchmark --scene h24 --partition all --timed-samples 250

# Use the full holdout partition for both live accuracy and timing.
python -m experiments.m3n_vc.live_cascade_benchmark --timed-samples 0
```

The report contains `avg_ms`, `median_ms`, `p95_ms`, `p99_ms`, `wcet_ms`
(the largest measured latency), `min_ms`, and `std_ms` for both policies.
It also contains `accuracy`, `macro_accuracy`, `worst_class_accuracy`, and
`per_class_accuracy` from live predictions. The accuracy count includes the
untimed warmup samples; `--timed-samples 0` loads the complete selected
partition. Timing starts after input loading and host-to-device transfer,
matching the existing per-Ki profiler.

## Troubleshooting

- **Missing checkpoints**: run `python -m experiments.m3n_vc.verify_setup`; weights live in `checkpoints/K*.pt`.
- **Scene skipped (data not found)**: download that scene from M3N-VC and place under `datasets/<scene>/`.
- **i22 multi-target runs**: i22 currently has no single-vehicle runs. The
  single-label Ki cascade cannot process it; it needs a multi-label cascade
  and evaluator rather than an arbitrary choice of one vehicle per segment.
- **Memory on large scenes**: processing is file-by-file; use `--scenes` to run one scene at a time.

# Explanation and  Pseudo-Code

### General Terms for Hierarchical IDK Cascades
- **model**: This is the trained unit of the cascade which takes in the input and returns either a class or IDK based on whether the classification met its confidence threshold. Models are either `Intermediate`, `Specialized`, `Global` or `Deterministic`
  - `Intermediate`: classifies into sub-classes which require further classification. This can only be run in the trunk of the hierarchy. (SUV or COUPE)
  - `Specialized`: classifies known sub-class into general class. This can only be run in branch of Intermediate Classifier (COUPE -> (MUSTANG or MX 5) or SUV -> (CX 30 or GLE 350))
  - `Global`: classifies into general class. This can be run anywhere in the hierarchy (MUSTANG, MX 5, CX 30, GLE 350)
  - `Deterministic`: Same as Global classifer but does not return IDK. This is always run at the end of a module (MUSTANG, MX 5, CX 30, GLE 350)
- **classifier**: This is an instance of a model in the cascade. The same model can appear multiple times in the layout (e.g. once at SUV branch and another at COUPE branch). However, classifiers are location dependent and therefore unique.

## Algorithm for Threshold Optimization

The threshold optimizer can use either:

- **exhaustive search** over every combination of the allowed thresholds. If
  every one of the `n` `classifiers` has `q` threshold values, this is
  `O(q^n)` evaluations. **Note** that thresholds are determined by `classifiers` and not `models`. So if the same `model` appears twice, they can have different thresholds.
- **simulated annealing with coordinate descent**, which evaluates a limited
  number of random proposals for `t` iterations, then greedily polishes the
  best policy it found. This runs in `O(t)`

### Terms for the Optimizer
- **Cached confidence score**: the maximum softmax probability produced by a
  `model` for one saved sample. These scores are collected once in
  `empirical_outcomes.pkl`; threshold tuning does not rerun the models.

- **Quantile points**: the number of equally spaced *percentiles* sampled
  from a model's cached confidence distribution. For example, four points are
  `0%, 33.3%, 66.7%, 100%`, not confidence values uniformly spaced from zero
  to one. A quantile is not an accuracy or recall value; it is a way of
  choosing thresholds where confidence values actually occur.

- **Threshold grid**: the allowed confidence thresholds for each classifier (each classifier has its own grid). It contains the selected confidence quantiles, the model's current
  threshold,
  `0.0` (accept every cached sample), and a value just above the maximum
  confidence (reject every cached sample). Between two adjacent cached
  confidence values, changing the threshold cannot change any cached route,
  so a continuous search would mostly repeat equivalent policies.

- **Policy**:  This is the collection of the  threshold slots. Replaying a policy gives end-to-end accuracy, expected runtime, routes, and per-class metrics. 
- **Policy key**: the hard final ranking rule used by both optimizers. A
  policy that meets the target accuracy always beats one that misses it. Among
  feasible policies, lower expected runtime wins; accuracy breaks an exact
  runtime tie. If no policy is feasible, the smallest accuracy shortfall wins,
  then lower runtime breaks a tie.

### Exhaustive Search

This is the brute-force baseline. It evaluates every combination of the
threshold grids, then selects the policy with the best policy key. It is exact
for that discrete grid, but becomes impractical once many occurrences or many
thresholds are used.

```text
best_policy = None

for policy in every_combination(threshold_grids):
    metrics = replay_cached_outcomes(policy)

    if policy_key(metrics) < policy_key(best_policy):
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

At each iteration, the annealer chooses one threshold slot:

- **80% chance**: move that occurrence's threshold index by a random local step.
  The maximum step decreases as the search progresses.
- **20% chance**: jump to a random threshold index in that occurrence's grid.

```python

def SA_threshold_optimize(cascade):
  for n_iterations:
    progress = iteration / (n_iterations - 1)
    temperature = exponential_decay(start_temperature, end_temperature, progress)

    previous_metric = evaluator.evaluate(cascade, cached_runs)

    classifier = random.choose_random_classifier(cascade)

    # step
    if random() < 0.8:
      max_step = max(1, int((1.0 - progress) * q_thresholds))

      new_threshold_index = clamp(classifer.threshold_index + random.int(-max_step, max_step + 1))

      classifier.threshold = classifier.threshold_grid[new_threshold_index]

    # completely new threshold
    else:
      classifier.threshold = random.choice(classifier.threshold_grid)

    proposal_metric = evaluator.evaluate(cascade, cached_runs)

    delta = energy(proposal_metrics) - energy(previous_metrics)

    if delta <= 0 or random() < exp(-delta / temperature):
      keep change
    else:
      discard change

    if policy_key(proposal_metric) < policy_key(best_metric):
      best_metric = proposal_metric
      best = cascade

  return coordinated_descent(best)

```

### Coordinate Descent Polish

Coordinate descent is the greedy finishing step. It holds every threshold
fixed except one, tries every value in that occurrence's grid, and keeps an
improvement according to the policy key. It repeats full passes until no
occurrence improves or the maximum number of passes is reached.

```python
def coordinated_descent(cascade):
  for max_passes:
    changed = false

    for classifier in cascade:
        try every threshold for classifier while holding all other thresholds fixed
        keep the best value if it improves the policy key
        changed = changed or an improvement was kept

    if not changed:
        break

  return current policy
```

## Algorithm for Joint Optimizer
The joint optimizer can use either:

- **brute-force search** over every combination of layouts. This has a time complexity of around `O(n!)`
- **Memetic Genetic Algorithm**, this is a genetic algorithm which evaluates `q` layouts for `t` generations. Thresholds are optimized using previous SA algorithm. This runs in around `O(qt)`

### Terms for the Optimizer

- **Tournament Sampler**: We sample $n$ different layouts from our layout pool and perform a tournament on them and return the best one. Usually $n = 2$ so we just sample 2 and return the better of the 2  
- **Module**: This is a trunk or branch
- **Trunk**: This is the main branch of the hierarchy/layout which you would follow given repeated IDK classifications. Each layout only has 1 trunk and it only contains global models or intermediates.
- **Branch**: This is the section of the hierarchy/layout after a specialized decision has already been made. For this repository, each hierarchy has a "Coupe" and "SUV" branch. Branches are allowed to be empty.

### Brute Force Search

This is the brute-force baseline that we aim to match. It explores every possible layout of the cascade and optimizes the thresholds of each layout to reach the target accuracy using the previous SA technique. It then selects the layout and threshold with the lowest cost. This method is extremely slow even for low classifer counts but should theoretically discover the best `layout x threshold` configuration possible that is within the budget for our threshold optimization.

```text
best_policy = None

for layout in every_combination(cascade_models):
    optimized_layout = threshold_optimize_SA(layout)
    metrics = replay_cached_outcomes(optimized_layout)

    if policy_key(metrics) < policy_key(best_policy):
        best_policy = metrics

return best_policy
```

### Memetic Genetic Algorithm

This is a genetic optimization search over cascade layouts. This search does not consider thresholds as a criteria for optimization.

We initially start off with a population of random layouts. Each layout's thresholds are optimized using the previous SA technique. The 4 best layouts are considered the elites and are added to the next population. To determine the rest of the layouts for the next population, we either chose a layout from the current population using a `tournament sampler` and then alter it through `crossovers` and/or `mutations` (determined by a crossover rate and mutation rate), or just created it from scratch (called an `immigrant`, determined by immigrant rate). The `tournament sampler` is a heuristic to have generally select good layouts, while still considering worse layouts. 

`crossovers` allow for 2 layouts to "breed" through the combination of their `modules` (`trunks` and `branches`). It does this through the `recombine` function which runs on each module of the 2 layouts. After each of the modules has been recombined, we run a `repair` function on the final layout.
- `recombine` produces a new module based on 2 modules, 1 from each layout. These modules have a probability to `select` or `mix`. `select` chooses 1 of the 2 modules to become the new module. `mix` forms a breed between the 2 modules, going through each of their classifiers, where a classifier which appears in both is always kept, while one that is in either has a 50-50 chance to be kept. The ordering of the classifier gives priority to the primary parent, which is decided randomly.
- `repair` takes in a layout and removes classifiers which are repeated, specialized branches if there are no intermediates and classifiers which would be run twice.

`mutations` allow for the layout to be directly changed. It does this by first conducting a weighted probability of which module to select (trunk - 0.4, each branch - 0.3. if there are no branches, it always edits trunk). It then performs a change, which is either `local` or a complete `resampling` (local 70%, complete resampling 30%). After this the new layout is passed through `repair`.
- `local` mutations are characterized as one of the following: a random `insertion`, `deletion`, `replacement`, `swap` or `relocation` of any classifier.
- `resampling` mutations completely discard the module and generate a new random order. The length of this order is also generated randomly.
- `repair` takes in a layout and removes classifiers which are repeated and also removes specialized branches if there are no intermediates.

`immigrants` purely create a new cascade from scratch without any prior influence.

We do not select layouts that we have already tried (except for elites). We do not select layouts that are already in the next population.

This entire process is repeated for t generations, or until no unseen legal layouts are proposed.


```python
# algorithm previously discussed
def optimize_thresholds(layout):
  ... 

def tournament_sample(layouts, n):
  random_layouts = random.choose_n(layouts, n)

  tournament = run_tournament(random_layouts)

  return tournament.winner

def repair(layout):
  remove_specialized branches(layout)
  remove_repeated_classifiers(layout)

def mutate(layout)
  module = random.weighted_choice(["trunk", "SUV", "Coupe"], [0.4, 0.3, 0.3])
  change_type = random.weighted_choice(["local", "resampling"], [0.7, 0.3])
  if change_type == "local":
    random_classifier = random.select_classifier(layout, module)
    valid_changes = get_valid_changes([insertion, deletion, replacement, swap, relocation], module)

    random_change = random.choice(valid_changes)

    random_change(random_classifier)
  
  elif change_type == "resampling":
    module = random.generate_allowed_module(module)

  repair(layout)
  return layout


def crossover(layout1, layout2):
  
  def recombine(module1, module2):
    new_module = Module()
    parent, other = random.shuffle([module1, module2])
    for classifiers in (parent ∪ other):
      if classifier in both:
        new_module.add(classifier)
      else:
        if random() < 0.5:
          new_module.add(classifier)
    return new_module

  new_layout = Layout()

  for module1, module2 in layout1.modules, layout2.modules:
    change_type = random.weighted_choice(["select", "mix"], [0.7, 0.3])
    new_module = None

    if change_type == "select":
      new_module = random.choice([module1, module2])

    elif change_type == "mix":
      new_module = recombine(module1, module2)

    new_layout.add(new_module)

  repair(new_layout)

  return new_layout

def get_immigrant():
  return random.create_layout()

def GA_joint_optimizer():
  current_population = create_starting_population()

  for n_generations:

    for layout in current_population:
      optimize_thresholds(layout)

    next_population = []

    elites = select_top_n(current_population, n_elites)
    next_population.add(elites)

    for population_count - n_elites:
      new_layout = None
      
      if random() < immigrant_rate:
        new_layout = get_immigrant()

      else:
        new_layout = tournament_sample(current_population)

        if random() < crossover_rate:
          secondary = tournament_sample(current_population)

          new_layout = crossover(new_layout, secondary)
        
        if random() < mutation_rate:
          new_layout = mutate(new_layout)


      if tried_before(new_layout) or already_inside_next_population(new_layout):
        redo

      next_population.add(new_layout)

    current_population = next_population
    
  return select_top_n(current_population, 1)
```


# Notes and Limitations
- Performance differs quite a bit. It performs worse when we use a real deterministic classifier compared to the 10000ms one referenced in the paper
- Accuracy differs between validation and holdout set.

## Threshold-optimizer experiment variants

`experiment_threshold_variants.py` sweeps the threshold optimizer across
**layouts**, **accuracy targets**, **detector modes**, **scenes**, **search
settings**, and **h24→scene transfer**. It does **not** train per-scene
classifiers or run scene switching (those are separate, later experiments).

```bash
# Full suite (writes checkpoints/threshold_experiments/)
python -m experiments.m3n_vc.experiment_threshold_variants

# Subset
python -m experiments.m3n_vc.experiment_threshold_variants --suites layouts targets transfer
```

See `checkpoints/threshold_experiments/MASTER_SUMMARY.md` for the latest table.

### Paper figures

```bash
python -m experiments.m3n_vc.plot_threshold_experiments
```

PNGs (300 DPI, serif, print-safe colors) land in
`checkpoints/figures/threshold_experiments/`:

| Figure | Content |
|--------|---------|
| `fig0_main_summary.png` | 4-panel teaser (layouts, speedup, transfer, trained-Kdet) |
| `fig1_layouts_accuracy_cost.png` | Accuracy + cost bars by cascade layout |
| `fig2_layouts_pareto.png` | Accuracy–efficiency arrows (baseline → optimized) |
| `fig3_layouts_speedup.png` | Speedup ranking by layout |
| `fig4_targets_accuracy_cost.png` | Target accuracy sweep (paper vs trained Kdet) |
| `fig5_scenes_trained_kdet.png` | Per-scene trained-Kdet retune |
| `fig6_transfer_zero_shot_vs_retune.png` | h24 layout transfer |
| `fig7_layouts_by_scene_heatmaps.png` | Layout × scene accuracy/speedup |
| `fig8_search_settings.png` | Grid density / holdout-split sensitivity |

### Per-scene threshold bank

```bash
python -m experiments.m3n_vc.experiment_per_scene_thresholds
python -m experiments.m3n_vc.plot_threshold_experiments   # regenerates fig9 / fig10 too
```

Writes:
- `checkpoints/scene_threshold_bank_{paper,trained}.json` (detector-ready bank)
- `checkpoints/threshold_experiments/per_scene_thresholds/COMPARISON.md`
- figures `fig9_per_scene_thresholds.png`, `fig10_per_scene_threshold_values.png`

Two modes: **per-scene structure+thresholds** vs **shared h24 structure + per-scene thresholds**.

### Promising next threshold-optimizer experiments

These stay in “thresholds + topology” land (no scene-switching yet):

1. ~~**Per-scene joint bank**~~ — done (`experiment_per_scene_thresholds.py`).
2. ~~**Shared structure, per-scene thresholds**~~ — done (same script, `shared_h24_structure` mode).
3. **Alternating structure ↔ thresholds** — (a) DP synthesize, (b) tune
   thresholds, (c) rebuild empirical accept tables at new thresholds, (d)
   re-synthesize cascade; repeat.
4. **Sequence-order ablations** — exhaust small permutations of initial-chain
   order (K0/K2/K3…) with the same threshold tuner.
5. **Independent vs joint thresholds** — calibrate each Ki alone (precision /
   P(IDK) matching) vs the current joint end-to-end anneal.
6. **Constraint variants** — optimize under macro-accuracy or worst-class
   accuracy floors, not only micro accuracy.
7. **Detector-cost sensitivity** — paper Kdet cost sweep (1e2…1e4 ms) with
   threshold retune after each structure change.
8. **Train/tune/test across scenes** — tune on scene A, select on B, report on C
   (stronger generalization claim than single-scene holdout).
