# CIFAR-100 empirical candidate pipeline

This experiment trains and evaluates the 24 non-deterministic candidates and
one terminal endpoint used by the depth-one CIFAR-100 hierarchy:

- 20 independent five-way WRN-16-2 specialists, one per official coarse class;
- WRN-16-2 and ResNet-18 20-way intermediate classifiers;
- ResNet-18 and WRN-28-10 100-way global classifiers.
- a pretrained ConvNeXt V2-L backbone with a new 100-way CIFAR head.

The ConvNeXt backbone is frozen. Its deterministic, evaluation-preprocessed
features are extracted once, and only the new linear head is optimized. It
always returns a prediction and is stored as the measured terminal detector;
it is not assumed to be perfectly accurate.

## Data isolation and split

Only `torchvision.datasets.CIFAR100(train=True)` is constructed. The official
10,000-image test set is never loaded. The official 50,000-image training set
is split once with seed 2025, stratified independently within every fine
class:

| Partition | Per fine class | Total | Permitted use |
|---|---:|---:|---|
| Training | 425 | 42,500 | Gradient updates |
| Model selection | 25 | 2,500 | Checkpoint selection/early stopping |
| Cascade validation | 50 | 5,000 | Empirical outcomes only |

The exact indices, source hashes, and array hashes are saved in
`checkpoints/cifar100/splits/`. The dataset archive is obtained from the
official University of Toronto URL through torchvision. Its expected archive
MD5 is `eb9058c3a382ffc7106e4002c42a8d85`.

## Reproduction

Install dependencies and prepare the verified split:

```powershell
pip install -r requirements.txt
python -m experiments.cifar100.prepare_data --download
```

Train the local WRN-16-2 initializer followed by all 24 independent models:

```powershell
python -m experiments.cifar100.train --candidate all --device cuda
```

Fit the CIFAR-100 detector head on the frozen pretrained ConvNeXt V2-L:

```powershell
python -m experiments.cifar100.train_detector --device cuda
```

The default detector feature batch is 16. This deliberately fits the 16 GB
P100 and V100 GPUs available on Sabine without requiring mixed precision;
only one GPU is used. The model, cached float32 features, and optimizer state
also fit comfortably within a Sabine GPU node's 250 GB host memory.

The pretrained model is
`convnextv2_large.fcmae_ft_in1k`, whose weights originate from Meta's official
ConvNeXt V2 release and are distributed through timm. The training record
stores the origin, delivery identifier, license, preprocessing, and SHA-256
fingerprint of the exact loaded backbone tensors. These weights are
CC-BY-NC-4.0; confirm that this is compatible with the intended publication
and downstream use.

Candidates may instead be trained in separate cluster jobs. Train
`wrn16_2_base` before any WRN-16-2 child, let every job write its own
candidate directory, and rebuild the shared manifest after all jobs finish:

```powershell
python -m experiments.cifar100.build_training_manifest `
  --output-dir checkpoints/cifar100/training `
  --device-description "HPC jobs; see per-candidate metrics"
```

Sequential partial invocations merge compatible manifest records. The final
rebuild is still required after parallel jobs because two jobs can finish at
the same time. It validates every checkpoint against the configured split
before publishing a stable, registry-ordered manifest.

On a CPU-only machine, replace `--device cuda` with `--device cpu`; the full
run is expected to be substantially slower. The default recipe is in
`configs/default_training.json`. WRN candidates never request external
weights. ResNet-18 uses only the official
`torchvision.models.ResNet18_Weights.DEFAULT` initialization; its URL, local
cache location, and SHA-256 are written into each training record. All task
heads are fully fine-tuned, and every deployable checkpoint contains a full,
independently loadable state dictionary.

Benchmark all candidates on CPU using real, evaluation-normalized
cascade-validation images that are loaded before timing:

```powershell
python -m experiments.cifar100.benchmark `
  --manifest checkpoints/cifar100/training/training_manifest.json `
  --output checkpoints/cifar100/latency.json `
  --data-root datasets/cifar100 `
  --split-npz checkpoints/cifar100/splits/cifar100_split_indices.npz `
  --threads 1 --warmups 20 --samples 500 --include-detector
```

This benchmark is intentionally CPU-only and single-threaded so its costs are
comparable between candidates. Run it as a separate Slurm compute job after
training; allocating a GPU does not accelerate this command.

Collect all 24 non-deterministic models and the trained detector over all 5,000
cascade-validation images:

```powershell
python -m experiments.cifar100.collect_empirical_outcomes `
  --training-manifest checkpoints/cifar100/training/training_manifest.json `
  --latency checkpoints/cifar100/latency.json `
  --output-dir checkpoints/cifar100/empirical
```

Generate candidate, confidence-coverage, routing, specialist OOD, latency,
complementarity, and dominance tables without rerunning inference:

```powershell
python -m experiments.cifar100.report `
  --empirical-manifest checkpoints/cifar100/empirical/manifest.json `
  --latency checkpoints/cifar100/latency.json `
  --output-dir checkpoints/cifar100/reports
```

## Smoke run

Before a full run, the following covers every non-deterministic architecture
and role with one specialist and tiny deterministic subsets. The initializer
must appear first:

```powershell
python -m experiments.cifar100.train --smoke --device cpu --cpu-threads 4 `
  --candidate wrn16_2_base wrn16_2_specialist_aquatic_mammals `
              wrn16_2_coarse resnet18_coarse resnet18_global wrn28_10_global

# Run this separately on a GPU; it still downloads the full pretrained backbone.
python -m experiments.cifar100.train_detector --smoke --device cuda

python -m experiments.cifar100.benchmark `
  --manifest checkpoints/cifar100/smoke/training/training_manifest.json `
  --output checkpoints/cifar100/smoke/latency.json `
  --data-root datasets/cifar100 `
  --split-npz checkpoints/cifar100/splits/cifar100_split_indices.npz `
  --threads 4 --warmups 2 --samples 5 --input-pool-size 16 `
  --include-detector

python -m experiments.cifar100.collect_empirical_outcomes `
  --training-manifest checkpoints/cifar100/smoke/training/training_manifest.json `
  --latency checkpoints/cifar100/smoke/latency.json `
  --output-dir checkpoints/cifar100/smoke/empirical --max-samples 16 `
  --candidate wrn16_2_specialist_aquatic_mammals wrn16_2_coarse `
              resnet18_coarse resnet18_global wrn28_10_global `
              convnextv2_large_detector
```

The collection threshold is deliberately `0.0` (accept all). It is not tuned
on model-selection data; the generic threshold optimizer later constructs
per-occurrence grids from the saved confidences. Specialists remain pure
five-way models and are evaluated on every cascade-validation image, including
the 95 out-of-group fine classes.

With `train_detector` completed, collection produces a validation-only bundle
with `detector_status="available"`. CIFAR optimization must explicitly use
`detector_mode="trained"`; the synthetic perfect-detector mode is invalid for
this experiment. A separately authorized final-evaluation partition is still
needed without repurposing these 5,000 tuning rows as a reported test set.

## Outputs

Large generated files are ignored under `checkpoints/cifar100/`:

- `splits/`: exact NPZ indices and readable checksummed manifest;
- `training/<candidate>/best.pt`: full standalone checkpoints and metrics;
- `training/convnextv2_large_detector/best.pt`: frozen-backbone endpoint plus fitted head;
- `training/training_manifest.json`: all candidate/configuration provenance;
- `latency.json`: batch-one CPU statistics and environment;
- `empirical/empirical_outcomes.pkl`: compact optimizer-compatible tables;
- `empirical/raw_outputs/*.npz`: float32 logits and probabilities;
- `empirical/manifest.json`: candidate, cost, checkpoint, mapping, and artifact manifest;
- `reports/`: JSON, CSV, and Markdown summary tables.

Run all automated tests with:

```powershell
python -m unittest discover -v
```
