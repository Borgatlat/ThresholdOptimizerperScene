# Per-Position Threshold Rerun vs. Model-Level Thresholds

## Settings

- Original and rerun: 8,000 annealing iterations, 50 quantile points, seed 0
- Same empirical outcome files, detector modes, layouts, targets, and holdout splits
- Full six-suite threshold-variant matrix: 56 comparable summary rows
- Full per-scene matrix: 20 comparable rows

## Overall result

- 34 of 56 master-summary rows changed; 22 were identical.
- Among the changed rows, holdout accuracy increased in 23, decreased in 8,
  and was unchanged in 3.
- Holdout cost decreased in 12 changed rows and increased in 22.
- Two runs changed from holdout-infeasible to feasible:
  `targets/paper_acc_0.95` and `layouts_by_scene/s31__dp_optimal`.
- Layouts without repeated model occurrences were metric-identical controls.

## Selected holdout deltas

| run | old accuracy | new accuracy | old cost (ms) | new cost (ms) | result |
|---|---:|---:|---:|---:|---|
| h24 DP baseline, 8k | 0.967097 | 0.971613 | 697.60 | 832.66 | higher accuracy, higher cost |
| paper target 0.95 | 0.947097 | 0.951613 | 305.32 | 389.10 | became feasible |
| paper target 0.98 | 0.984516 | 0.984516 | 1478.19 | 1426.62 | same accuracy, 51.58 ms cheaper |
| trained target 0.95 | 0.939355 | 0.940000 | 15.22 | 13.56 | more accurate and cheaper |
| s31 DP | 0.781746 | 0.788095 | 4001.30 | 4316.63 | became feasible |
| i29 DP | 0.739093 | 0.739188 | 5738.92 | 5682.93 | more accurate and cheaper |
| a06 DP | 0.742917 | 0.710949 | 4857.65 | 4348.15 | lower accuracy and cost |

## The optimizer selected genuinely different thresholds by position

For the h24 paper DP layout:

| model | old shared threshold | new position thresholds |
|---|---:|---|
| K3 | 0.7308 | initial: 0.7914; K0/coupe: 0.0000; K0/suv: 0.5286 |
| K2 | 0.9397 | initial: 0.9397; K0/coupe: 0.2872; K0/suv: 0.8869 |
| K6 | 0.9994 | K0/coupe: 0.9990; K1/coupe: 0.0000 |

## Search-budget check

The old model-level policy replayed through the new evaluator with exactly the
same validation and holdout metrics, so backward compatibility is exact.

The expanded h24 policy has 11 search coordinates instead of 6. At the
original 8,000-iteration budget, its baseline validation cost was worse than
the old result (865.52 vs. 780.58 ms), showing incomplete annealer convergence.
A focused 40,000-iteration rerun reached:

- Validation: 0.965462 accuracy at 774.82 ms
- Holdout: 0.967097 accuracy at 691.92 ms

That preserves the old accuracy while reducing holdout cost by 5.68 ms. The
full 8,000-iteration matrix is therefore a fair same-budget comparison, but
larger repeated-position layouts should use a larger search budget or an
incumbent shared-threshold seed for best results.
