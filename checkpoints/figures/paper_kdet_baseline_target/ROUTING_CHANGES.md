# Routing Changes After Per-Position Threshold Optimization

The paper-Kdet baseline-target reports were rerun with their original settings:
10,000 annealing iterations, 50 quantile points, seed 0, and a blocked 20%
holdout split.

Baseline route counts are exactly unchanged. The table compares the old and
new annealed policies. “Routing shift” is the total-variation distance between
their terminal-route distributions.

| scene | routing shift | largest terminal-route changes | holdout accuracy change | expected-cost change |
|---|---:|---|---:|---:|
| h24 | 12.13% | K3 −12.13 pp; K4 +8.32 pp; K6 +1.81 pp | +0.258 pp | +141.24 ms |
| h08 | 0.15% | K6 −0.13 pp; K3 +0.07 pp; K4 +0.05 pp | 0.000 pp | +1.68 ms |
| s31 | 5.62% | K3 −3.93 pp; K6 +2.02 pp; K2 +1.83 pp | −0.893 pp | −74.56 ms |
| a06 | 13.27% | K4 +11.26 pp; K2 −7.14 pp; detector −5.93 pp | −4.116 pp | −595.81 ms |
| i29 | 18.77% | K3 −18.77 pp; K0 +8.77 pp; K2 +4.91 pp | +0.453 pp | +8.17 ms |

## Raw annealed holdout route counts

| scene | old model-level policy | new per-position policy |
|---|---|---|
| h24 | K2 7; K3 1356; K4 20; K6 61; detector 106 | K2 16; K3 1168; K4 149; K6 89; detector 128 |
| h08 | K1 3; K2 2; K3 4553; K4 18; K6 10; detector 1382 | K1 2; K2 3; K3 4557; K4 21; K6 2; detector 1383 |
| s31 | K0 9; K1 16; K2 23; K3 2687; K4 55; K5 30; K6 172; detector 2048 | K0 98; K1 8; K2 115; K3 2489; K4 46; K6 274; detector 2010 |
| a06 | K1 2; K2 415; K3 2376; K4 223; K6 8; detector 2200 | K2 42; K3 2481; K4 811; detector 1890 |
| i29 | K1 38; K3 4585; K4 28; K6 21; detector 5918 | K0 929; K1 47; K2 520; K3 2597; K4 40; K6 527; detector 5930 |
