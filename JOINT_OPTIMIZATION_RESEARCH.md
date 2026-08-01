# Joint hierarchy/threshold search: method selection

## Experimental contract

The approximate optimizer changes only the outer layout search. Every visited
non-detector-only layout is scored under the completed brute-force contract:

- h24 `checkpoints/empirical_outcomes.pkl`;
- K1 removed before splitting;
- the same 5,545-layout legal grammar;
- perfect paper Kdet at 10,000 ms;
- blocked-per-run 80/20 validation/holdout split, split seed 0;
- target validation accuracy `0.9833763718528082`;
- 50 empirical confidence quantiles per threshold occurrence; and
- a fresh 8,000-step threshold SA with seed 0, followed by coordinate descent.

The detector-only layout is scored directly in both methods. Layout and
threshold selection use validation only. Holdout is replayed once after the
winning validation policy is frozen.

## Methods considered

| Method | Strength here | Why it was not selected first |
|---|---|---|
| One joint simulated-annealing chain | Simple; direct precedent for changing order, membership, and thresholds exists in [US10452839B1](https://patents.google.com/patent/US10452839B1/en). | One trajectory cannot recombine a good trunk and branches found in different regions. The measured layout landscape has many local minima. |
| Parallel tempering | Hot replicas can cross energy barriers by exchanging states; see [Hukushima and Nemoto (1996)](https://doi.org/10.1143/JPSJ.65.1604) and the [Earl–Deem review](https://doi.org/10.1039/B509983H). | A fully joint replica has topology-dependent threshold dimension and needs carefully defined cross-layout moves. A nested version still pays an 8k inner anneal for every outer state and adds a temperature ladder and exchange calibration. |
| Greedy or beam search | Very cheap and interpretable. [Streeter (ICML 2018)](https://proceedings.mlr.press/v80/streeter18a.html) gives a strong approximation result for suitable linear cascading constraints. | This experiment's fixed global accuracy constraint is not the decomposable constraint required by that guarantee. A weak partial branch can also become useful only after another routing decision, so prefix pruning is unsafe. |
| Dynamic programming / branch and bound | Can be exact when useful subproblem structure or bounds exist. The [Microsoft cascade patent](https://patents.google.com/patent/US20070112701A1/en) describes DP, SA, and branch-and-bound for quantized thresholds in a fixed linear cascade. | Per-sample correlated routing means a topology edit changes the populations reaching all downstream thresholds. No useful layout lower bound is currently available. |
| Probabilistic/differentiable relaxation | Soft cascades can optimize cost and accuracy jointly; examples include [Raykar et al. (KDD 2010)](https://doi.org/10.1145/1835804.1835912) and [Chen et al. (AISTATS 2012)](https://proceedings.mlr.press/v22/chen12c.html). | Those approaches retrain differentiable stage classifiers and generally assume a fixed/cumulative feature sequence. This repository has frozen K0–K6 predictions and a discrete routed hierarchy. |
| Fixed-topology probabilistic operating points | The IBM method in [US8433669B2](https://patents.google.com/patent/US8433669B2/en) optimizes independent outgoing-edge operating points subject to resource constraints. | It assumes a supplied topology and propagates aggregate detection/false-alarm rates. That loses the cross-model, per-sample correlation retained by the empirical outcome table. |
| Monte Carlo tree search | CASCARO searches order with MCTS and iterates rejection regions; see [Hanczar and Bar-Hen (2021)](https://doi.org/10.1016/j.patrec.2021.06.010). | Its stages add cumulative variables and retrain classifiers. Here, a partial layout lacks a reliable cheap reward unless it receives the expensive inner threshold optimization. |
| Genetic/evolutionary search | Natural variable-length ordered representation, modular branch crossover, population diversity, independent parallel fitness calls. A problem-specific evolutionary cascade is also studied by [Hamilton and Fulp (CEC 2022)](https://arxiv.org/abs/2205.00570). | It has no approximation guarantee and needs an equal-budget random control. |

The closest direct design precedent is IBM's
[“Optimizing cascade of classifiers schema using genetic search”](https://patents.google.com/patent/US12443855B2/en),
which represents stage count, classifier choices, hyperparameters, and
thresholds in a genetic search. That disclosure is for linear cascades and is
not evidence that a GA will outperform another method on this dataset, but it
supports the representation choice.

## Selected method

The implementation uses a constrained **memetic genetic algorithm**:

```text
outer GA: legal hierarchy topology
    -> exact per-layout 8k threshold SA
    -> coordinate-descent threshold polish
    -> hard-constrained validation fitness
```

This is bilevel joint optimization: a layout is never judged at registry
thresholds; it is judged only with thresholds optimized specifically for its
occurrences.

The genome is `(initial, coupe, suv)`, without terminal detector nodes. Repair
projects every mutation/crossover back into the exhaustive grammar. Operators
insert, delete, replace, swap, relocate, or resample an ordered component;
crossover exchanges or mixes complete trunk/branch building blocks. The same
model may occur on mutually exclusive paths, and its threshold occurrences
remain independent because every new topology gets a fresh evaluator.

The exact outer ordering is retained:

1. every feasible policy beats every infeasible policy;
2. feasible policies minimize validation expected cost, then maximize accuracy;
3. infeasible policies maximize accuracy, then minimize cost; and
4. exhaustive layout index breaks a remaining tie.

Defaults are population 32, four elites, tournament size 2, 80% crossover,
80% mutation, 30% whole-component resampling within mutation, 20% uniform
random immigrants, and a restart after six stagnant generations. Only the
pre-existing K3→Kdet reference is seeded; all other initial layouts are
uniform samples. The primary stop is 512 unique layouts, not proposal count.

## Budget and expected runtime

The completed exhaustive run took 17,066.2 seconds for 5,545 layouts, or
3.078 seconds/layout. Therefore:

| Unique layouts | Space | Sequential estimate | Inner SA steps |
|---:|---:|---:|---:|
| 512 (default) | 9.23% | 26.3 min | about 4.10 million |
| 768 | 13.85% | 39.4 min | about 6.14 million |
| 5,545 exhaustive | 100% | 4.74 h observed | about 44.35 million |

`--workers N` evaluates a generation concurrently. Dividing the sequential
estimate by `N` is only an ideal lower bound because generations synchronize
and NumPy/CPU/memory contention can limit scaling.

## Ground-truth landscape and interpretation

The exhaustive validation results contain 30 layouts within 5 ms, 100 within
10 ms, and 260 within 20 ms of the optimum. At the same time, there are many
strict local minima and the exact winner has a small one-edit basin. This
explains why local search can become trapped while uniform sampling can still
find a strong result.

In a post-search topology-only replay of the final defaults over 1,000 outer
seeds (using exhaustive **validation** fitness as an oracle, never holdout),
the GA and uniform random search were close at a 512-layout budget:

| Search | Median rank | 90th-percentile rank | Top-10 | Exact winner |
|---|---:|---:|---:|---:|
| Memetic GA | 7 | 26 | 63.5% | 5.8% |
| K3-seeded uniform random | 7 | 24 | 64.0% | 9.3% |

The GA had a better worst observed rank (41 versus 72) and reached the top 1%
in every replay, but it did **not** dominate random search on this small,
forgiving instance. The GA is selected as the more extensible structured
optimizer—not because this result establishes superiority. Any research
report should include the equal-budget random control and emphasize validation
regret/rank rather than exact-winner recovery.

For outer seed 0, topology replay predicts rank 2 at 1,627.920 ms validation
cost, 0.691 ms above the exhaustive optimum; its paired random control reaches
rank 4. The real run still needs to be executed because this replay validates
the topology search using already-computed fitness rather than producing a new
GA checkpoint.

## Running it

```bash
python joint_optimize_hierarchy_ga.py --dry-run
python joint_optimize_hierarchy_ga.py
```

Use a new output directory for independent outer seeds:

```bash
python joint_optimize_hierarchy_ga.py --outer-seed 1 \
  --output-dir checkpoints/joint_ga_k1_free_h24_seed1
```

The output cache is guarded by hashes of the outcome table, legal catalogue,
and fitness implementation. `summary.json` reports best-so-far history,
holdout metrics for the frozen winner, Pareto points, exact exhaustive rank
and regret when a complete compatible reference exists, and a matched uniform
random control computed only after search.
