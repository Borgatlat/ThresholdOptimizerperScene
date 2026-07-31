# A4 Nested Joint Approximator

## What it is

**A4** is a **joint approximator** over \((S, H)\):

1. **Outer loop:** try a small discrete family of cascade layouts \(S\) (size \(L=4\)).
2. **Inner loop:** for each fixed \(S\), run the **SA threshold approximator** (8000 iters + coordinate descent) to get \(H\).
3. **Pick** the feasible bank with lowest validation expected cost under  
   \(F = \min \mathbb{E}[\mathrm{cost}]\) s.t. \(\mathrm{Acc}_{\mathrm{val}} \ge 0.98337\).

A4 is **not** only a threshold approximator: it can change layout.  
It **uses** the SA approximator inside each layout.

## Layout family

| Name | Initial |
|------|---------|
| `dp_expand` | validation DP / EXPAND |
| `teammate_short` | `K0 → K3 → detector` |
| `short_global` | `K0 → K2 → detector` |
| `k0_branchy` | `K0 → detector` |

K1 is removed from the candidate pool.

## Time complexity

\[
O(L \cdot I \cdot N \cdot D)
\]

- \(L\) = #layouts (4)  
- \(I\) = SA iterations (8000)  
- \(N\) = validation samples  
- \(D\) = cascade depth  

≈ \(L\) times one A1 SA run (not exponential in Kis).

## Run

```text
python joint_a4_nested.py --outcomes <empirical_outcomes.pkl> --save
```

## Frozen h24 results (this branch)

See `checkpoints/joint_opt/h24/`:

| Method | Joint? | Val Acc | Val cost | Holdout Acc | Holdout cost | Initial |
|--------|--------|---------|----------|-------------|--------------|---------|
| A1 | no | 0.983376 | 1655.51 | 0.985161 | 1557.62 | K0→K3→K2→detector |
| A4 | **yes** | 0.983376 | **1627.23** | 0.984516 | **1530.94** | **K0→K3→detector** |

Winner: **A4 / `teammate_short`**.
