# Joint bake-off (h24)

F: min val cost s.t. Acc_val ≥ 0.98337

| method | joint? | val_acc | val_cost_ms | holdout_acc | holdout_cost_ms | initial_layout |
|--------|--------|---------|-------------|-------------|-----------------|----------------|
| a1 | no | 0.9833763718528082 | 1655.5135166234913 | 0.9851612903225806 | 1557.6168289414752 | `['K0', 'K3', 'K2', 'detector']` |
| a2 | no | 0.9759522272433828 | 4101.0526577059245 | 0.9780645161290322 | 4116.161260550632 | `['K3', 'K2', 'detector']` |
| a3 | yes | 0.9833763718528082 | 1647.2738027822938 | 0.9851612903225806 | 1557.4205699356567 | `['K0', 'K3', 'K2', 'detector']` |
| a4 | yes | 0.9833763718528082 | 1627.2298352506248 | 0.984516129032258 | 1530.9425544711232 | `['K0', 'K3', 'detector']` |

## Reminder
- **A1** = thresholds only (freeze S).
- **A2** = layout only (freeze H, EXPAND S).
- **A3 / A4** = **joint** optimizers (S and H both change).

