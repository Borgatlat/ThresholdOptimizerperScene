"""Run-level train/val/test splits for h24 (no segment leakage).



Each run records a single vehicle type (or background). Validation must include at

least one run per class; otherwise macro-F1 looks like "class collapse" even when

the model is fine.

"""



from __future__ import annotations



import numpy as np

import pandas as pd



from utils.labels import KiSpec



# One long run per vehicle type + background for training.

DEFAULT_TRAIN_RUNS = {"run0", "run2", "run4", "run6", "run8"}

# Shorter companion runs: gle350, cx30, mustang, miata (all vehicle types in val).

DEFAULT_VAL_RUNS = {"run1", "run3", "run5", "run7"}

# Held-out background for final evaluation (paper used run8+run9; run8 moved to train).

DEFAULT_TEST_RUNS = {"run9"}



# Specialized SUV: train gle350+cx30 long runs; val on short companion runs.

SUV_TRAIN_RUNS = {"run0", "run2"}

SUV_VAL_RUNS = {"run1", "run3"}



# Specialized Coupe: train mustang+miata long runs; val on short companion runs.

COUPE_TRAIN_RUNS = {"run4", "run6"}

COUPE_VAL_RUNS = {"run5", "run7"}





def runs_for_ki(spec: KiSpec) -> tuple[set[str], set[str], set[str]]:

    if spec.subset == "suv":

        return SUV_TRAIN_RUNS, SUV_VAL_RUNS, DEFAULT_TEST_RUNS

    if spec.subset == "coupe":

        return COUPE_TRAIN_RUNS, COUPE_VAL_RUNS, DEFAULT_TEST_RUNS

    return DEFAULT_TRAIN_RUNS, DEFAULT_VAL_RUNS, DEFAULT_TEST_RUNS





def run_level_masks(

    metadata: pd.DataFrame,

    train_runs: set[str] | None = None,

    val_runs: set[str] | None = None,

    test_runs: set[str] | None = None,

    spec: KiSpec | None = None,

) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """Return boolean masks aligned with metadata rows."""

    if spec is not None and train_runs is None:

        train_runs, val_runs, test_runs = runs_for_ki(spec)



    train_runs = train_runs or DEFAULT_TRAIN_RUNS

    val_runs = val_runs or DEFAULT_VAL_RUNS

    test_runs = test_runs or DEFAULT_TEST_RUNS



    run_ids = metadata["run_id"].astype(str)

    train_mask = run_ids.isin(train_runs).to_numpy()

    val_mask = run_ids.isin(val_runs).to_numpy()

    test_mask = run_ids.isin(test_runs).to_numpy()

    return train_mask, val_mask, test_mask





def apply_ki_subset(

    metadata: pd.DataFrame,

    mask: np.ndarray,

    subset: str,

) -> np.ndarray:

    """Further filter indices for specialized Ki (suv / coupe branches)."""

    if subset == "all":

        return mask

    if subset == "suv":

        return mask & metadata["intermediate_label"].eq("suv").to_numpy()

    if subset == "coupe":

        return mask & metadata["intermediate_label"].eq("coupe").to_numpy()

    raise ValueError(f"Unknown subset: {subset}")


def apply_background_val_holdout(
    metadata: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    background_run: str = "run8",
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Move a segment fraction of run8 (background) from train into val.

    Val runs (run1,3,5,7) omit background; this holdout enables fair macro-F1
    during development without touching held-out test run9.
    """
    train_mask = train_mask.copy()
    val_mask = val_mask.copy()
    run_rows = metadata["run_id"].astype(str).eq(background_run).to_numpy()
    run_indices = np.where(run_rows)[0]
    if len(run_indices) == 0:
        return train_mask, val_mask

    rng = np.random.default_rng(seed)
    perm = rng.permutation(run_indices)
    n_val = max(1, int(len(perm) * holdout_frac))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_mask[run_rows] = False
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    return train_mask, val_mask
