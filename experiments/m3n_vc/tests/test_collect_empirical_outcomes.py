import pandas as pd

from experiments.m3n_vc.collect_empirical_outcomes import _shared_eval_mask
from experiments.m3n_vc.utils.splits import H24_EMPIRICAL_RUNS


def test_h24_default_empirical_runs_include_background_last() -> None:
    metadata = pd.DataFrame(
        {
            "run_id": [
                "run0",
                "run1",
                "run1",
                "run2",
                "run3",
                "run4",
                "run5",
                "run6",
                "run7",
                "run8",
                "run9",
                "run9",
            ]
        }
    )

    selected = metadata.loc[_shared_eval_mask(metadata, None), "run_id"]

    assert tuple(selected.drop_duplicates()) == H24_EMPIRICAL_RUNS
    assert selected.tolist() == [
        "run1",
        "run1",
        "run3",
        "run5",
        "run7",
        "run9",
        "run9",
    ]


def test_explicit_eval_runs_still_override_h24_default() -> None:
    metadata = pd.DataFrame({"run_id": ["run1", "run3", "run9"]})

    selected = metadata.loc[
        _shared_eval_mask(metadata, {"run3"}),
        "run_id",
    ]

    assert selected.tolist() == ["run3"]
