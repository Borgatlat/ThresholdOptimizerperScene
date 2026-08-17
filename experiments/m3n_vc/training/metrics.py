"""Evaluation metrics for hierarchical Ki training."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def _present_label_indices(y_true: np.ndarray, num_classes: int) -> list[int]:
    """Class ids that actually appear in y_true (val may omit background, etc.)."""
    present = set(int(v) for v in np.unique(y_true))
    return [i for i in range(num_classes) if i in present]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    labels = list(range(len(class_names)))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    )
    present = _present_label_indices(y_true, len(class_names))
    macro_f1_present = float(
        f1_score(
            y_true,
            y_pred,
            labels=present,
            average="macro",
            zero_division=0,
        )
    ) if present else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    per_recall = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_f1_present": macro_f1_present,
        "present_classes": [class_names[i] for i in present],
        "per_class_recall": {
            class_names[i]: float(per_recall[i]) for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
    }
