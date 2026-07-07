"""Registry table for trained Ki classifiers (RTS cascade synthesis inputs).

Each row stores the probabilistic + timing characterization required by the
hierarchical IDK cascade model: C_i (runtime), P{IDK}, P{correct}, confusion
matrix, and DAG routing constraints (allowed_next).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.labels import KI_REGISTRY, KiSpec, is_deterministic_ki, threshold_hi_for_ki

# Kdet: deterministic fallback — always eligible as terminal arbiter.
KDET = "Kdet"

# Static hierarchy DAG: for each Ki, map routing outcome -> eligible next classifiers.
# Outcomes use class label strings; "IDK" = confidence below calibrated H_i.
ALLOWED_NEXT: dict[str, dict[str, list[str]]] = {
    "K0": {
        "IDK": ["K1", "K2", "K3", KDET],
        "suv": ["K4", "K2", "K3", KDET],
        "coupe": ["K5", "K6", "K2", "K3", KDET],
        "background": [KDET],
    },
    "K1": {
        "IDK": ["K0", "K2", "K3", KDET],
        "suv": ["K4", "K2", "K3", KDET],
        "coupe": ["K5", "K6", "K2", "K3", KDET],
        "background": [KDET],
    },
    "K2": {
        "IDK": ["K0", "K1", "K3", "K4", "K5", "K6", KDET],
        "gle350": [KDET],
        "cx30": [KDET],
        "mustang": [KDET],
        "miata": [KDET],
        "background": [KDET],
    },
    "K3": {
        "IDK": ["K0", "K1", "K2", "K4", "K5", "K6", KDET],
        "gle350": [KDET],
        "cx30": [KDET],
        "mustang": [KDET],
        "miata": [KDET],
        "background": [KDET],
    },
    "K4": {
        "IDK": ["K2", "K3", KDET],
        "gle350": [KDET],
        "cx30": [KDET],
    },
    "K5": {
        "IDK": ["K2", "K3", "K6", KDET],
        "mustang": [KDET],
        "miata": [KDET],
    },
    "K6": {
        "IDK": ["K2", "K3", "K5", KDET],
        "mustang": [KDET],
        "miata": [KDET],
    },
    KDET: {},
}


@dataclass
class ClassifierRecord:
    """One trained Ki row in the cascade classifier table."""

    name: str
    level: str
    class_names: list[str]
    runtime_ms: float | None
    wcet_ms: float | None
    p_idk: float | None
    p_correct: float
    confusion_matrix: list[list[int]]
    allowed_next: dict[str, list[str]]
    threshold_hi: float | None = None
    loss_key: str | None = None
    modality: str | None = None
    num_params: int | None = None
    checkpoint: str | None = None
    macro_f1: float | None = None

    def to_dict(self) -> dict:
        row = asdict(self)
        # Paper Table IV "Success rate" ≡ val accuracy on non-deferred predictions.
        row["p_success"] = row["p_correct"]
        return row


def p_correct_from_confusion(cm: list[list[int]]) -> float:
    """P{correct} = trace(cm) / sum(cm) — paper's success probability on covered inputs."""
    arr = np.asarray(cm, dtype=np.float64)
    total = arr.sum()
    if total <= 0:
        return 0.0
    return float(np.trace(arr) / total)


@torch.inference_mode()
def compute_p_idk(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    threshold_hi: float,
) -> float:
    """Fraction of val batches where max(softmax) < H_i → model would defer (IDK)."""
    model.eval()
    idk_count = 0
    total = 0

    for batch in loader:
        if modality == "mic":
            x, _ = batch
            x = x.to(device, non_blocking=True)
            logits = model(x)
        else:
            mic, geo, _ = batch
            mic = mic.to(device, non_blocking=True)
            geo = geo.to(device, non_blocking=True)
            logits = model(mic, geo)

        probs = torch.softmax(logits, dim=1)
        max_conf = probs.max(dim=1).values
        idk_count += int((max_conf < threshold_hi).sum().item())
        total += logits.size(0)

    return float(idk_count / max(total, 1))


def allowed_next_for_ki(ki_name: str) -> dict[str, list[str]]:
    """Hierarchy routing constraints for one Ki (copy so callers cannot mutate global)."""
    return {k: list(v) for k, v in ALLOWED_NEXT.get(ki_name, {}).items()}


def record_from_train_result(
    ki_name: str,
    spec: KiSpec,
    best_metrics: dict,
    *,
    loss_key: str | None = None,
    num_params: int | None = None,
    checkpoint: str | None = None,
    runtime_ms: float | None = None,
    wcet_ms: float | None = None,
    p_idk: float | None = None,
    threshold_hi: float | None = None,
    macro_f1: float | None = None,
) -> ClassifierRecord:
    cm = best_metrics.get("confusion_matrix", [])
    p_correct = best_metrics.get("accuracy", p_correct_from_confusion(cm))
    ckpt_ref = None
    if checkpoint:
        ckpt_ref = Path(str(checkpoint).replace("\\", "/")).name

    return ClassifierRecord(
        name=ki_name,
        level=spec.level,
        class_names=list(spec.class_names),
        runtime_ms=runtime_ms,
        wcet_ms=wcet_ms,
        p_idk=p_idk,
        p_correct=float(p_correct),
        confusion_matrix=cm,
        allowed_next=allowed_next_for_ki(ki_name),
        threshold_hi=threshold_hi,
        loss_key=loss_key,
        modality=spec.modality,
        num_params=num_params,
        checkpoint=ckpt_ref,
        macro_f1=macro_f1,
    )


class ClassifierRegistry:
    """In-memory table keyed by Ki name; persists to JSON (+ optional Parquet)."""

    def __init__(self) -> None:
        self._records: dict[str, ClassifierRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, ki_name: str) -> bool:
        return ki_name in self._records

    def get(self, ki_name: str) -> ClassifierRecord | None:
        return self._records.get(ki_name)

    def upsert(self, record: ClassifierRecord) -> None:
        self._records[record.name] = record

    def upsert_from_train_result(self, result, threshold_hi: float | None = None) -> ClassifierRecord:
        """Build a row from TrainResult (training.trainer.TrainResult)."""
        spec = KI_REGISTRY[result.ki]
        record = record_from_train_result(
            result.ki,
            spec,
            result.best_metrics,
            loss_key=result.loss_key,
            num_params=result.num_params,
            checkpoint=result.checkpoint,
            runtime_ms=result.inference_avg_ms,
            wcet_ms=result.inference_wcet_ms,
            p_idk=getattr(result, "p_idk", None),
            threshold_hi=threshold_hi,
            macro_f1=result.best_val_macro_f1,
        )
        self.upsert(record)
        return record

    def to_dataframe(self) -> pd.DataFrame:
        rows = [r.to_dict() for r in self._records.values()]
        if not rows:
            return pd.DataFrame(
                columns=[
                    "name",
                    "level",
                    "runtime_ms",
                    "wcet_ms",
                    "p_idk",
                    "p_correct",
                    "macro_f1",
                    "threshold_hi",
                    "loss_key",
                    "modality",
                    "num_params",
                    "checkpoint",
                    "class_names",
                    "confusion_matrix",
                    "allowed_next",
                ]
            )
        return pd.DataFrame(rows)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classifiers": [r.to_dict() for r in sorted(self._records.values(), key=lambda r: r.name)],
        }
        path.write_text(json.dumps(payload, indent=2))

        parquet_path = path.with_suffix(".parquet")
        df = self.to_dataframe()
        # Nested columns → JSON strings for Parquet compatibility.
        for col in ("class_names", "confusion_matrix", "allowed_next"):
            if col in df.columns:
                df[col] = df[col].apply(json.dumps)
        df.to_parquet(parquet_path, index=False)

    @classmethod
    def load(cls, path: Path) -> ClassifierRegistry:
        path = Path(path)
        data = json.loads(path.read_text())
        reg = cls()
        valid_keys = {f.name for f in fields(ClassifierRecord)}
        for row in data.get("classifiers", []):
            reg.upsert(ClassifierRecord(**{k: v for k, v in row.items() if k in valid_keys}))
        return reg

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: Path,
        *,
        wcet_path: Path | None = None,
        default_threshold_hi: float | None = None,
    ) -> ClassifierRegistry:
        """Rebuild registry from existing {Ki}_metrics.json + optional wcet_profile.json."""
        checkpoint_dir = Path(checkpoint_dir)
        reg = cls()

        wcet_by_ki: dict[str, dict] = {}
        wcet_file = wcet_path or checkpoint_dir / "wcet_profile.json"
        if wcet_file.exists():
            for entry in json.loads(wcet_file.read_text()):
                wcet_by_ki[entry["ki"]] = entry

        for ki_name, spec in KI_REGISTRY.items():
            metrics_path = checkpoint_dir / f"{ki_name}_metrics.json"
            if not metrics_path.exists():
                continue

            raw = json.loads(metrics_path.read_text())
            wcet = wcet_by_ki.get(ki_name, {})
            runtime_ms = raw.get("inference_avg_ms") or wcet.get("avg_ms")
            wcet_ms = raw.get("inference_wcet_ms") or wcet.get("wcet_ms")
            hi = raw.get("threshold_hi")
            if hi is None and not is_deterministic_ki(ki_name):
                hi = default_threshold_hi if default_threshold_hi is not None else threshold_hi_for_ki(ki_name)
            

            record = record_from_train_result(
                ki_name,
                spec,
                raw.get("best_metrics", {}),
                loss_key=raw.get("loss_key"),
                num_params=raw.get("num_params"),
                checkpoint=raw.get("checkpoint"),
                runtime_ms=runtime_ms,
                wcet_ms=wcet_ms,
                p_idk=raw.get("p_idk"),
                threshold_hi=hi,
                macro_f1=raw.get("best_val_macro_f1"),
            )
            reg.upsert(record)

        return reg

    def summary_table(self) -> str:
        """Human-readable ASCII table of core cascade fields."""
        df = self.to_dataframe()
        if df.empty:
            return "(empty registry)"
        cols = [
            "name",
            "level",
            "threshold_hi",
            "runtime_ms",
            "wcet_ms",
            "p_idk",
            "p_success",
            "p_correct",
            "macro_f1",
        ]
        present = [c for c in cols if c in df.columns]
        return df[present].to_string(index=False, float_format=lambda x: f"{x:.4f}")
