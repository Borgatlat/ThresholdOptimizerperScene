"""Reproducible model-only CPU latency benchmarks for CIFAR-100 candidates.

Each candidate checkpoint is loaded independently.  Timing deliberately starts
after input preparation, so the reported values describe model execution and
not storage or data-loader overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from experiments.cifar100.labels import CIFAR100_PROFILE


LATENCY_SCHEMA_VERSION = "cifar100-latency/v1"
DEFAULT_DATA_ROOT = Path("datasets/cifar100")
DEFAULT_SPLIT_NPZ = Path(
    "checkpoints/cifar100/splits/cifar100_split_indices.npz"
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Settings shared by every candidate in one benchmark run."""

    warmups: int = 20
    timed_samples: int = 500
    cpu_threads: int = 1
    input_channels: int = 3
    input_height: int = 32
    input_width: int = 32
    precision: str = "float32"
    seed: int = 20260805
    input_pool_size: int = 500

    def validate(self) -> None:
        if self.warmups < 0:
            raise ValueError("warmups must be nonnegative")
        if self.timed_samples <= 0:
            raise ValueError("timed_samples must be positive")
        if self.cpu_threads <= 0:
            raise ValueError("cpu_threads must be positive")
        if min(self.input_channels, self.input_height, self.input_width) <= 0:
            raise ValueError("input dimensions must be positive")
        if self.precision != "float32":
            raise ValueError("CPU benchmarking currently supports float32 only")
        if self.input_pool_size <= 0:
            raise ValueError("input_pool_size must be positive")


ModelLoader = Callable[[Mapping[str, object]], object]
InputFactory = Callable[[Mapping[str, object], BenchmarkConfig], Sequence[torch.Tensor]]


def _manifest_candidates(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    raw = manifest.get("candidates")
    if isinstance(raw, Mapping):
        candidates = []
        for candidate_id, value in raw.items():
            if not isinstance(value, Mapping):
                raise ValueError("Every candidate manifest entry must be an object.")
            item = dict(value)
            item.setdefault("candidate_id", str(candidate_id))
            candidates.append(item)
        return candidates
    if not isinstance(raw, list):
        raise ValueError("The manifest requires a candidates list or object.")
    if not all(isinstance(item, Mapping) for item in raw):
        raise ValueError("Every candidate manifest entry must be an object.")
    return [dict(item) for item in raw]


def _candidate_id(entry: Mapping[str, object]) -> str:
    value = entry.get("candidate_id", entry.get("id"))
    nested = entry.get("candidate")
    if value is None and isinstance(nested, Mapping):
        value = nested.get("candidate_id", nested.get("id"))
    if value is None or not str(value):
        raise ValueError("A candidate entry is missing candidate_id.")
    return str(value)


def _candidate_value(entry: Mapping[str, object], key: str, default: object = None) -> object:
    if key in entry:
        return entry[key]
    nested = entry.get("candidate")
    if isinstance(nested, Mapping) and key in nested:
        return nested[key]
    return default


def _entry_path(entry: Mapping[str, object], key: str) -> str | None:
    direct = entry.get(key)
    if direct is not None:
        return str(direct)
    nested_name = key.removesuffix("_path")
    nested = entry.get(nested_name)
    if isinstance(nested, Mapping) and nested.get("path") is not None:
        return str(nested["path"])
    return None


def _resolve_path(path: str | Path, base_dir: Path | None) -> Path:
    result = Path(path).expanduser()
    if not result.is_absolute() and base_dir is not None:
        result = base_dir / result
    return result.resolve()


def _registry_lookup(registry: object, candidate_id: str) -> object:
    by_id = getattr(registry, "by_id", None)
    if callable(by_id):
        return by_id(candidate_id)
    if isinstance(by_id, Mapping):
        try:
            return by_id[candidate_id]
        except KeyError as exc:
            raise ValueError(f"Unknown candidate id {candidate_id!r}.") from exc
    if isinstance(registry, Mapping):
        try:
            return registry[candidate_id]
        except KeyError as exc:
            raise ValueError(f"Unknown candidate id {candidate_id!r}.") from exc
    candidates = getattr(registry, "candidates", None)
    if candidates is not None:
        for candidate in candidates:
            value = getattr(candidate, "candidate_id", getattr(candidate, "id", None))
            if str(value) == candidate_id:
                return candidate
    raise TypeError("The candidate registry has no supported by-id interface.")


def load_manifest_candidate(
    entry: Mapping[str, object], *, manifest_dir: str | Path | None = None
) -> tuple[torch.nn.Module, Mapping[str, object]]:
    """Load and validate one independently saved candidate checkpoint."""

    from experiments.cifar100.models import candidate_specs, load_checkpoint

    candidate_id = _candidate_id(entry)
    registry = candidate_specs(
        CIFAR100_PROFILE.groups,
        CIFAR100_PROFILE.global_classes,
    )
    expected_spec = _registry_lookup(registry, candidate_id)
    path_value = _entry_path(entry, "checkpoint_path")
    if path_value is None:
        raise ValueError(f"Candidate {candidate_id!r} has no checkpoint path.")
    base = None if manifest_dir is None else Path(manifest_dir)
    checkpoint_path = _resolve_path(path_value, base)
    loaded = load_checkpoint(
        checkpoint_path,
        expected_spec=expected_spec,
        map_location="cpu",
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise TypeError("load_checkpoint must return (model, metadata).")
    model, metadata = loaded
    if not isinstance(model, torch.nn.Module):
        raise TypeError("The loaded checkpoint did not produce a torch module.")
    if not isinstance(metadata, Mapping):
        raise TypeError("Checkpoint metadata must be a mapping.")
    return model, metadata


def _default_inputs(
    _entry: Mapping[str, object], config: BenchmarkConfig
) -> Sequence[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    pool_size = min(
        config.input_pool_size,
        max(1, config.warmups + config.timed_samples),
    )
    # Inputs are allocated before timing and already have a batch dimension of one.
    tensor = torch.randn(
        pool_size,
        config.input_channels,
        config.input_height,
        config.input_width,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    )
    return tuple(tensor[index : index + 1] for index in range(pool_size))


def load_cascade_validation_inputs(
    data_root: str | Path,
    split_npz: str | Path,
    *,
    config: BenchmarkConfig,
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Preload normalized official-training cascade-validation inputs."""

    from experiments.cifar100.data import (
        CIFAR100_MEAN,
        CIFAR100_STD,
        IMAGENET_MEAN,
        IMAGENET_STD,
        build_convnext_evaluation_transform,
        build_dataset_view,
        build_evaluation_transform,
        load_split_bundle,
        load_training_dataset,
    )

    config.validate()
    data_path = Path(data_root).resolve()
    split_path = Path(split_npz).resolve()
    split_manifest = split_path.with_suffix(".json")
    dataset = load_training_dataset(data_path, download=False)
    splits = load_split_bundle(
        split_path,
        manifest_path=split_manifest,
        fine_targets=dataset.fine_targets,
        coarse_targets=dataset.coarse_targets,
    )
    pool_size = min(config.input_pool_size, config.timed_samples)
    source_indices = splits.cascade_validation[:pool_size]
    if not len(source_indices):
        raise ValueError("The cascade-validation input pool is empty.")
    detector_input = config.input_height == 224 and config.input_width == 224
    if detector_input:
        transform = build_convnext_evaluation_transform()
        preprocessing_mean = IMAGENET_MEAN
        preprocessing_std = IMAGENET_STD
        pipeline = ["Resize(256,bicubic)", "CenterCrop(224)", "ToTensor", "Normalize"]
    elif config.input_height == 32 and config.input_width == 32:
        transform = build_evaluation_transform()
        preprocessing_mean = CIFAR100_MEAN
        preprocessing_std = CIFAR100_STD
        pipeline = ["ToTensor", "Normalize"]
    else:
        raise ValueError("Unsupported CIFAR benchmark input resolution.")
    view = build_dataset_view(
        dataset,
        source_indices,
        target_mode="fine",
        transform=transform,
    )
    inputs: list[torch.Tensor] = []
    for index in range(len(view)):
        sample, _ = view[index]
        if not isinstance(sample, torch.Tensor) or sample.shape != (
            config.input_channels,
            config.input_height,
            config.input_width,
        ):
            raise ValueError("Evaluation preprocessing produced an unexpected tensor.")
        inputs.append(sample.to(dtype=torch.float32, device="cpu").unsqueeze(0).contiguous())
    index_digest = hashlib.sha256(
        np.asarray(source_indices, dtype=np.int64).tobytes()
    ).hexdigest()
    provenance: dict[str, object] = {
        "source": "official_training_cascade_validation",
        "data_root": str(data_path),
        "split_indices": str(split_path),
        "split_indices_sha256": _file_sha256(split_path),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": _file_sha256(split_manifest),
        "pool_samples": len(inputs),
        "source_indices_sha256": index_digest,
        "preprocessing": {
            "pipeline": pipeline,
            "mean": list(preprocessing_mean),
            "std": list(preprocessing_std),
            "input_resolution": [config.input_height, config.input_width],
        },
        "preloaded_before_timing": True,
    }
    return tuple(inputs), provenance


def _unwrap_loaded(value: object) -> tuple[torch.nn.Module, Mapping[str, object]]:
    if isinstance(value, torch.nn.Module):
        return value, {}
    if isinstance(value, tuple) and len(value) == 2:
        model, metadata = value
        if isinstance(model, torch.nn.Module) and isinstance(metadata, Mapping):
            return model, metadata
    raise TypeError("A model loader must return a module or (module, metadata).")


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _checkpoint_sha256(
    entry: Mapping[str, object], metadata: Mapping[str, object]
) -> str | None:
    direct = entry.get("checkpoint_sha256")
    nested = entry.get("checkpoint")
    declared = (
        direct
        if direct is not None
        else nested.get("sha256")
        if isinstance(nested, Mapping)
        else None
    )
    measured = metadata.get("checkpoint_sha256", metadata.get("sha256"))
    if declared is not None and measured is not None and str(declared) != str(measured):
        raise ValueError("Declared checkpoint checksum does not match the loaded file.")
    value = measured if measured is not None else declared
    return None if value is None else str(value)


def _config_hash(
    entry: Mapping[str, object], metadata: Mapping[str, object]
) -> str | None:
    declared = entry.get("config_hash")
    measured = metadata.get("config_hash")
    if declared is not None and measured is not None and str(declared) != str(measured):
        raise ValueError("Manifest and checkpoint configuration hashes differ.")
    value = measured if measured is not None else declared
    return None if value is None else str(value)


def benchmark_candidate(
    entry: Mapping[str, object],
    *,
    config: BenchmarkConfig,
    model_loader: ModelLoader,
    input_factory: InputFactory | None = None,
) -> dict[str, object]:
    """Benchmark one candidate with batch-one, inference-only CPU execution."""

    config.validate()
    candidate_id = _candidate_id(entry)
    model, metadata = _unwrap_loaded(model_loader(entry))
    model = model.to(device="cpu", dtype=torch.float32)
    model.eval()
    inputs = tuple((input_factory or _default_inputs)(entry, config))
    if not inputs:
        raise ValueError("The input factory returned no inputs.")
    for item in inputs:
        if not isinstance(item, torch.Tensor):
            raise TypeError("Benchmark inputs must be torch tensors.")
        if item.device.type != "cpu" or item.dtype != torch.float32:
            raise ValueError("Benchmark inputs must be float32 CPU tensors.")
        if item.ndim != 4 or item.shape[0] != 1:
            raise ValueError("Every benchmark input must have batch size one.")

    with torch.inference_mode():
        for index in range(config.warmups):
            model(inputs[index % len(inputs)])

        elapsed_ms: list[float] = []
        for index in range(config.timed_samples):
            sample = inputs[index % len(inputs)]
            started = time.perf_counter_ns()
            model(sample)
            finished = time.perf_counter_ns()
            elapsed_ms.append((finished - started) / 1_000_000.0)

    if not elapsed_ms or not all(np.isfinite(elapsed_ms)):
        raise RuntimeError(f"Candidate {candidate_id!r} produced invalid timings.")
    return {
        "candidate_id": candidate_id,
        "role": str(_candidate_value(entry, "role", _candidate_value(entry, "kind", "unknown"))),
        "group": _candidate_value(entry, "group"),
        "architecture": _candidate_value(
            entry,
            "architecture",
            metadata.get("architecture", (
                metadata.get("spec", {}).get("architecture")
                if isinstance(metadata.get("spec"), Mapping)
                else None
            )),
        ),
        "checkpoint_path": _entry_path(entry, "checkpoint_path"),
        "checkpoint_sha256": _checkpoint_sha256(entry, metadata),
        "config_hash": _config_hash(entry, metadata),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "latency_ms": {
            "mean": float(statistics.fmean(elapsed_ms)),
            "median": float(statistics.median(elapsed_ms)),
            "std": float(np.std(np.asarray(elapsed_ms), ddof=0)),
            "p95": _percentile(elapsed_ms, 95.0),
            "p99": _percentile(elapsed_ms, 99.0),
            "min": float(min(elapsed_ms)),
            "max": float(max(elapsed_ms)),
        },
        "warmups": config.warmups,
        "timed_samples": config.timed_samples,
        "batch_size": 1,
        "input_resolution": list(inputs[0].shape[-2:]),
    }


def _cpu_model() -> str:
    value = platform.processor().strip()
    if value:
        return value
    if sys.platform == "linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.machine() or "unknown"


def environment_metadata(config: BenchmarkConfig) -> dict[str, object]:
    """Return runtime facts necessary to interpret latency measurements."""

    return {
        "device": "cpu",
        "cpu_model": _cpu_model(),
        "cpu_threads": config.cpu_threads,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "logical_cpu_count": os.cpu_count(),
        "precision": config.precision,
        "input_shape": [1, config.input_channels, config.input_height, config.input_width],
        "input_resolution": [config.input_height, config.input_width],
        "pytorch_version": torch.__version__,
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def benchmark_candidates(
    candidates: Sequence[Mapping[str, object]],
    *,
    config: BenchmarkConfig | None = None,
    model_loader: ModelLoader,
    input_factory: InputFactory | None = None,
    input_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Benchmark all entries under one fixed thread and input configuration."""

    actual_config = config or BenchmarkConfig()
    actual_config.validate()
    if not candidates:
        raise ValueError("At least one candidate is required.")
    ids = [_candidate_id(item) for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate ids must be unique.")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(actual_config.cpu_threads)
    try:
        environment = environment_metadata(actual_config)
        results = [
            benchmark_candidate(
                entry,
                config=actual_config,
                model_loader=model_loader,
                input_factory=input_factory,
            )
            for entry in candidates
        ]
    finally:
        torch.set_num_threads(previous_threads)
    provenance = dict(
        input_provenance
        or {
            "source": "synthetic_smoke",
            "pool_samples": min(
                actual_config.input_pool_size,
                max(1, actual_config.warmups + actual_config.timed_samples),
            ),
            "preprocessing": None,
            "preloaded_before_timing": True,
        }
    )
    return {
        "schema_version": LATENCY_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            **asdict(actual_config),
            "batch_size": 1,
            "measurement": "model_only",
            "clock": "perf_counter_ns",
            "input_preparation": provenance["source"],
            "input_preparation_in_timing": False,
        },
        "input": provenance,
        "environment": environment,
        "candidates": results,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def benchmark_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    config: BenchmarkConfig | None = None,
    candidate_ids: Sequence[str] | None = None,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    split_npz: str | Path = DEFAULT_SPLIT_NPZ,
    synthetic_inputs: bool = False,
    include_detector: bool = False,
) -> dict[str, object]:
    """Load candidates from a training manifest, benchmark, and write JSON."""

    source = Path(manifest_path).resolve()
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("The training manifest must contain a JSON object.")
    if manifest.get("schema_version") != "cifar100-training-manifest/v1":
        raise ValueError("Unsupported CIFAR-100 training manifest schema.")
    dataset_id = manifest.get("dataset_id")
    if dataset_id is not None and str(dataset_id) != CIFAR100_PROFILE.dataset_id:
        raise ValueError("Training manifest belongs to another dataset.")
    profile_fingerprint = manifest.get("profile_fingerprint")
    if (
        profile_fingerprint is not None
        and str(profile_fingerprint) != CIFAR100_PROFILE.fingerprint
    ):
        raise ValueError("Training manifest hierarchy fingerprint differs.")
    if manifest.get("official_test_used") is not False:
        raise ValueError("Training manifest does not affirm held-back data isolation.")
    entries = _manifest_candidates(manifest)
    if include_detector:
        detector = manifest.get("detector")
        if not isinstance(detector, Mapping):
            raise ValueError("Training manifest has no trained detector.")
        entries.append(dict(detector))
    entries = [
        item
        for item in entries
        if _candidate_value(item, "is_candidate", True) is not False
    ]
    selected = None if candidate_ids is None else set(candidate_ids)
    if selected is not None:
        entries = [item for item in entries if _candidate_id(item) in selected]
        missing = selected - {_candidate_id(item) for item in entries}
        if missing:
            raise ValueError(f"Unknown requested candidate ids: {sorted(missing)}")

    def loader(entry: Mapping[str, object]) -> object:
        return load_manifest_candidate(entry, manifest_dir=source.parent)

    actual_config = config or BenchmarkConfig()
    if synthetic_inputs:
        input_factory = None
        input_provenance: Mapping[str, object] = {
            "source": "synthetic_smoke",
            "pool_samples": min(
                actual_config.input_pool_size,
                max(1, actual_config.warmups + actual_config.timed_samples),
            ),
            "preprocessing": None,
            "preloaded_before_timing": True,
        }
    else:
        has_detector = any(
            _candidate_value(item, "architecture") == "convnextv2_large"
            for item in entries
        )
        has_standard = any(
            _candidate_value(item, "architecture") != "convnextv2_large"
            for item in entries
        )
        input_pools: dict[str, tuple[torch.Tensor, ...]] = {}
        provenances: dict[str, Mapping[str, object]] = {}
        if has_standard:
            input_pools["cifar32"], provenances["cifar32"] = (
                load_cascade_validation_inputs(
                    data_root,
                    split_npz,
                    config=actual_config,
                )
            )
        if has_detector:
            detector_config = replace(
                actual_config, input_height=224, input_width=224
            )
            input_pools["convnext224"], provenances["convnext224"] = (
                load_cascade_validation_inputs(
                    data_root,
                    split_npz,
                    config=detector_config,
                )
            )
        primary = next(iter(provenances.values()))
        input_provenance = {
            **dict(primary),
            "candidate_preprocessing": {
                key: value["preprocessing"] for key, value in provenances.items()
            },
        }

        def input_factory(
            entry: Mapping[str, object], _config: BenchmarkConfig
        ) -> Sequence[torch.Tensor]:
            key = (
                "convnext224"
                if _candidate_value(entry, "architecture") == "convnextv2_large"
                else "cifar32"
            )
            return input_pools[key]

    payload = benchmark_candidates(
        entries,
        config=actual_config,
        model_loader=loader,
        input_factory=input_factory,
        input_provenance=input_provenance,
    )
    payload["source_manifest"] = {
        "path": str(source),
        "sha256": _file_sha256(source),
        "schema_version": manifest.get("schema_version"),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="append", dest="candidate_ids")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--input-pool-size", type=int, default=500)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split-npz", type=Path, default=DEFAULT_SPLIT_NPZ)
    parser.add_argument(
        "--synthetic-inputs",
        action="store_true",
        help="Use deterministic synthetic inputs for smoke checks only.",
    )
    parser.add_argument(
        "--include-detector",
        action="store_true",
        help="Also benchmark the trained ConvNeXt V2-L endpoint.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        warmups=args.warmups,
        timed_samples=args.samples,
        cpu_threads=args.threads,
        seed=args.seed,
        input_pool_size=args.input_pool_size,
    )
    payload = benchmark_manifest(
        args.manifest,
        args.output,
        config=config,
        candidate_ids=args.candidate_ids,
        data_root=args.data_root,
        split_npz=args.split_npz,
        synthetic_inputs=args.synthetic_inputs,
        include_detector=args.include_detector,
    )
    print(f"Wrote {len(payload['candidates'])} candidate benchmarks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkConfig",
    "LATENCY_SCHEMA_VERSION",
    "benchmark_candidate",
    "benchmark_candidates",
    "benchmark_manifest",
    "environment_metadata",
    "load_cascade_validation_inputs",
    "load_manifest_candidate",
]
