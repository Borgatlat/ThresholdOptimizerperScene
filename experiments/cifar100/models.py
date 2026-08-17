"""Independent CIFAR-100 model definitions and checkpoint utilities.

The Wide ResNets use the canonical CIFAR architecture: a 3x3 stride-one
stem, three residual groups, and ``depth = 6n + 4``.  ResNet-18 also uses a
CIFAR stem (3x3 stride one, with no max-pool).  Official torchvision
ImageNet weights are optional for ResNet-18 and are never requested unless
``pretrained=True`` is passed explicitly.

Every call to :func:`build_model` creates a new end-to-end module.  The
registry describes twenty independent specialists, two intermediate
routers, and two global classifiers.  Its WRN-16-2 base initializer is
deliberately kept outside the deployable candidate tuple.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


CHECKPOINT_SCHEMA_VERSION = "cifar100-model-checkpoint/v1"
CONVNEXT_V2_LARGE_PRETRAINED_MODEL = "convnextv2_large.fcmae_ft_in1k"
CONVNEXT_V2_LARGE_ORIGIN = "https://github.com/facebookresearch/ConvNeXt-V2"


class WideBasicBlock(nn.Module):
    """Pre-activation residual block used by the original Wide ResNet."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.equal_channels = in_channels == out_channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.dropout_rate = float(dropout_rate)
        self.shortcut = (
            None
            if self.equal_channels and stride == 1
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        activated = self.relu1(self.bn1(inputs))
        residual = self.conv1(activated)
        residual = self.relu2(self.bn2(residual))
        if self.dropout_rate:
            residual = F.dropout(
                residual,
                p=self.dropout_rate,
                training=self.training,
            )
        residual = self.conv2(residual)
        shortcut = inputs if self.shortcut is None else self.shortcut(activated)
        return shortcut + residual


class WideResidualGroup(nn.Sequential):
    def __init__(
        self,
        block_count: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_rate: float,
    ) -> None:
        blocks = [
            WideBasicBlock(
                in_channels,
                out_channels,
                stride,
                dropout_rate,
            )
        ]
        blocks.extend(
            WideBasicBlock(
                out_channels,
                out_channels,
                1,
                dropout_rate,
            )
            for _ in range(1, block_count)
        )
        super().__init__(*blocks)


class WideResNet(nn.Module):
    """Wide ResNet for 32x32 RGB images."""

    def __init__(
        self,
        depth: int,
        widen_factor: int,
        num_classes: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if (depth - 4) % 6:
            raise ValueError("Wide ResNet depth must satisfy depth = 6n + 4.")
        if depth < 10 or widen_factor < 1 or num_classes < 1:
            raise ValueError("Invalid Wide ResNet dimensions.")
        if not 0.0 <= float(dropout_rate) < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")

        blocks_per_group = (depth - 4) // 6
        widths = (16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor)
        self.depth = int(depth)
        self.widen_factor = int(widen_factor)
        self.num_classes = int(num_classes)
        self.dropout_rate = float(dropout_rate)

        self.conv1 = nn.Conv2d(
            3, widths[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.block1 = WideResidualGroup(
            blocks_per_group,
            widths[0],
            widths[1],
            1,
            dropout_rate,
        )
        self.block2 = WideResidualGroup(
            blocks_per_group,
            widths[1],
            widths[2],
            2,
            dropout_rate,
        )
        self.block3 = WideResidualGroup(
            blocks_per_group,
            widths[2],
            widths[3],
            2,
            dropout_rate,
        )
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        features = self.conv1(inputs)
        features = self.block1(features)
        features = self.block2(features)
        features = self.block3(features)
        features = self.relu(self.bn(features))
        features = self.pool(features).flatten(1)
        return self.fc(features)


def build_wrn_16_2(
    num_classes: int,
    *,
    dropout_rate: float = 0.0,
) -> WideResNet:
    return WideResNet(16, 2, num_classes, dropout_rate)


def build_wrn_28_10(
    num_classes: int,
    *,
    dropout_rate: float = 0.0,
) -> WideResNet:
    return WideResNet(28, 10, num_classes, dropout_rate)


def build_resnet18(num_classes: int, *, pretrained: bool = False) -> nn.Module:
    """Build a torchvision ResNet-18 with a CIFAR-compatible stem.

    With ``pretrained=True``, torchvision's official ImageNet weights are
    requested explicitly.  The center 3x3 region of the pretrained 7x7 stem
    initializes the CIFAR stem; all residual-layer weights are retained.
    Calling with the default ``False`` cannot trigger a weights download.
    """

    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:  # pragma: no cover - exercised in lean installs
        raise RuntimeError(
            "torchvision is required to build ResNet-18 candidates."
        ) from exc

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    original_stem = model.conv1
    cifar_stem = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    if pretrained:
        with torch.no_grad():
            cifar_stem.weight.copy_(original_stem.weight[:, :, 2:5, 2:5])
    else:
        nn.init.kaiming_normal_(
            cifar_stem.weight, mode="fan_out", nonlinearity="relu"
        )
    model.conv1 = cifar_stem
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, int(num_classes))
    model.num_classes = int(num_classes)
    model.initialization_provenance = (
        "torchvision.ResNet18_Weights.DEFAULT"
        if pretrained
        else "random"
    )
    return model


@dataclass(frozen=True)
class CandidateSpec:
    """Serializable definition of one independently executed model."""

    candidate_id: str
    role: str
    kind: str
    architecture: str
    output_labels: tuple[str, ...]
    group: str | None = None
    is_candidate: bool = True

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty.")
        if self.role not in {
            "specialized",
            "intermediate",
            "global",
            "detector",
            "base_initializer",
        }:
            raise ValueError(f"Unknown role: {self.role!r}")
        if self.kind not in {
            "specialized",
            "identifier",
            "global",
            "detector",
            "initializer",
        }:
            raise ValueError(f"Unknown optimizer kind: {self.kind!r}")
        if self.architecture not in {
            "wrn16_2",
            "wrn28_10",
            "resnet18",
            "convnextv2_large",
        }:
            raise ValueError(f"Unknown architecture: {self.architecture!r}")
        if not self.output_labels or len(set(self.output_labels)) != len(
            self.output_labels
        ):
            raise ValueError("output_labels must be non-empty and unique.")
        if (self.role == "specialized") != (self.group is not None):
            raise ValueError("Exactly specialized specs must declare a group.")
        if self.role == "intermediate" and self.kind != "identifier":
            raise ValueError("Intermediate specs must use optimizer kind 'identifier'.")
        if self.role == "detector" and self.kind != "detector":
            raise ValueError("Detector specs must use optimizer kind 'detector'.")
        if self.role == "base_initializer" and (
            self.kind != "initializer" or self.is_candidate
        ):
            raise ValueError("The base initializer is not a deployable candidate.")

    @property
    def id(self) -> str:
        """Compatibility alias used by candidate-manifest writers."""

        return self.candidate_id

    @property
    def num_classes(self) -> int:
        return len(self.output_labels)

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "role": self.role,
            "kind": self.kind,
            "architecture": self.architecture,
            "output_labels": list(self.output_labels),
            "group": self.group,
            "is_candidate": self.is_candidate,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "CandidateSpec":
        return cls(
            candidate_id=str(value["candidate_id"]),
            role=str(value["role"]),
            kind=str(value["kind"]),
            architecture=str(value["architecture"]),
            output_labels=tuple(str(item) for item in value["output_labels"]),
            group=None if value.get("group") is None else str(value["group"]),
            is_candidate=bool(value.get("is_candidate", True)),
        )


@dataclass(frozen=True)
class CandidateRegistry:
    candidates: tuple[CandidateSpec, ...]
    wrn_base_initializer: CandidateSpec
    detector: CandidateSpec

    def __post_init__(self) -> None:
        if len(self.candidates) != 24:
            raise ValueError("The CIFAR-100 registry must contain 24 candidates.")
        identifiers = [item.candidate_id for item in self.all_specs]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Registry model ids must be unique.")
        if any(not item.is_candidate for item in self.candidates):
            raise ValueError("All entries in candidates must be deployable.")
        if self.wrn_base_initializer.is_candidate:
            raise ValueError("The WRN base initializer cannot be a candidate.")
        if self.detector.role != "detector" or self.detector.kind != "detector":
            raise ValueError("The registry detector must be a deterministic endpoint.")

    @property
    def all_specs(self) -> tuple[CandidateSpec, ...]:
        return (*self.candidates, self.wrn_base_initializer, self.detector)

    @property
    def by_id(self) -> dict[str, CandidateSpec]:
        return {item.candidate_id: item for item in self.all_specs}


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug:
        raise ValueError(f"Cannot make a candidate id from group {value!r}.")
    return slug


def candidate_specs(
    groups: Mapping[str, Sequence[str]],
    global_labels: Sequence[str] | None = None,
) -> CandidateRegistry:
    """Create the fixed CIFAR-100 candidate registry from official labels.

    Production callers should pass the ordered official coarse-to-fine map
    and official fine-label order.  ``global_labels=None`` is convenient for
    isolated tests and flattens the supplied groups in insertion order.
    """

    ordered_groups = {
        str(group): tuple(str(label) for label in labels)
        for group, labels in groups.items()
    }
    if len(ordered_groups) != 20:
        raise ValueError("CIFAR-100 must provide exactly 20 coarse groups.")
    if any(len(labels) != 5 for labels in ordered_groups.values()):
        raise ValueError("Every CIFAR-100 coarse group must contain five labels.")
    grouped_labels = tuple(
        label for labels in ordered_groups.values() for label in labels
    )
    if len(grouped_labels) != 100 or len(set(grouped_labels)) != 100:
        raise ValueError("The coarse groups must partition 100 unique fine labels.")

    resolved_global_labels = (
        grouped_labels
        if global_labels is None
        else tuple(str(label) for label in global_labels)
    )
    if (
        len(resolved_global_labels) != 100
        or len(set(resolved_global_labels)) != 100
        or set(resolved_global_labels) != set(grouped_labels)
    ):
        raise ValueError(
            "global_labels must contain the same 100 unique labels as groups."
        )

    group_names = tuple(ordered_groups)
    slugs = tuple(_slug(group) for group in group_names)
    if len(slugs) != len(set(slugs)):
        raise ValueError("Coarse group names produce colliding candidate ids.")

    specialists = tuple(
        CandidateSpec(
            candidate_id=f"wrn16_2_specialist_{slug}",
            role="specialized",
            kind="specialized",
            architecture="wrn16_2",
            output_labels=ordered_groups[group],
            group=group,
        )
        for group, slug in zip(group_names, slugs)
    )
    intermediates = (
        CandidateSpec(
            candidate_id="wrn16_2_coarse",
            role="intermediate",
            kind="identifier",
            architecture="wrn16_2",
            output_labels=group_names,
        ),
        CandidateSpec(
            candidate_id="resnet18_coarse",
            role="intermediate",
            kind="identifier",
            architecture="resnet18",
            output_labels=group_names,
        ),
    )
    globals_ = (
        CandidateSpec(
            candidate_id="resnet18_global",
            role="global",
            kind="global",
            architecture="resnet18",
            output_labels=resolved_global_labels,
        ),
        CandidateSpec(
            candidate_id="wrn28_10_global",
            role="global",
            kind="global",
            architecture="wrn28_10",
            output_labels=resolved_global_labels,
        ),
    )
    base = CandidateSpec(
        candidate_id="wrn16_2_base",
        role="base_initializer",
        kind="initializer",
        architecture="wrn16_2",
        output_labels=resolved_global_labels,
        is_candidate=False,
    )
    detector = CandidateSpec(
        candidate_id="convnextv2_large_detector",
        role="detector",
        kind="detector",
        architecture="convnextv2_large",
        output_labels=resolved_global_labels,
    )
    return CandidateRegistry(
        (*specialists, *intermediates, *globals_), base, detector
    )


def build_model(spec: CandidateSpec, *, pretrained: bool = False) -> nn.Module:
    """Build a fresh model for ``spec``; no parameters are shared."""

    if spec.architecture == "wrn16_2":
        if pretrained:
            raise ValueError(
                "WRN-16-2 has no implicit external weights; initialize it "
                "with transfer_wrn_base_features instead."
            )
        return build_wrn_16_2(spec.num_classes)
    if spec.architecture == "wrn28_10":
        if pretrained:
            raise ValueError("WRN-28-10 has no configured external weights.")
        return build_wrn_28_10(spec.num_classes)
    if spec.architecture == "convnextv2_large":
        try:
            import timm
        except ImportError as exc:  # pragma: no cover - lean installations
            raise RuntimeError(
                "timm is required for the ConvNeXt V2-L detector."
            ) from exc
        model = timm.create_model(
            CONVNEXT_V2_LARGE_PRETRAINED_MODEL,
            pretrained=pretrained,
            num_classes=spec.num_classes,
        )
        model.num_classes = spec.num_classes
        model.initialization_provenance = (
            CONVNEXT_V2_LARGE_PRETRAINED_MODEL if pretrained else "local_checkpoint"
        )
        return model
    return build_resnet18(spec.num_classes, pretrained=pretrained)


def _json_compatible(value: Any) -> Any:
    if is_dataclass(value):
        return _json_compatible(asdict(value))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Configuration value {value!r} is not JSON serializable.")


def config_hash(config: Mapping[str, object] | CandidateSpec) -> str:
    canonical = json.dumps(
        _json_compatible(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    spec: CandidateSpec,
    training_config: Mapping[str, object],
    *,
    metrics: Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None,
) -> Path:
    """Save a portable state-dict checkpoint with reproducibility metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable_config = _json_compatible(training_config)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "spec": spec.as_dict(),
        "training_config": serializable_config,
        "config_hash": config_hash(training_config),
        "metrics": _json_compatible(metrics or {}),
        "extra": _json_compatible(extra or {}),
        "pytorch_version": str(torch.__version__),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
    }
    torch.save(payload, destination)
    return destination


def _load_checkpoint_payload(
    path: str | Path,
    map_location: str | torch.device,
) -> dict[str, object]:
    source = Path(path)
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older PyTorch
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint {source} is not a mapping.")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema: {payload.get('schema_version')!r}"
        )
    if not isinstance(payload.get("spec"), Mapping):
        raise ValueError("Checkpoint has no valid model spec.")
    if not isinstance(payload.get("model_state_dict"), Mapping):
        raise ValueError("Checkpoint has no model_state_dict.")
    training_config = payload.get("training_config")
    if not isinstance(training_config, Mapping):
        raise ValueError("Checkpoint has no valid training configuration.")
    if payload.get("config_hash") != config_hash(training_config):
        raise ValueError("Checkpoint training configuration hash does not match.")
    return payload


def load_checkpoint_metadata(
    path: str | Path,
    *,
    expected_spec: CandidateSpec | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    """Validate a checkpoint without constructing its model architecture."""

    payload = _load_checkpoint_payload(path, map_location)
    spec = CandidateSpec.from_dict(payload["spec"])
    if expected_spec is not None and spec != expected_spec:
        raise ValueError(
            f"Checkpoint spec {spec.candidate_id!r} does not match expected "
            f"{expected_spec.candidate_id!r}."
        )
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    metadata["checkpoint_sha256"] = file_sha256(path)
    return metadata


def load_checkpoint(
    path: str | Path,
    *,
    expected_spec: CandidateSpec | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, object]]:
    """Reconstruct a model from a validated local checkpoint."""

    payload = _load_checkpoint_payload(path, map_location)
    spec = CandidateSpec.from_dict(payload["spec"])
    if expected_spec is not None and spec != expected_spec:
        raise ValueError(
            f"Checkpoint spec {spec.candidate_id!r} does not match expected "
            f"{expected_spec.candidate_id!r}."
        )
    model = build_model(spec, pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    metadata = {key: value for key, value in payload.items() if key != "model_state_dict"}
    metadata["checkpoint_sha256"] = file_sha256(path)
    return model, metadata


@dataclass(frozen=True)
class TransferReport:
    copied_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]


def transfer_wrn_base_features(
    source: nn.Module | str | Path,
    target_model: nn.Module,
    *,
    map_location: str | torch.device = "cpu",
) -> TransferReport:
    """Copy a WRN-16-2 backbone without copying its task-specific head."""

    if not isinstance(target_model, WideResNet) or (
        target_model.depth,
        target_model.widen_factor,
    ) != (16, 2):
        raise ValueError("target_model must be a WRN-16-2 instance.")

    if isinstance(source, nn.Module):
        if not isinstance(source, WideResNet) or (
            source.depth,
            source.widen_factor,
        ) != (16, 2):
            raise ValueError("source model must be a WRN-16-2 instance.")
        source_state = source.state_dict()
    else:
        payload = _load_checkpoint_payload(source, map_location)
        source_spec = CandidateSpec.from_dict(payload["spec"])
        if source_spec.architecture != "wrn16_2":
            raise ValueError("Source checkpoint must contain a WRN-16-2 model.")
        source_state = payload["model_state_dict"]

    target_state = target_model.state_dict()
    copied: list[str] = []
    skipped: list[str] = []
    for key, value in source_state.items():
        if key.startswith("fc.") or key not in target_state:
            skipped.append(key)
            continue
        if target_state[key].shape != value.shape:
            skipped.append(key)
            continue
        target_state[key] = value.detach().to(
            device=target_state[key].device,
            dtype=target_state[key].dtype,
        ).clone()
        copied.append(key)

    target_model.load_state_dict(target_state, strict=True)
    if not copied:
        raise ValueError("No compatible WRN-16-2 backbone parameters were copied.")
    return TransferReport(tuple(copied), tuple(skipped))
