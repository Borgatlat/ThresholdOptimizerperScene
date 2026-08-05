"""Dataset-neutral hierarchy metadata shared by collectors and optimizers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence


PROFILE_SCHEMA_VERSION = "cascade-profile/v1"


@dataclass(frozen=True)
class HierarchyProfile:
    """Describe one depth-one hierarchical classification problem.

    ``groups`` maps every routable superclass to its leaf classes. Router
    predictions use ``router_outputs`` order; outputs need not all have a
    specialized branch (for example an M3N-VC background output).
    """

    dataset_id: str
    global_classes: tuple[str, ...]
    groups: Mapping[str, tuple[str, ...]]
    router_outputs: tuple[str, ...]
    split_group_column: str | None = None
    max_router_depth: int = 1
    schema_version: str = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported profile schema: {self.schema_version!r}")
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty.")
        if not self.global_classes or len(set(self.global_classes)) != len(
            self.global_classes
        ):
            raise ValueError("global_classes must be non-empty and unique.")
        if len(set(self.router_outputs)) != len(self.router_outputs):
            raise ValueError("router_outputs must be unique.")
        if self.max_router_depth != 1:
            raise ValueError("Only depth-one intermediate routers are supported.")

        known_classes = set(self.global_classes)
        seen_classes: set[str] = set()
        for group, class_names in self.groups.items():
            if group not in self.router_outputs:
                raise ValueError(f"Group {group!r} is not a router output.")
            if not class_names:
                raise ValueError(f"Group {group!r} has no leaf classes.")
            unknown = set(class_names) - known_classes
            if unknown:
                raise ValueError(
                    f"Group {group!r} contains unknown classes: {sorted(unknown)}"
                )
            overlap = seen_classes & set(class_names)
            if overlap:
                raise ValueError(
                    "Depth-one groups must be disjoint; repeated classes: "
                    f"{sorted(overlap)}"
                )
            seen_classes.update(class_names)

    @property
    def group_ids(self) -> tuple[str, ...]:
        return tuple(self.groups)

    @property
    def global_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.global_classes)}

    @property
    def router_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.router_outputs)}

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "global_classes": list(self.global_classes),
            "groups": {
                group: list(class_names) for group, class_names in self.groups.items()
            },
            "router_outputs": list(self.router_outputs),
            "split_group_column": self.split_group_column,
            "max_router_depth": self.max_router_depth,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "HierarchyProfile":
        groups_value = value.get("groups", {})
        if not isinstance(groups_value, Mapping):
            raise ValueError("profile.groups must be an object.")
        return cls(
            dataset_id=str(value["dataset_id"]),
            global_classes=tuple(str(item) for item in value["global_classes"]),
            groups={
                str(group): tuple(str(item) for item in class_names)
                for group, class_names in groups_value.items()
                if isinstance(class_names, Sequence)
                and not isinstance(class_names, (str, bytes))
            },
            router_outputs=tuple(str(item) for item in value["router_outputs"]),
            split_group_column=(
                None
                if value.get("split_group_column") is None
                else str(value["split_group_column"])
            ),
            max_router_depth=int(value.get("max_router_depth", 1)),
            schema_version=str(
                value.get("schema_version", PROFILE_SCHEMA_VERSION)
            ),
        )

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def profile_from_payload(payload: Mapping[str, object]) -> HierarchyProfile:
    value = payload.get("profile")
    if not isinstance(value, Mapping):
        raise ValueError(
            "Empirical outcomes have no hierarchy profile. Regenerate them or "
            "provide a cascade-profile/v1 sidecar JSON file."
        )
    return HierarchyProfile.from_dict(value)
