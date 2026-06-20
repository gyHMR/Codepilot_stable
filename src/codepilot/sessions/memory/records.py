from __future__ import annotations

"""Structured memory records and shared value types."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


MemoryKind = Literal["task", "file", "failure", "decision", "project"]
MemoryScope = Literal["session", "project"]
MemoryTrust = Literal["observed", "verified", "user_given", "model_claim"]
MemoryStatus = Literal["active", "stale", "superseded", "deleted"]

MEMORY_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    id: str
    kind: MemoryKind
    scope: MemoryScope
    content: dict[str, Any]
    source: str
    source_run_id: str | None = None
    related_paths: list[str] = field(default_factory=list)
    source_hashes: dict[str, str] = field(default_factory=dict)
    trust: MemoryTrust = "observed"
    status: MemoryStatus = "active"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=str(value.get("id", "")),
            kind=_memory_kind(value.get("kind")),
            scope=_memory_scope(value.get("scope")),
            content=dict(value.get("content", {})) if isinstance(value.get("content"), dict) else {},
            source=str(value.get("source", "unknown")),
            source_run_id=value.get("source_run_id") if isinstance(value.get("source_run_id"), str) else None,
            related_paths=[str(item) for item in value.get("related_paths", []) if isinstance(item, str)],
            source_hashes={
                str(key): str(item)
                for key, item in value.get("source_hashes", {}).items()
            } if isinstance(value.get("source_hashes"), dict) else {},
            trust=_memory_trust(value.get("trust")),
            status=_memory_status(value.get("status")),
            created_at=str(value.get("created_at", utc_now_iso())),
            updated_at=str(value.get("updated_at", utc_now_iso())),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryQuery:
    text: str
    active_paths: list[str]
    limit: int = 8


@dataclass(frozen=True)
class RetrievedMemory:
    record: MemoryRecord
    score: int
    reasons: list[str]


def _memory_kind(value: object) -> MemoryKind:
    return value if value in {"task", "file", "failure", "decision", "project"} else "project"  # type: ignore[return-value]


def _memory_scope(value: object) -> MemoryScope:
    return value if value in {"session", "project"} else "session"  # type: ignore[return-value]


def _memory_trust(value: object) -> MemoryTrust:
    return value if value in {"observed", "verified", "user_given", "model_claim"} else "observed"  # type: ignore[return-value]


def _memory_status(value: object) -> MemoryStatus:
    return value if value in {"active", "stale", "superseded", "deleted"} else "active"  # type: ignore[return-value]


__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "MemoryKind",
    "MemoryQuery",
    "MemoryRecord",
    "MemoryScope",
    "MemoryStatus",
    "MemoryTrust",
    "RetrievedMemory",
    "utc_now_iso",
]
