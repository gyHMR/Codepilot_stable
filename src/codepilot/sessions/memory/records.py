from __future__ import annotations

# 新手导读：records.py 定义 MemoryRecord、RetrievedMemory 等记忆数据结构。
# 关注点：读这里能快速理解一条记忆有哪些可信度、scope 和状态字段。

"""Memory v2 data contracts.

Memory stores durable, reusable knowledge only.  Task progress, file
freshness, tool logs, and transient failures live in neighboring session,
context, and run stores.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast


MEMORY_SCHEMA_VERSION = 2

MemoryKind = Literal["correction", "constraint", "decision", "experience"]
MemoryScope = Literal["session", "project"]
MemoryStatus = Literal["active", "superseded", "deleted"]
MemorySource = Literal["user", "command", "run", "promoted"]

_MEMORY_KINDS = frozenset({"correction", "constraint", "decision", "experience"})
_MEMORY_SCOPES = frozenset({"session", "project"})
_MEMORY_STATUSES = frozenset({"active", "superseded", "deleted"})
_MEMORY_SOURCES = frozenset({"user", "command", "run", "promoted"})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MemoryRecord:
    """A durable memory record suitable for future recall."""

    id: str
    scope: MemoryScope
    kind: MemoryKind
    key: str
    text: str
    source: MemorySource
    status: MemoryStatus = "active"
    triggers: list[str] = field(default_factory=list)
    related_paths: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    occurrences: int = 1
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", _ensure_memory_scope(self.scope))
        object.__setattr__(self, "kind", _ensure_memory_kind(self.kind))
        object.__setattr__(self, "status", _ensure_memory_status(self.status))
        object.__setattr__(self, "source", _ensure_memory_source(self.source))
        self.id = _require_text(self.id, "memory id")
        self.key = _require_text(self.key, "memory key")
        self.text = _require_text(self.text, "memory text")
        self.triggers = _dedupe_text(self.triggers)
        self.related_paths = _dedupe_text(self.related_paths)
        self.evidence_refs = _dedupe_text(self.evidence_refs)
        self.supersedes = _dedupe_text(self.supersedes)
        if self.occurrences <= 0:
            raise ValueError("memory occurrences must be positive")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryRecord":
        schema = value.get("schema_version")
        if schema != MEMORY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported memory schema_version: {schema}")
        return cls(
            id=str(value.get("id", "")),
            scope=_ensure_memory_scope(value.get("scope")),
            kind=_ensure_memory_kind(value.get("kind")),
            status=_ensure_memory_status(value.get("status", "active")),
            key=str(value.get("key", "")),
            text=str(value.get("text", "")),
            triggers=_strings(value.get("triggers")),
            related_paths=_strings(value.get("related_paths")),
            evidence_refs=_strings(value.get("evidence_refs")),
            source=_ensure_memory_source(value.get("source")),
            supersedes=_strings(value.get("supersedes")),
            occurrences=_positive_int(value.get("occurrences", 1)),
            created_at=str(value.get("created_at", utc_now_iso())),
            updated_at=str(value.get("updated_at", utc_now_iso())),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = MEMORY_SCHEMA_VERSION
        return payload

    @property
    def is_retrievable(self) -> bool:
        return self.status == "active"

    def retrieval_exclusion_reason(self) -> str | None:
        if self.status != "active":
            return f"status:{self.status}"
        return None


@dataclass(frozen=True)
class MemoryQuery:
    """Query signals used to recall durable memory."""

    text: str
    active_paths: list[str]
    limit: int = 8
    task_phase: str | None = None
    action_intent: str | None = None
    recent_error: str | None = None
    retrieval_mode: str | None = None


@dataclass(frozen=True)
class RetrievedMemory:
    record: MemoryRecord
    score: int
    reasons: list[str]


@dataclass(frozen=True)
class MemoryRecall:
    """Layered memory recall result consumed by ContextGovernor."""

    pinned_text: str = ""
    always: list[RetrievedMemory] = field(default_factory=list)
    selected: list[RetrievedMemory] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)

    @property
    def retrieved(self) -> list[RetrievedMemory]:
        return [*self.always, *self.selected]


def _ensure_memory_kind(value: object) -> MemoryKind:
    if value not in _MEMORY_KINDS:
        raise ValueError(f"Unknown memory kind: {value}")
    return cast(MemoryKind, value)


def _ensure_memory_scope(value: object) -> MemoryScope:
    if value not in _MEMORY_SCOPES:
        raise ValueError(f"Unknown memory scope: {value}")
    return cast(MemoryScope, value)


def _ensure_memory_status(value: object) -> MemoryStatus:
    if value not in _MEMORY_STATUSES:
        raise ValueError(f"Unknown memory status: {value}")
    return cast(MemoryStatus, value)


def _ensure_memory_source(value: object) -> MemorySource:
    if value not in _MEMORY_SOURCES:
        raise ValueError(f"Unknown memory source: {value}")
    return cast(MemorySource, value)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe_text(value)


def _dedupe_text(values: list[object]) -> list[str]:
    out: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text and text not in out:
            out.append(text)
    return out


def _positive_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("memory occurrences must be positive")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("memory occurrences must be positive") from exc
    if number <= 0:
        raise ValueError("memory occurrences must be positive")
    return number


__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "MemoryKind",
    "MemoryQuery",
    "MemoryRecall",
    "MemoryRecord",
    "MemoryScope",
    "MemorySource",
    "MemoryStatus",
    "RetrievedMemory",
    "utc_now_iso",
]
