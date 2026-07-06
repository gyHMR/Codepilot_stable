from __future__ import annotations

# 新手导读：records.py 定义 MemoryRecord、RetrievedMemory 等记忆数据结构。
# 关注点：MemoryRecord v3 只暴露结构化 subject/predicate/value；旧字段只在读取边界转换。

"""Structured durable memory data contracts."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, cast


MEMORY_SCHEMA_VERSION = 3

MemoryType = str
MemoryScope = str
MemoryStatus = Literal["candidate", "active", "disabled", "superseded", "deleted"]
MemorySource = str
MemoryConfidence = Literal["explicit", "observed", "inferred"]

_MEMORY_STATUSES = frozenset(
    {"candidate", "active", "disabled", "superseded", "deleted"}
)
_MEMORY_CONFIDENCE = frozenset({"explicit", "observed", "inferred"})


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(init=False)
class MemoryRecord:
    """A structured long-term memory record suitable for future recall."""

    id: str
    type: MemoryType
    scope: MemoryScope
    subject: str
    predicate: str
    value: str
    content: str
    keywords: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    status: MemoryStatus = "active"
    source: MemorySource = "task_summary"
    confidence: MemoryConfidence = "inferred"
    priority: int = 1
    created_by_session_id: str | None = None
    created_by_run_id: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    occurrences: int = 1
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def __init__(
        self,
        *,
        id: str,
        type: str,
        scope: str = "project",
        subject: str,
        predicate: str,
        value: str,
        content: str,
        keywords: list[object] | None = None,
        paths: list[object] | None = None,
        status: str = "active",
        source: str,
        confidence: str = "inferred",
        priority: int = 1,
        created_by_session_id: str | None = None,
        created_by_run_id: str | None = None,
        evidence_refs: list[object] | None = None,
        supersedes: list[object] | None = None,
        occurrences: int = 1,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        self.id = _require_text(id, "memory id")
        self.type = _require_text(type, "memory type")
        self.scope = _require_text(scope, "memory scope")
        self.subject = _require_text(subject, "memory subject")
        self.predicate = _require_text(predicate, "memory predicate")
        self.value = _require_text(value, "memory value")
        self.content = _require_text(content, "memory content")
        self.keywords = _dedupe_text(keywords or [])
        self.paths = _dedupe_text(paths or [])
        self.status = _ensure_memory_status(status)
        self.source = _require_text(source, "memory source")
        self.confidence = _ensure_memory_confidence(confidence)
        self.priority = _non_negative_int(priority, field_name="memory priority")
        self.created_by_session_id = _optional_text(created_by_session_id)
        self.created_by_run_id = _optional_text(created_by_run_id)
        self.evidence_refs = _dedupe_text(evidence_refs or [])
        self.supersedes = _dedupe_text(supersedes or [])
        self.occurrences = _positive_int(occurrences)
        self.created_at = str(created_at or utc_now_iso())
        self.updated_at = str(updated_at or utc_now_iso())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryRecord":
        schema = value.get("schema_version")
        if schema != MEMORY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported memory schema_version: {schema}")
        return cls(
            id=str(value.get("id", "")),
            type=str(value.get("type", "")),
            scope=str(value.get("scope", "")),
            subject=str(value.get("subject", "")),
            predicate=str(value.get("predicate", "")),
            value=str(value.get("value", "")),
            content=str(value.get("content", "")),
            keywords=_strings(value.get("keywords")),
            paths=_strings(value.get("paths")),
            status=_ensure_memory_status(value.get("status", "active")),
            source=str(value.get("source", "")),
            confidence=_ensure_memory_confidence(value.get("confidence", "inferred")),
            priority=_non_negative_int(
                value.get("priority", 1),
                field_name="memory priority",
            ),
            created_by_session_id=_optional_text(value.get("created_by_session_id")),
            created_by_run_id=_optional_text(value.get("created_by_run_id")),
            evidence_refs=_strings(value.get("evidence_refs")),
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


def normalize_memory_record_payload(raw: object) -> dict[str, Any]:
    """Normalize persisted memory JSON into the canonical v3 payload."""

    if not isinstance(raw, dict):
        raise TypeError("memory record payload must be a mapping")
    memory_type = _memory_type(raw.get("type") or raw.get("kind"))
    content = _memory_content(raw)
    subject = _optional_text(raw.get("subject") or raw.get("key")) or _subject_from_content(content)
    predicate = _optional_text(raw.get("predicate")) or "is"
    value = _optional_text(raw.get("value")) or content
    source = _optional_text(raw.get("source")) or _default_source(memory_type)
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "id": str(raw.get("id", "")),
        "type": memory_type,
        "scope": str(raw.get("scope", "project")),
        "subject": subject,
        "predicate": predicate,
        "value": value,
        "content": content,
        "keywords": _strings(
            raw.get("keywords")
            if isinstance(raw.get("keywords"), list)
            else raw.get("triggers")
        ),
        "paths": _strings(
            raw.get("paths")
            if isinstance(raw.get("paths"), list)
            else raw.get("related_paths")
        ),
        "status": _ensure_memory_status(raw.get("status", "active")),
        "source": source,
        "confidence": _ensure_memory_confidence(raw.get("confidence", "inferred")),
        "priority": _non_negative_int(
            raw.get("priority", 1),
            field_name="memory priority",
        ),
        "created_by_session_id": _optional_text(raw.get("created_by_session_id")),
        "created_by_run_id": _optional_text(raw.get("created_by_run_id")),
        "evidence_refs": _strings(raw.get("evidence_refs")),
        "supersedes": _strings(raw.get("supersedes")),
        "occurrences": _positive_int(raw.get("occurrences", 1)),
        "created_at": str(raw.get("created_at", utc_now_iso())),
        "updated_at": str(raw.get("updated_at", utc_now_iso())),
    }


@dataclass(frozen=True)
class MemoryQuery:
    """Query signals used to recall durable memory."""

    text: str
    active_paths: list[str]
    limit: int = 5
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


def _ensure_memory_status(value: object) -> MemoryStatus:
    if value not in _MEMORY_STATUSES:
        raise ValueError(f"Unknown memory status: {value}")
    return cast(MemoryStatus, value)


def _ensure_memory_confidence(value: object) -> MemoryConfidence:
    if value not in _MEMORY_CONFIDENCE:
        raise ValueError(f"Unknown memory confidence: {value}")
    return cast(MemoryConfidence, value)


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


def _memory_type(value: object) -> str:
    text = _optional_text(value)
    return text or "fact"


def _memory_content(raw: dict[str, Any]) -> str:
    for key in ("content", "text", "value"):
        text = _optional_text(raw.get(key))
        if text:
            return text
    return ""


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


def _non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be non-negative")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be non-negative") from exc
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _legacy_kind(memory_type: str) -> str:
    if memory_type in {"correction", "constraint", "decision", "experience"}:
        return memory_type
    lowered = memory_type.lower()
    if "correction" in lowered:
        return "correction"
    if "decision" in lowered:
        return "decision"
    if "experience" in lowered or "workflow" in lowered:
        return "experience"
    return "constraint"


def _default_source(memory_type: str) -> str:
    if _legacy_kind(memory_type) == "correction":
        return "user_correction"
    return "task_summary"


def _subject_from_content(content: str) -> str:
    compact = "_".join(content.lower().split())[:80]
    return compact or "memory"


__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "MemoryConfidence",
    "MemoryQuery",
    "MemoryRecall",
    "MemoryRecord",
    "MemoryScope",
    "MemorySource",
    "MemoryStatus",
    "MemoryType",
    "RetrievedMemory",
    "normalize_memory_record_payload",
    "utc_now_iso",
]
