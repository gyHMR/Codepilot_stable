from __future__ import annotations

# 新手导读：records.py 定义 MemoryRecord、MemoryQuery 和召回结果。
# 关注点：MemoryRecord v4 是唯一长期记忆 schema，不做旧字段兼容。

"""Canonical durable memory data contracts."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, cast


MEMORY_SCHEMA_VERSION = 4

MemoryType = Literal[
    "preference",
    "constraint",
    "decision",
    "workflow",
    "correction",
    "experience",
]
MemoryScope = Literal["project", "workspace", "global"]
MemoryStatus = Literal["candidate", "active", "disabled", "superseded", "deleted"]
MemorySource = Literal[
    "user_explicit",
    "user_correction",
    "task_experience",
    "user_approved",
    "manual_edit",
]
MemoryConfidence = Literal["explicit", "observed", "inferred"]

_MEMORY_TYPES = frozenset(
    {"preference", "constraint", "decision", "workflow", "correction", "experience"}
)
_MEMORY_SCOPES = frozenset({"project", "workspace", "global"})
_MEMORY_STATUSES = frozenset(
    {"candidate", "active", "disabled", "superseded", "deleted"}
)
_MEMORY_SOURCES = frozenset(
    {"user_explicit", "user_correction", "task_experience", "user_approved", "manual_edit"}
)
_MEMORY_CONFIDENCE = frozenset({"explicit", "observed", "inferred"})
_MEMORY_RECORD_KEYS = frozenset(
    {
        "schema_version",
        "id",
        "type",
        "scope",
        "subject",
        "predicate",
        "value",
        "content",
        "keywords",
        "paths",
        "status",
        "source",
        "confidence",
        "priority",
        "created_by_session_id",
        "created_by_run_id",
        "source_message_id",
        "source_event_id",
        "evidence_refs",
        "supersedes",
        "superseded_by",
        "occurrences",
        "created_at",
        "updated_at",
    }
)
_LEGACY_MEMORY_KEYS = frozenset(
    {"k" + "ind", "k" + "ey", "t" + "ext", "tri" + "ggers", "related" + "_paths"}
)
_EVIDENCE_REF_PREFIXES = (
    "message:",
    "event:",
    "run:",
    "tool:",
    "verification:",
    "artifact:",
    "session:",
)
_MAX_MEMORY_CONTENT_CHARS = 1600


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
    source: MemorySource = "user_explicit"
    confidence: MemoryConfidence = "inferred"
    priority: int = 1
    created_by_session_id: str | None = None
    created_by_run_id: str | None = None
    source_message_id: str | None = None
    source_event_id: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    superseded_by: str | None = None
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
        source_message_id: str | None = None,
        source_event_id: str | None = None,
        evidence_refs: list[object] | None = None,
        supersedes: list[object] | None = None,
        superseded_by: str | None = None,
        occurrences: int = 1,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        self.id = _require_text(id, "memory id")
        self.type = _ensure_memory_type(type)
        self.scope = _ensure_memory_scope(scope)
        self.subject = _require_text(subject, "memory subject")
        self.predicate = _require_text(predicate, "memory predicate")
        self.value = _require_text(value, "memory value")
        self.content = _content_text(content)
        self.keywords = _dedupe_text(keywords or [])
        self.paths = _dedupe_text(paths or [])
        self.status = _ensure_memory_status(status)
        self.source = _ensure_memory_source(source)
        self.confidence = _ensure_memory_confidence(confidence)
        self.priority = _priority(priority)
        self.created_by_session_id = _optional_text(created_by_session_id)
        self.created_by_run_id = _optional_text(created_by_run_id)
        self.source_message_id = _optional_text(source_message_id)
        self.source_event_id = _optional_text(source_event_id)
        self.evidence_refs = _evidence_refs(evidence_refs or [])
        self.supersedes = _dedupe_text(supersedes or [])
        self.superseded_by = _optional_text(superseded_by)
        self.occurrences = _positive_int(occurrences)
        self.created_at = str(created_at or utc_now_iso())
        self.updated_at = str(updated_at or utc_now_iso())
        _ensure_source_trace(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MemoryRecord":
        payload = validate_memory_record_payload(value)
        return cls(
            id=payload["id"],
            type=payload["type"],
            scope=payload["scope"],
            subject=payload["subject"],
            predicate=payload["predicate"],
            value=payload["value"],
            content=payload["content"],
            keywords=payload["keywords"],
            paths=payload["paths"],
            status=payload["status"],
            source=payload["source"],
            confidence=payload["confidence"],
            priority=payload["priority"],
            created_by_session_id=payload["created_by_session_id"],
            created_by_run_id=payload["created_by_run_id"],
            source_message_id=payload["source_message_id"],
            source_event_id=payload["source_event_id"],
            evidence_refs=payload["evidence_refs"],
            supersedes=payload["supersedes"],
            superseded_by=payload["superseded_by"],
            occurrences=payload["occurrences"],
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
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


def validate_memory_record_payload(raw: object) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("memory record payload must be a mapping")
    keys = set(raw)
    schema = raw.get("schema_version")
    if schema != MEMORY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported memory schema_version: {schema}")
    legacy = sorted(keys & _LEGACY_MEMORY_KEYS)
    if legacy:
        raise ValueError("legacy memory fields are not supported: " + ", ".join(legacy))
    missing = sorted(_MEMORY_RECORD_KEYS - keys)
    if missing:
        raise ValueError("missing memory fields: " + ", ".join(missing))
    unknown = sorted(keys - _MEMORY_RECORD_KEYS)
    if unknown:
        raise ValueError("unknown memory fields: " + ", ".join(unknown))
    payload = dict(raw)
    payload["id"] = _require_text(payload.get("id"), "memory id")
    payload["type"] = _ensure_memory_type(payload.get("type"))
    payload["scope"] = _ensure_memory_scope(payload.get("scope"))
    payload["subject"] = _require_text(payload.get("subject"), "memory subject")
    payload["predicate"] = _require_text(payload.get("predicate"), "memory predicate")
    payload["value"] = _require_text(payload.get("value"), "memory value")
    payload["content"] = _content_text(payload.get("content"))
    payload["keywords"] = _string_list(payload.get("keywords"), "keywords")
    payload["paths"] = _string_list(payload.get("paths"), "paths")
    payload["status"] = _ensure_memory_status(payload.get("status"))
    payload["source"] = _ensure_memory_source(payload.get("source"))
    payload["confidence"] = _ensure_memory_confidence(payload.get("confidence"))
    payload["priority"] = _priority(payload.get("priority"))
    payload["created_by_session_id"] = _optional_text(payload.get("created_by_session_id"))
    payload["created_by_run_id"] = _optional_text(payload.get("created_by_run_id"))
    payload["source_message_id"] = _optional_text(payload.get("source_message_id"))
    payload["source_event_id"] = _optional_text(payload.get("source_event_id"))
    payload["evidence_refs"] = _evidence_refs(payload.get("evidence_refs"))
    payload["supersedes"] = _string_list(payload.get("supersedes"), "supersedes")
    payload["superseded_by"] = _optional_text(payload.get("superseded_by"))
    payload["occurrences"] = _positive_int(payload.get("occurrences"))
    payload["created_at"] = _require_text(payload.get("created_at"), "created_at")
    payload["updated_at"] = _require_text(payload.get("updated_at"), "updated_at")
    _ensure_source_trace_mapping(payload)
    return payload


@dataclass(frozen=True)
class MemoryQuery:
    """Query signals used to recall durable memory."""

    latest_user_message: str = ""
    raw_user_request: str = ""
    goal: str = ""
    current_mode: str | None = None
    current_step_title: str | None = None
    current_step_kind: str | None = None
    verification_status: str | None = None
    blocked_reason: str | None = None
    active_paths: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    artifact_summaries: list[str] = field(default_factory=list)
    session_id: str | None = None
    run_id: str | None = None
    limit: int = 5

    @property
    def text(self) -> str:
        return "\n".join(
            part
            for part in (
                self.latest_user_message,
                self.raw_user_request,
                self.goal,
                self.current_step_title or "",
                self.blocked_reason or "",
            )
            if part
        )


@dataclass(frozen=True)
class RetrievedMemory:
    record: MemoryRecord
    score: int
    reasons: list[str]


@dataclass(frozen=True)
class MemoryRecall:
    """Memory recall result consumed by ContextGovernor."""

    retrieved: list[RetrievedMemory] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)


def _ensure_memory_type(value: object) -> MemoryType:
    if value not in _MEMORY_TYPES:
        raise ValueError(f"Unknown memory type: {value}")
    return cast(MemoryType, value)


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


def _content_text(value: object) -> str:
    text = _require_text(value, "memory content")
    if len(text) > _MAX_MEMORY_CONTENT_CHARS:
        raise ValueError("memory content is too long")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
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


def _evidence_refs(value: object) -> list[str]:
    refs = _string_list(value, "evidence_refs")
    invalid = [
        ref for ref in refs
        if not any(ref.startswith(prefix) for prefix in _EVIDENCE_REF_PREFIXES)
    ]
    if invalid:
        raise ValueError("invalid memory evidence_refs: " + ", ".join(invalid))
    return refs


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


def _priority(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("memory priority must be between 0 and 5")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("memory priority must be between 0 and 5") from exc
    if number < 0 or number > 5:
        raise ValueError("memory priority must be between 0 and 5")
    return number


def _ensure_source_trace(record: MemoryRecord) -> None:
    if not (
        record.created_by_session_id
        or record.created_by_run_id
        or record.source_message_id
        or record.source_event_id
        or record.evidence_refs
    ):
        raise ValueError("memory record requires at least one source reference")


def _ensure_source_trace_mapping(payload: Mapping[str, object]) -> None:
    if not (
        payload.get("created_by_session_id")
        or payload.get("created_by_run_id")
        or payload.get("source_message_id")
        or payload.get("source_event_id")
        or payload.get("evidence_refs")
    ):
        raise ValueError("memory record requires at least one source reference")


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
    "utc_now_iso",
    "validate_memory_record_payload",
]
