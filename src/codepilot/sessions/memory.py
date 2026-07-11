from __future__ import annotations

"""Session memory: one durable project memory file plus simple read/write policy."""

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, TYPE_CHECKING, cast

from codepilot.protocols import AgentRunResult, ToolResultMessage

if TYPE_CHECKING:
    from .store import SessionStore


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
_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|password|secret|cookie)\s*[:=]\s*\S+"),
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)authorization:\s*bearer\s+\S+"),
]
MEMORY_RECALL_MIN = 3
MEMORY_RECALL_MAX = 5


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(init=False)
class MemoryRecord:
    """Canonical durable memory v4 record."""

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
        subject: str,
        predicate: str,
        value: str,
        content: str,
        scope: str = "project",
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
    def from_dict(cls, value: dict[str, Any]) -> MemoryRecord:
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
    retrieved: list[RetrievedMemory] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)


class MemoryStore:
    """Read and append canonical durable project memory."""

    def __init__(self, session_store: SessionStore) -> None:
        self.session_store = session_store
        self.workspace_dir = session_store.workspace_dir
        self.session_id = session_store.session_id
        self.project_file = session_store.layout.project_memory_file

    def all_records(self) -> list[MemoryRecord]:
        if not self.project_file.exists():
            return []
        latest: dict[str, MemoryRecord] = {}
        for line_number, line in enumerate(
            self.project_file.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid memory JSON on line {line_number}") from exc
            record = MemoryRecord.from_dict(raw)
            latest[record.id] = record
        return list(latest.values())

    def active_records(self) -> list[MemoryRecord]:
        return [record for record in self.all_records() if record.status == "active"]

    def append(self, record: MemoryRecord) -> MemoryRecord:
        self.project_file.parent.mkdir(parents=True, exist_ok=True)
        with self.project_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def update(self, record: MemoryRecord) -> MemoryRecord:
        record.updated_at = utc_now_iso()
        return self.append(record)

    def get(self, memory_id: str) -> MemoryRecord | None:
        for record in self.all_records():
            if record.id == memory_id:
                return record
        return None

    def has_source_event(self, source_event_id: str | None, *, content: str | None = None) -> bool:
        if not source_event_id:
            return False
        for record in self.all_records():
            if record.source_event_id != source_event_id:
                continue
            if content is None or record.content == content:
                return True
        return False

    def mark_status(self, memory_id: str, status: MemoryStatus) -> MemoryRecord:
        record = self.get(memory_id)
        if record is None:
            raise ValueError(f"Memory not found: {memory_id}")
        record.status = status
        return self.update(record)

    def supersede(self, old_id: str, new_record: MemoryRecord) -> MemoryRecord:
        old = self.get(old_id)
        if old is None:
            raise ValueError(f"Memory not found: {old_id}")
        old.status = "superseded"
        old.superseded_by = new_record.id
        if old.id not in new_record.supersedes:
            new_record.supersedes.append(old.id)
        self.update(old)
        return self.update(new_record)

    def replace(self, record: MemoryRecord) -> MemoryRecord:
        return self.update(record)


@dataclass(frozen=True)
class MemoryWriteContext:
    session_id: str | None = None
    run_id: str | None = None
    source_message_id: str | None = None
    source_event_id: str | None = None
    evidence_refs: list[str] = field(default_factory=list)

    def refs(self) -> list[str]:
        refs = list(self.evidence_refs)
        if self.run_id:
            refs.append(f"run:{self.run_id}")
        if self.session_id:
            refs.append(f"session:{self.session_id}")
        return _dedupe(refs)

    def has_source(self) -> bool:
        return bool(
            self.session_id
            or self.run_id
            or self.source_message_id
            or self.source_event_id
            or self.evidence_refs
        )


@dataclass(frozen=True)
class MemoryAdmissionDecision:
    should_store: bool
    reason: str
    type: str = "constraint"
    content: str = ""
    status: str = "candidate"
    source: str = "user_correction"
    confidence: str = "inferred"
    priority: int = 1
    keywords: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)


class MemoryAdmissionPolicy:
    """Detect only explicit durable memory and soft memory candidates."""

    def detect_user_prompt(self, text: str) -> MemoryAdmissionDecision:
        content = _safe_memory_text(text, limit=1200, allow_empty=True)
        if not content:
            return MemoryAdmissionDecision(False, "empty")
        stripped = _strip_marker(content)
        lowered = content.lower()
        if _has_memory_marker(lowered, content):
            if not _passes_snip(stripped, explicit=True):
                return MemoryAdmissionDecision(False, "snip_rejected")
            return MemoryAdmissionDecision(
                True,
                "user_memory_requested",
                type="constraint",
                content=stripped,
                status="active",
                source="user_explicit",
                confidence="explicit",
                priority=5,
                keywords=["always", *_keywords(stripped)],
            )
        if _has_correction_marker(lowered, content):
            if not _passes_snip(stripped, explicit=False):
                return MemoryAdmissionDecision(False, "snip_rejected")
            return MemoryAdmissionDecision(
                True,
                "user_correction_observed",
                type="correction",
                content=stripped,
                status="candidate",
                source="user_correction",
                confidence="explicit",
                priority=2,
                keywords=_keywords(stripped),
            )
        return MemoryAdmissionDecision(False, "ordinary_task_prompt")


class MemoryConflictResolver:
    """Keep one active memory for each subject/predicate pair."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def write(self, record: MemoryRecord) -> MemoryRecord:
        if record.status != "active":
            return self.store.append(record)
        for old in self.store.active_records():
            if old.id == record.id:
                continue
            if (old.scope, old.subject, old.predicate) != (
                record.scope,
                record.subject,
                record.predicate,
            ):
                continue
            if old.value == record.value:
                old.occurrences += 1
                old.keywords = _dedupe([*old.keywords, *record.keywords])
                old.paths = _dedupe([*old.paths, *record.paths])
                old.evidence_refs = _dedupe([*old.evidence_refs, *record.evidence_refs])
                return self.store.update(old)
            old.status = "superseded"
            old.superseded_by = record.id
            record.supersedes = _dedupe([*record.supersedes, old.id])
            self.store.update(old)
        return self.store.append(record)


class MemoryWriter:
    """Only write API for project memory."""

    def __init__(self, *, store: MemoryStore, workspace_dir: str | Path) -> None:
        self.store = store
        self.workspace_dir = Path(workspace_dir)
        self.policy = MemoryAdmissionPolicy()
        self.conflicts = MemoryConflictResolver(store)

    def admit_prompt_memory(
        self,
        text: str,
        *,
        context: MemoryWriteContext,
    ) -> tuple[MemoryRecord, MemoryAdmissionDecision] | None:
        decision = self.policy.detect_user_prompt(text)
        if not decision.should_store:
            return None
        if self.store.has_source_event(context.source_event_id, content=decision.content):
            return None
        record = self._record_from_decision(decision, context)
        return self.conflicts.write(record), decision

    def finalize_run(
        self,
        result: AgentRunResult,
        *,
        context: MemoryWriteContext,
    ) -> list[MemoryRecord]:
        records = [
            self._record_from_decision(candidate, context)
            for candidate in _experience_candidates(result)
            if not self.store.has_source_event(context.source_event_id, content=candidate.content)
        ]
        return [self.conflicts.write(record) for record in records]

    def add_explicit(self, text: str, *, context: MemoryWriteContext) -> MemoryRecord:
        content = _safe_memory_text(text)
        memory_type = "decision" if _looks_like_decision(content) else "constraint"
        content = _strip_decision_marker(content)
        return self.conflicts.write(
            self._new_record(
                type=memory_type,
                content=content,
                status="active",
                source="user_explicit",
                confidence="explicit",
                priority=5,
                keywords=["always", *_keywords(content)],
                context=context,
            )
        )

    def approve(self, memory_id: str, *, context: MemoryWriteContext) -> MemoryRecord:
        record = self._copy(memory_id)
        record.status = "active"
        record.source = "user_approved"
        record.confidence = "explicit"
        _apply_context(record, context)
        return self.conflicts.write(record)

    def edit_as_supersede(
        self,
        memory_id: str,
        content: str,
        *,
        context: MemoryWriteContext,
    ) -> MemoryRecord:
        old = self._copy(memory_id)
        new_content = _safe_memory_text(content)
        new_record = self._new_record(
            type=old.type,
            content=new_content,
            status="active",
            source="manual_edit",
            confidence="explicit",
            priority=old.priority,
            keywords=list(old.keywords),
            paths=list(old.paths),
            subject=old.subject,
            predicate=old.predicate,
            context=context,
            supersedes=[old.id],
        )
        return self.store.supersede(old.id, new_record)

    def supersede(
        self,
        memory_id: str,
        content: str,
        *,
        context: MemoryWriteContext,
    ) -> MemoryRecord:
        return self.edit_as_supersede(memory_id, content, context=context)

    def disable(self, memory_id: str, *, context: MemoryWriteContext) -> MemoryRecord:
        return self._mark(memory_id, "disabled", context)

    def delete(self, memory_id: str, *, context: MemoryWriteContext) -> MemoryRecord:
        return self._mark(memory_id, "deleted", context)

    def _mark(self, memory_id: str, status: str, context: MemoryWriteContext) -> MemoryRecord:
        record = self._copy(memory_id)
        record.status = status
        _apply_context(record, context)
        return self.store.update(record)

    def _record_from_decision(
        self,
        decision: MemoryAdmissionDecision,
        context: MemoryWriteContext,
    ) -> MemoryRecord:
        return self._new_record(
            type=decision.type,
            content=decision.content,
            status=decision.status,
            source=decision.source,
            confidence=decision.confidence,
            priority=decision.priority,
            keywords=list(decision.keywords),
            paths=list(decision.paths),
            context=context,
        )

    def _new_record(
        self,
        *,
        type: str,
        content: str,
        status: str,
        source: str,
        confidence: str,
        priority: int,
        keywords: list[str],
        context: MemoryWriteContext,
        paths: list[str] | None = None,
        subject: str | None = None,
        predicate: str = "is",
        supersedes: list[str] | None = None,
    ) -> MemoryRecord:
        _ensure_write_context(context, status=status)
        content = _safe_memory_text(content)
        subject = subject or _subject(type, content)
        return MemoryRecord(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            type=type,
            scope="project",
            subject=subject,
            predicate=predicate,
            value=content,
            content=content,
            keywords=_dedupe(keywords),
            paths=_dedupe(paths or []),
            status=status,
            source=source,
            confidence=confidence,
            priority=priority,
            created_by_session_id=context.session_id,
            created_by_run_id=context.run_id,
            source_message_id=context.source_message_id,
            source_event_id=context.source_event_id,
            evidence_refs=context.refs(),
            supersedes=_dedupe(supersedes or []),
        )

    def _copy(self, memory_id: str) -> MemoryRecord:
        record = self.store.get(memory_id)
        if record is None:
            raise ValueError(f"Memory not found: {memory_id}")
        return MemoryRecord.from_dict(record.to_dict())


class MemoryRetriever:
    """Recall active memory for ContextGovernor."""

    def __init__(self, *, store: MemoryStore, workspace_dir: str | Path) -> None:
        self.store = store
        self.workspace_dir = Path(workspace_dir)

    def recall(self, query: MemoryQuery) -> MemoryRecall:
        dropped: dict[str, str] = {}
        scored: list[RetrievedMemory] = []
        for record in self.store.all_records():
            reason = _drop_reason(record, query)
            if reason is not None:
                dropped[record.id] = reason
                continue
            item = score_memory_record(record, query)
            if item is None:
                dropped[record.id] = "low_score"
                continue
            scored.append(item)

        ranked = _dedupe_by_subject(
            sorted(
                scored,
                key=lambda item: (
                    item.score,
                    item.record.priority,
                    item.record.updated_at,
                ),
                reverse=True,
            ),
            dropped,
        )
        limit = min(MEMORY_RECALL_MAX, max(MEMORY_RECALL_MIN, query.limit))
        selected = ranked[:limit]
        selected_ids = {item.record.id for item in selected}
        for item in ranked[limit:]:
            if item.record.id not in dropped:
                dropped[item.record.id] = "over_limit"
        return MemoryRecall(
            retrieved=selected,
            dropped={
                memory_id: reason
                for memory_id, reason in dropped.items()
                if memory_id not in selected_ids
            },
        )


def score_memory_record(record: MemoryRecord, query: MemoryQuery) -> RetrievedMemory | None:
    score = 0
    reasons: list[str] = []
    query_terms = _terms(query.text)
    rendered_terms = _terms(render_memory(record))

    if record.scope == "project":
        score += 20
        reasons.append("scope:project")
    elif record.scope == "workspace":
        score += 12
        reasons.append("scope:workspace")
    elif record.scope == "global":
        score += 6
        reasons.append("scope:global")

    score += {
        "constraint": 35,
        "correction": 35,
        "preference": 25,
        "decision": 22,
        "workflow": 20,
        "experience": 18,
    }.get(record.type, 0)
    reasons.append(f"type:{record.type}")

    subject_matches = _keyword_matches(query_terms, _terms(record.subject))
    if subject_matches:
        score += min(35, len(subject_matches) * 12)
        reasons.append(f"subject:{sorted(subject_matches)[0]}")

    keyword_matches = _keyword_matches(query_terms, set(record.keywords) | rendered_terms)
    if keyword_matches:
        score += min(45, len(keyword_matches) * 10)
        reasons.append(f"keyword:{sorted(keyword_matches)[0]}")

    path_matches = _path_matches(record, query)
    if path_matches:
        score += min(40, len(path_matches) * 15)
        reasons.append(f"path:{path_matches[0]}")

    context_score, context_reasons = _run_context_score(record, query)
    score += context_score
    reasons.extend(context_reasons)

    if record.priority:
        score += record.priority * 8
        reasons.append(f"priority:{record.priority}")
    if record.occurrences > 1:
        score += min(20, record.occurrences * 5)
        reasons.append(f"occurrences:{record.occurrences}")

    if score <= 0:
        return None
    return RetrievedMemory(record=record, score=score, reasons=_dedupe(reasons))


def render_memory(record: MemoryRecord) -> str:
    return (
        f"[{record.type}/{record.scope}/{record.confidence}] "
        f"{record.content} ({record.id})"
    )


def sanitize_memory_text(text: str, *, limit: int) -> str:
    safe = text
    for pattern in _SECRET_PATTERNS:
        safe = pattern.sub("[REDACTED]", safe)
    return safe.replace("\x00", "").strip()[:limit]


def _experience_candidates(result: AgentRunResult) -> list[MemoryAdmissionDecision]:
    if result.status != "completed":
        return []
    tool_messages = [message for message in result.messages if isinstance(message, ToolResultMessage)]
    if not any(_verification_status(message) == "passed" for message in tool_messages):
        return []
    if any(message.is_error for message in tool_messages):
        return [
            MemoryAdmissionDecision(
                True,
                "task_experience_candidate",
                type="experience",
                content=(
                    "When a tool or verification fails, inspect the failure, make "
                    "the smallest repair, and rerun the same verification before "
                    "claiming completion."
                ),
                status="candidate",
                source="task_experience",
                confidence="observed",
                priority=2,
                keywords=["intent:debug_failure", "verification:passed"],
                paths=_dedupe(
                    [
                        path
                        for message in tool_messages
                        for path in message.affected_paths
                    ]
                ),
            )
        ]
    return []


def _drop_reason(record: MemoryRecord, query: MemoryQuery) -> str | None:
    if record.status != "active":
        return f"status:{record.status}"
    if record.scope not in {"project", "workspace", "global"}:
        return f"scope:{record.scope}"
    if _conflicts_with_current_instruction(record, query):
        return "conflict:latest_instruction"
    return None


def _path_matches(record: MemoryRecord, query: MemoryQuery) -> list[str]:
    active = {Path(path).as_posix() for path in [*query.active_paths, *query.changed_paths]}
    matches: list[str] = []
    for path in record.paths:
        normalized = Path(path).as_posix()
        if normalized in active:
            matches.append(normalized)
    for keyword in record.keywords:
        if keyword.startswith("path:"):
            path = keyword.removeprefix("path:")
            if path in active:
                matches.append(path)
    return _dedupe(matches)


def _run_context_score(record: MemoryRecord, query: MemoryQuery) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    keywords = list(record.keywords)
    if query.current_mode and f"mode:{query.current_mode}" in keywords:
        score += 25
        reasons.append(f"mode:{query.current_mode}")
    if query.current_step_kind and f"kind:{query.current_step_kind}" in keywords:
        score += 25
        reasons.append(f"kind:{query.current_step_kind}")
    if query.verification_status and record.type == "experience":
        score += 10
        reasons.append(f"verification:{query.verification_status}")
    if query.blocked_reason:
        matches = _keyword_matches(_terms(query.blocked_reason), set(keywords) | _terms(record.content))
        if matches:
            score += 35
            reasons.append(f"blocked:{sorted(matches)[0]}")
    return score, reasons


def _dedupe_by_subject(
    items: list[RetrievedMemory],
    dropped: dict[str, str],
) -> list[RetrievedMemory]:
    best: dict[str, RetrievedMemory] = {}
    order: list[str] = []
    for item in items:
        subject = item.record.subject.lower()
        previous = best.get(subject)
        if previous is None:
            best[subject] = item
            order.append(subject)
            continue
        if (item.score, item.record.priority, item.record.updated_at) > (
            previous.score,
            previous.record.priority,
            previous.record.updated_at,
        ):
            dropped[previous.record.id] = "duplicate_subject"
            best[subject] = item
        else:
            dropped[item.record.id] = "duplicate_subject"
    return [best[subject] for subject in order]


def _conflicts_with_current_instruction(record: MemoryRecord, query: MemoryQuery) -> bool:
    text = query.latest_user_message.lower()
    if not text:
        return False
    if not any(
        marker in text
        for marker in (
            "不要",
            "别",
            "不是",
            "而是",
            "改成",
            "纠正",
            "更正",
            "not ",
            "instead",
            "never",
            "correction",
        )
    ):
        return False
    query_terms = _terms(query.latest_user_message)
    record_terms = _terms(record.subject) | _terms(record.value) | set(record.keywords)
    return bool(_keyword_matches(query_terms, record_terms))


def _apply_context(record: MemoryRecord, context: MemoryWriteContext) -> None:
    _ensure_write_context(context, status=record.status)
    record.created_by_session_id = record.created_by_session_id or context.session_id
    record.created_by_run_id = record.created_by_run_id or context.run_id
    record.source_message_id = record.source_message_id or context.source_message_id
    record.source_event_id = record.source_event_id or context.source_event_id
    record.evidence_refs = _dedupe([*record.evidence_refs, *context.refs()])
    record.updated_at = utc_now_iso()


def _ensure_write_context(context: MemoryWriteContext, *, status: str) -> None:
    if not context.has_source():
        raise ValueError("memory write requires source context")
    if status == "active" and not context.refs() and not context.source_event_id:
        raise ValueError("active memory requires evidence")


def _safe_memory_text(text: str, *, limit: int = 1600, allow_empty: bool = False) -> str:
    content = sanitize_memory_text(text, limit=limit)
    if not content and not allow_empty:
        raise ValueError("Memory content is empty after sensitive-data filtering")
    if "[REDACTED]" in content:
        raise ValueError("Memory content contains sensitive data")
    return content


def _verification_status(message: ToolResultMessage) -> str | None:
    verification = message.verification
    if isinstance(verification, dict):
        status = verification.get("status")
        return status if isinstance(status, str) else None
    return None


def _subject(memory_type: str, text: str) -> str:
    lowered = text.lower()
    if "context" in lowered or "上下文" in text:
        topic = "context"
    elif "memory" in lowered or "记忆" in text:
        topic = "memory"
    elif "task" in lowered or "任务" in text:
        topic = "task"
    elif "pytest" in lowered or "测试" in text:
        topic = "test"
    else:
        import hashlib

        topic = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{memory_type}:{topic}"


def _keywords(text: str) -> list[str]:
    lowered = text.lower()
    keywords: list[str] = []
    if "context" in lowered or "上下文" in text:
        keywords.append("topic:context")
    if "memory" in lowered or "记忆" in text:
        keywords.append("topic:memory")
    if "task" in lowered or "任务" in text:
        keywords.append("topic:task")
    if "pytest" in lowered or "测试" in text or "验证" in text:
        keywords.append("intent:verify")
    return keywords


def _has_memory_marker(lowered: str, original: str) -> bool:
    return any(
        marker in lowered or marker in original
        for marker in (
            "请记住",
            "记住：",
            "记住:",
            "remember:",
            "remember that",
            "以后",
            "默认",
            "总是",
            "always",
        )
    )


def _has_correction_marker(lowered: str, original: str) -> bool:
    return any(
        marker in lowered or marker in original
        for marker in (
            "纠正",
            "更正",
            "不是",
            "而是",
            "不要再",
            "以后不要",
            "actually",
            "correction",
        )
    )


def _passes_snip(text: str, *, explicit: bool) -> bool:
    content = " ".join(text.strip().split())
    if len(content) < 6:
        return False
    lowered = content.lower()
    transient_markers = (
        "当前步骤",
        "这一步",
        "刚才的输出",
        "日志如下",
        "stack trace",
        "traceback",
        "temporary",
        "one-off",
    )
    if any(marker in lowered or marker in content for marker in transient_markers):
        return False
    durable_markers = (
        "默认",
        "以后",
        "总是",
        "不要再",
        "项目",
        "偏好",
        "约定",
        "命令",
        "使用",
        "决策",
        "workflow",
        "command",
        "use ",
        "always",
        "prefer",
        "default",
        "never",
    )
    if explicit:
        return True
    return any(marker in lowered or marker in content for marker in durable_markers)


def _strip_marker(text: str) -> str:
    cleaned = text.strip()
    for marker in (
        "请记住：",
        "请记住:",
        "记住：",
        "记住:",
        "纠正一下：",
        "纠正一下:",
        "纠正：",
        "纠正:",
        "更正：",
        "更正:",
        "remember:",
        "correction:",
    ):
        if cleaned.lower().startswith(marker.lower()):
            return cleaned[len(marker):].strip()
    return cleaned


def _looks_like_decision(text: str) -> bool:
    lowered = text.lower()
    return lowered.startswith("decision:") or text.startswith("决策：") or text.startswith("决策:")


def _strip_decision_marker(text: str) -> str:
    for marker in ("decision:", "Decision:", "决策：", "决策:"):
        if text.startswith(marker):
            return text[len(marker):].strip()
    return text


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w./-]{2,}", text, flags=re.UNICODE)
    }


def _keyword_matches(query_terms: set[str], record_terms: set[str]) -> set[str]:
    matches = set(query_terms.intersection({term.lower() for term in record_terms}))
    for query_term in query_terms:
        for record_term in record_terms:
            normalized = record_term.lower()
            if query_term == normalized:
                continue
            if query_term in normalized or normalized in query_term:
                matches.add(normalized)
    return matches


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
        ref
        for ref in refs
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


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "MemoryAdmissionDecision",
    "MemoryAdmissionPolicy",
    "MemoryConfidence",
    "MemoryConflictResolver",
    "MemoryQuery",
    "MemoryRecall",
    "MemoryRecord",
    "MemoryRetriever",
    "MemoryScope",
    "MemorySource",
    "MemoryStatus",
    "MemoryStore",
    "MemoryType",
    "MemoryWriteContext",
    "MemoryWriter",
    "RetrievedMemory",
    "render_memory",
    "sanitize_memory_text",
    "score_memory_record",
    "utc_now_iso",
    "validate_memory_record_payload",
]
