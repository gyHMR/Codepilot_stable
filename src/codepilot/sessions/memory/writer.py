from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from codepilot.protocols import AgentRunResult, ToolResultMessage

from .files import sanitize_memory_text
from .records import MemoryRecord, utc_now_iso
from .store import MemoryStore


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
    """Small deterministic policy for explicit memory boundaries."""

    def detect_user_prompt(self, text: str) -> MemoryAdmissionDecision:
        content = _safe_memory_text(text, limit=1200, allow_empty=True)
        if not content:
            return MemoryAdmissionDecision(False, "empty")
        stripped = _strip_marker(content)
        lowered = content.lower()
        if _has_memory_marker(lowered, content):
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
            durable = _has_memory_marker(lowered, content)
            return MemoryAdmissionDecision(
                True,
                "user_correction_observed",
                type="correction",
                content=stripped,
                status="active" if durable else "candidate",
                source="user_correction",
                confidence="explicit",
                priority=4 if durable else 2,
                keywords=_keywords(stripped),
            )
        return MemoryAdmissionDecision(False, "ordinary_task_prompt")


class MemoryConflictResolver:
    """Keep one active memory per subject/predicate pair."""

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
    """The only write API for durable project memory."""

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
        record = self._record_from_decision(decision, context)
        return self.conflicts.write(record), decision

    def finalize_run(
        self,
        result: AgentRunResult,
        *,
        context: MemoryWriteContext,
    ) -> list[MemoryRecord]:
        candidates = _experience_candidates(result)
        records = [self._record_from_decision(candidate, context) for candidate in candidates]
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

    def promote(self, memory_id: str, *, context: MemoryWriteContext) -> MemoryRecord:
        record = self._copy(memory_id)
        if record.status != "candidate":
            raise ValueError("Only candidate memory can be promoted")
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

    def supersede(self, memory_id: str, content: str, *, context: MemoryWriteContext) -> MemoryRecord:
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
        _ensure_context(context, status=status)
        content = _safe_memory_text(content)
        subject = subject or _subject(type, content)
        return MemoryRecord(
            id=_new_memory_id(),
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


def _apply_context(record: MemoryRecord, context: MemoryWriteContext) -> None:
    _ensure_context(context, status=record.status)
    record.created_by_session_id = record.created_by_session_id or context.session_id
    record.created_by_run_id = record.created_by_run_id or context.run_id
    record.source_message_id = record.source_message_id or context.source_message_id
    record.source_event_id = record.source_event_id or context.source_event_id
    record.evidence_refs = _dedupe([*record.evidence_refs, *context.refs()])
    record.updated_at = utc_now_iso()


def _ensure_context(context: MemoryWriteContext, *, status: str) -> None:
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
        for marker in ("请记住", "记住：", "记住:", "remember:", "remember that", "以后", "默认", "总是", "always")
    )


def _has_correction_marker(lowered: str, original: str) -> bool:
    return any(
        marker in lowered or marker in original
        for marker in ("纠正", "更正", "不是", "而是", "不要再", "以后不要", "actually", "correction")
    )


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


def _new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


__all__ = [
    "MemoryAdmissionDecision",
    "MemoryAdmissionPolicy",
    "MemoryConflictResolver",
    "MemoryWriteContext",
    "MemoryWriter",
]
