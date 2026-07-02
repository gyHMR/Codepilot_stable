from __future__ import annotations

"""Deterministic experience extraction and consolidation."""

import hashlib
import uuid
from dataclasses import dataclass

from codepilot.protocols import AgentRunResult, ToolResultMessage

from .records import MemoryRecord
from .store import MemoryStore


PROMOTE_EXPERIENCE_OCCURRENCES = 2


@dataclass(frozen=True)
class ExperienceCandidate:
    key: str
    text: str
    triggers: list[str]
    related_paths: list[str]
    evidence_refs: list[str]


class ExperienceExtractor:
    """Extract reusable experience from verified failure-repair loops."""

    def extract(self, result: AgentRunResult) -> list[ExperienceCandidate]:
        messages = [
            message for message in result.messages
            if isinstance(message, ToolResultMessage)
        ]
        return [
            *self._extract_edit_repair(messages),
            *self._extract_verification_repair(messages),
        ]

    def _extract_edit_repair(
        self,
        messages: list[ToolResultMessage],
    ) -> list[ExperienceCandidate]:
        failed_match = next(
            (
                (index, message)
                for index, message in enumerate(messages)
                if message.tool_name == "edit"
                and message.is_error
                and message.error_code in {"multiple_matches", "unexpected_match_count"}
            ),
            None,
        )
        if failed_match is None:
            return []
        failed_index, failed = failed_match
        succeeded_match = next(
            (
                (index, message)
                for index, message in enumerate(messages[failed_index + 1:], failed_index + 1)
                if message.tool_name == "edit"
                and not message.is_error
                and message.status == "success"
            ),
            None,
        )
        if succeeded_match is None:
            return []
        succeeded_index, succeeded = succeeded_match
        passed = _passed_verifications_after(messages, succeeded_index)
        if not passed:
            return []
        error = failed.error_code or "edit_error"
        paths = _paths_from([failed, succeeded])
        key = f"experience:edit:{error}"
        return [
            ExperienceCandidate(
                key=key,
                text=(
                    "When edit fails because old_text is not unique or the match "
                    "count is unexpected, read the target area first and retry "
                    "with a longer unique old_text or occurrence_index."
                ),
                triggers=[
                    "phase:repair",
                    "intent:edit_file",
                    f"error:{error}",
                    *_path_triggers(paths),
                ],
                related_paths=paths,
                evidence_refs=_evidence_refs([failed, succeeded, *passed]),
            )
        ]

    def _extract_verification_repair(
        self,
        messages: list[ToolResultMessage],
    ) -> list[ExperienceCandidate]:
        failed_match = next(
            (
                (index, message)
                for index, message in enumerate(messages)
                if isinstance(message.verification, dict)
                and message.verification.get("status") == "failed"
            ),
            None,
        )
        if failed_match is None:
            return []
        failed_index, failed = failed_match
        passed = _passed_verifications_after(messages, failed_index)
        if not passed:
            return []
        paths = _paths_from([failed, *passed])
        return [
            ExperienceCandidate(
                key="experience:verification:failed_then_passed",
                text=(
                    "When verification fails after a change, inspect the failure "
                    "summary and related files, make the smallest repair, then "
                    "rerun the same verification command."
                ),
                triggers=[
                    "phase:repair",
                    "intent:debug_failure",
                    "error:verification_failed",
                    *_path_triggers(paths),
                ],
                related_paths=paths,
                evidence_refs=_evidence_refs([failed, *passed]),
            )
        ]


class MemoryConsolidator:
    """Merge duplicate memories and promote repeated verified experience."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def upsert_experience(
        self,
        candidate: ExperienceCandidate,
        *,
        run_id: str | None,
    ) -> MemoryRecord:
        existing = _active_by_key(
            self.store.load_session(),
            key=candidate.key,
            kind="experience",
        )
        if existing is None:
            record = MemoryRecord(
                id=_new_memory_id(),
                scope="session",
                kind="experience",
                key=candidate.key,
                text=candidate.text,
                triggers=list(candidate.triggers),
                related_paths=list(candidate.related_paths),
                evidence_refs=list(candidate.evidence_refs),
                source="run",
            )
        else:
            record = _merge_record(existing, candidate)
        record = self.store.update(record)
        if record.occurrences >= PROMOTE_EXPERIENCE_OCCURRENCES:
            self._promote_experience(record, run_id=run_id)
        return record

    def upsert_project_record(self, record: MemoryRecord) -> MemoryRecord:
        existing = _active_by_key(
            self.store.load_project(),
            key=record.key,
            kind=record.kind,
        )
        if existing is not None:
            existing.text = record.text
            existing.triggers = _dedupe([*existing.triggers, *record.triggers])
            existing.related_paths = _dedupe(
                [*existing.related_paths, *record.related_paths]
            )
            existing.evidence_refs = _dedupe(
                [*existing.evidence_refs, *record.evidence_refs]
            )
            existing.supersedes = _dedupe([*existing.supersedes, *record.supersedes])
            existing.occurrences += max(record.occurrences, 1)
            return self.store.update(existing)
        superseded = _active_conflicts(self.store.load_project(), record.key, record.kind)
        for old in superseded:
            old.status = "superseded"
            self.store.update(old)
            if old.id not in record.supersedes:
                record.supersedes.append(old.id)
        return self.store.update(record)

    def _promote_experience(self, session_record: MemoryRecord, *, run_id: str | None) -> MemoryRecord:
        existing = _active_by_key(
            self.store.load_project(),
            key=session_record.key,
            kind="experience",
        )
        if existing is not None:
            existing.occurrences = max(existing.occurrences, session_record.occurrences)
            existing.triggers = _dedupe([*existing.triggers, *session_record.triggers])
            existing.related_paths = _dedupe([*existing.related_paths, *session_record.related_paths])
            existing.evidence_refs = _dedupe([*existing.evidence_refs, *session_record.evidence_refs])
            return self.store.update(existing)
        promoted = MemoryRecord(
            id=_new_memory_id(),
            scope="project",
            kind="experience",
            key=session_record.key,
            text=session_record.text,
            triggers=list(session_record.triggers),
            related_paths=list(session_record.related_paths),
            evidence_refs=[
                *session_record.evidence_refs,
                *([f"run:{run_id}"] if run_id else []),
            ],
            source="promoted",
            occurrences=session_record.occurrences,
            supersedes=[session_record.id],
        )
        return self.store.update(promoted)


def memory_key_for_text(kind: str, text: str) -> str:
    lowered = text.lower()
    if "context" in lowered or "上下文" in text:
        topic = "context_design"
    elif "memory" in lowered or "记忆" in text:
        topic = "memory_contract"
    elif "生产级" in text or "学习" in text or "求职" in text:
        topic = "project_boundary"
    else:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        topic = digest
    return f"{kind}:{topic}"


def _active_by_key(
    records: list[MemoryRecord],
    *,
    key: str,
    kind: str,
) -> MemoryRecord | None:
    return next(
        (
            record
            for record in records
            if record.status == "active"
            and record.key == key
            and record.kind == kind
        ),
        None,
    )


def _active_conflicts(
    records: list[MemoryRecord],
    key: str,
    kind: str,
) -> list[MemoryRecord]:
    if kind == "correction":
        return [
            record
            for record in records
            if record.status == "active"
            and record.key == key
            and record.kind != "correction"
        ]
    return [
        record
        for record in records
        if record.status == "active"
        and record.key == key
        and record.kind == kind
    ]


def _merge_record(record: MemoryRecord, candidate: ExperienceCandidate) -> MemoryRecord:
    record.occurrences += 1
    record.triggers = _dedupe([*record.triggers, *candidate.triggers])
    record.related_paths = _dedupe([*record.related_paths, *candidate.related_paths])
    record.evidence_refs = _dedupe([*record.evidence_refs, *candidate.evidence_refs])
    return record


def _passed_verifications_after(
    messages: list[ToolResultMessage],
    index: int,
) -> list[ToolResultMessage]:
    return [
        message
        for message in messages[index + 1:]
        if isinstance(message.verification, dict)
        and message.verification.get("status") == "passed"
    ]


def _paths_from(messages: list[ToolResultMessage]) -> list[str]:
    paths: list[str] = []
    for message in messages:
        for path in message.affected_paths:
            if path not in paths:
                paths.append(path)
    return paths


def _path_triggers(paths: list[str]) -> list[str]:
    return [f"path:{path}" for path in paths]


def _evidence_refs(messages: list[ToolResultMessage]) -> list[str]:
    refs: list[str] = []
    for message in messages:
        if message.tool_call_id:
            refs.append(f"tool:{message.tool_call_id}")
            if isinstance(message.verification, dict):
                refs.append(f"verification:{message.tool_call_id}")
    return _dedupe(refs)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


__all__ = [
    "ExperienceExtractor",
    "MemoryConsolidator",
    "memory_key_for_text",
]
