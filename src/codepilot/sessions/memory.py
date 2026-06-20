from __future__ import annotations

"""Evidence-driven session and project memory.

MEMORY.md remains user-maintained pinned project memory. Automatic memories are
stored as structured records and are retrieved by ContextCompiler.
"""

import json
import logging
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from codepilot.protocols import AgentRunResult, ToolResultMessage
from codepilot.tools.sandbox import file_state_for_path

if TYPE_CHECKING:
    from .store import SessionStore

logger = logging.getLogger("codepilot.sessions.memory")

MemoryKind = Literal["task", "file", "failure", "decision", "project"]
MemoryScope = Literal["session", "project"]
MemoryTrust = Literal["observed", "verified", "user_given", "model_claim"]
MemoryStatus = Literal["active", "stale", "superseded", "deleted"]

MEMORY_SCHEMA_VERSION = 1
_SECRET_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|password|secret|cookie)\s*[:=]\s*\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{12,}\b"),
]
_FAILURE_CODES = {
    "unexpected_match_count",
    "multiple_matches",
    "path_not_found",
    "path_not_file",
    "permission_denied",
    "dangerous_command",
    "stale_file",
    "no_match",
}


def _utc_now_iso() -> str:
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
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

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
            created_at=str(value.get("created_at", _utc_now_iso())),
            updated_at=str(value.get("updated_at", _utc_now_iso())),
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


class MemoryStore:
    """Persist structured memory without deciding what should be remembered."""

    def __init__(self, session_store: "SessionStore") -> None:
        self.session_store = session_store
        self.workspace_dir = session_store.workspace_dir
        self.session_id = session_store.session_id
        self.session_file = session_store.memory_file
        self.project_file = self.workspace_dir / ".codepilot" / "memory" / "project.jsonl"

    def load_session(self) -> list[MemoryRecord]:
        if not self.session_file.exists():
            return []
        try:
            payload = json.loads(self.session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("failed to load session memory file=%s", self.session_file)
            return []
        records = payload.get("records", []) if isinstance(payload, dict) else []
        return [
            MemoryRecord.from_dict(item)
            for item in records
            if isinstance(item, dict)
        ]

    def save_session(self, records: list[MemoryRecord]) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "records": [record.to_dict() for record in records],
        }
        _atomic_write_json(self.session_file, payload)

    def load_project(self) -> list[MemoryRecord]:
        if not self.project_file.exists():
            return []
        latest: dict[str, MemoryRecord] = {}
        for line in self.project_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping invalid project memory line")
                continue
            if isinstance(raw, dict):
                record = MemoryRecord.from_dict(raw)
                latest[record.id] = record
        return list(latest.values())

    def append_project(self, record: MemoryRecord) -> None:
        self.project_file.parent.mkdir(parents=True, exist_ok=True)
        with self.project_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def upsert_session(self, record: MemoryRecord) -> MemoryRecord:
        records = self.load_session()
        for index, existing in enumerate(records):
            if existing.id == record.id:
                records[index] = record
                break
        else:
            records.append(record)
        self.save_session(records)
        return record

    def update(self, record: MemoryRecord) -> MemoryRecord:
        record.updated_at = _utc_now_iso()
        if record.scope == "project":
            self.append_project(record)
        else:
            self.upsert_session(record)
        return record

    def mark_status(self, memory_id: str, status: MemoryStatus) -> MemoryRecord:
        for record in self.load_session():
            if record.id == memory_id:
                record.status = status
                return self.update(record)
        for record in self.load_project():
            if record.id == memory_id:
                record.status = status
                return self.update(record)
        raise ValueError(f"Memory not found: {memory_id}")

    def get(self, memory_id: str) -> MemoryRecord | None:
        for record in [*self.load_session(), *self.load_project()]:
            if record.id == memory_id:
                return record
        return None


class MemoryWriter:
    """Conservatively turn user/tool/run facts into structured memory."""

    def __init__(self, *, store: MemoryStore, workspace_dir: str | Path) -> None:
        self.store = store
        self.workspace_dir = Path(workspace_dir)

    def remember_task(self, text: str, *, run_id: str | None = None) -> MemoryRecord:
        safe_text = sanitize_memory_text(text, limit=1200)
        records = self.store.load_session()
        existing = next(
            (record for record in records if record.kind == "task" and record.status == "active"),
            None,
        )
        if existing is None:
            existing = MemoryRecord(
                id=_new_memory_id(),
                kind="task",
                scope="session",
                content={
                    "goal": safe_text,
                    "constraints": [],
                    "confirmed_findings": [],
                    "open_questions": [],
                    "blocked_on": [],
                    "next_action": None,
                },
                source="user_prompt",
                source_run_id=run_id,
                trust="user_given",
            )
        else:
            if existing.content.get("goal") != safe_text:
                existing.content = {
                    "goal": safe_text,
                    "constraints": [],
                    "confirmed_findings": [],
                    "open_questions": [],
                    "blocked_on": [],
                    "next_action": None,
                }
            else:
                existing.content["goal"] = safe_text
            existing.source_run_id = run_id
            existing.source = "user_prompt"
            existing.trust = "user_given"
        return self.store.update(existing)

    def observe_tool_result(
        self,
        message: ToolResultMessage,
        *,
        run_id: str | None = None,
    ) -> list[MemoryRecord]:
        created: list[MemoryRecord] = []
        tool_name = message.tool_name.lower()
        details = message.details if isinstance(message.details, dict) else {}
        state = details.get("file_state")
        if not isinstance(state, dict):
            state = message.metadata.get("file_state")

        if message.workspace_changed:
            self.invalidate_paths(message.affected_paths)

        if tool_name == "read" and not message.is_error and isinstance(state, dict):
            record = self._remember_file(message, state, run_id=run_id)
            if record is not None:
                created.append(record)

        if message.is_error and message.error_code in _FAILURE_CODES:
            created.append(self._remember_failure(message, run_id=run_id))
        elif not message.is_error:
            created.extend(self._resolve_failures(message, run_id=run_id))

        if message.verification:
            task = self._active_task()
            if task is not None:
                verification = message.verification
                command = sanitize_memory_text(str(verification.get("command", "")), limit=300)
                status = str(verification.get("status", "unknown"))
                finding = f"Verification {status}: {command or message.tool_name}"
                findings = task.content.setdefault("confirmed_findings", [])
                if isinstance(findings, list) and finding not in findings:
                    findings.append(finding)
                task.trust = "verified"
                task.source_run_id = run_id
                self.store.update(task)
                created.append(task)
        return created

    def finalize_run(self, result: AgentRunResult) -> list[MemoryRecord]:
        task = self._active_task()
        if task is None:
            return []
        task.source_run_id = result.run_id
        if result.task is not None:
            self._project_task_summary(task, result)
        else:
            task.content["next_action"] = None
        if result.status == "completed" and (
            result.task is None or result.task.completion_satisfied
        ):
            task.content["next_action"] = None
            findings = task.content.setdefault("confirmed_findings", [])
            summary = f"Run completed: {result.stop_reason}"
            if isinstance(findings, list) and summary not in findings:
                findings.append(summary)
        elif result.error:
            blocked = task.content.setdefault("blocked_on", [])
            message = sanitize_memory_text(result.error.message, limit=500)
            if isinstance(blocked, list) and message not in blocked:
                blocked.append(message)
        return [self.store.update(task)]

    def _project_task_summary(
        self,
        task: MemoryRecord,
        result: AgentRunResult,
    ) -> None:
        summary = result.task
        if summary is None:
            return
        task.content["goal"] = sanitize_memory_text(summary.goal, limit=1200)
        task.content["task_progress"] = {
            "completed_steps": list(summary.completed_steps),
            "pending_steps": list(summary.pending_steps),
            "blocked_steps": list(summary.blocked_steps),
            "completion_satisfied": summary.completion_satisfied,
            "completion_reason": summary.completion_reason,
        }
        task.content["next_action"] = (
            None if summary.completion_satisfied else summary.next_action
        )
        findings = task.content.setdefault("confirmed_findings", [])
        if isinstance(findings, list):
            for step in summary.completed_steps:
                item = f"Completed step: {sanitize_memory_text(step, limit=200)}"
                if item not in findings:
                    findings.append(item)
        blocked = task.content.setdefault("blocked_on", [])
        if isinstance(blocked, list):
            for step in summary.blocked_steps:
                item = f"Blocked step: {sanitize_memory_text(step, limit=200)}"
                if item not in blocked:
                    blocked.append(item)
            if (
                not summary.completion_satisfied
                and summary.completion_reason
                and summary.completion_reason not in blocked
            ):
                blocked.append(
                    sanitize_memory_text(summary.completion_reason, limit=300)
                )
        task.trust = "verified" if summary.completion_satisfied else "observed"

    def add_project(self, text: str) -> MemoryRecord:
        content = sanitize_memory_text(text, limit=1600)
        if not content:
            raise ValueError("Memory content is empty after sensitive-data filtering")
        record = MemoryRecord(
            id=_new_memory_id(),
            kind="project",
            scope="project",
            content={"knowledge": content, "pinned": False},
            source="user_command",
            trust="user_given",
        )
        return self.store.update(record)

    def promote(self, memory_id: str) -> MemoryRecord:
        source = self.store.get(memory_id)
        if source is None:
            raise ValueError(f"Memory not found: {memory_id}")
        if source.status != "active":
            raise ValueError("Only active memory can be promoted")
        promoted = MemoryRecord(
            id=_new_memory_id(),
            kind="project" if source.kind == "task" else source.kind,
            scope="project",
            content=dict(source.content),
            source=f"promoted:{source.id}",
            source_run_id=source.source_run_id,
            related_paths=list(source.related_paths),
            source_hashes=dict(source.source_hashes),
            trust=source.trust,
        )
        self.store.update(promoted)
        return promoted

    def invalidate_paths(self, paths: list[str]) -> list[MemoryRecord]:
        normalized = {Path(path).as_posix() for path in paths}
        changed: list[MemoryRecord] = []
        for record in self.store.load_session():
            if (
                record.kind == "file"
                and record.status == "active"
                and normalized.intersection(record.related_paths)
            ):
                record.status = "stale"
                self.store.update(record)
                changed.append(record)
        return changed

    def validate_freshness(self) -> list[MemoryRecord]:
        changed: list[MemoryRecord] = []
        for record in [*self.store.load_session(), *self.store.load_project()]:
            if record.kind != "file" or record.status not in {"active", "stale"}:
                continue
            fresh = True
            for path, source_hash in record.source_hashes.items():
                state = file_state_for_path(self.workspace_dir, path)
                if not state.get("exists") or state.get("sha256") != source_hash:
                    fresh = False
                    break
            target_status: MemoryStatus = "active" if fresh else "stale"
            if record.status != target_status:
                record.status = target_status
                self.store.update(record)
                changed.append(record)
        return changed

    def _remember_file(
        self,
        message: ToolResultMessage,
        state: dict[str, Any],
        *,
        run_id: str | None,
    ) -> MemoryRecord | None:
        path = state.get("path")
        source_hash = state.get("sha256")
        if not isinstance(path, str) or not isinstance(source_hash, str):
            return None
        summary = sanitize_memory_text(_tool_result_text(message), limit=600)
        records = self.store.load_session()
        for existing in records:
            if existing.kind != "file" or path not in existing.related_paths:
                continue
            if existing.source_hashes.get(path) == source_hash:
                existing.status = "active"
                existing.content["summary"] = summary
                existing.content["access_count"] = int(existing.content.get("access_count", 1)) + 1
                existing.source_run_id = run_id
                return self.store.update(existing)
            if existing.status == "active":
                existing.status = "stale"
                self.store.update(existing)
        record = MemoryRecord(
            id=_new_memory_id(),
            kind="file",
            scope="session",
            content={
                "path": path,
                "role": "reference",
                "summary": summary,
                "key_symbols": [],
                "findings": [],
                "access_count": 1,
            },
            source=f"tool:{message.tool_name}",
            source_run_id=run_id,
            related_paths=[path],
            source_hashes={path: source_hash},
            trust="observed",
        )
        return self.store.update(record)

    def _remember_failure(
        self,
        message: ToolResultMessage,
        *,
        run_id: str | None,
    ) -> MemoryRecord:
        paths = [Path(path).as_posix() for path in message.affected_paths]
        target_path = message.metadata.get("tool_target_path")
        if isinstance(target_path, str):
            normalized_target = Path(target_path).as_posix()
            if normalized_target not in paths:
                paths.append(normalized_target)
        signature = message.error_code or "tool_error"
        action = message.tool_name
        for existing in self.store.load_session():
            if (
                existing.kind == "failure"
                and existing.content.get("action") == action
                and existing.content.get("failure_signature") == signature
                and existing.related_paths == paths
            ):
                existing.content["occurrence_count"] = int(
                    existing.content.get("occurrence_count", 1)
                ) + 1
                existing.content["last_seen_at"] = _utc_now_iso()
                existing.source_run_id = run_id
                return self.store.update(existing)
        return self.store.update(
            MemoryRecord(
                id=_new_memory_id(),
                kind="failure",
                scope="session",
                content={
                    "action": action,
                    "failure_signature": signature,
                    "cause": sanitize_memory_text(_tool_result_text(message), limit=400),
                    "resolution": None,
                    "occurrence_count": 1,
                    "last_seen_at": _utc_now_iso(),
                },
                source=f"tool:{message.tool_name}",
                source_run_id=run_id,
                related_paths=paths,
                trust="observed",
            )
        )

    def _resolve_failures(
        self,
        message: ToolResultMessage,
        *,
        run_id: str | None,
    ) -> list[MemoryRecord]:
        target_path = message.metadata.get("tool_target_path")
        success_paths = {Path(path).as_posix() for path in message.affected_paths}
        if isinstance(target_path, str):
            success_paths.add(Path(target_path).as_posix())
        resolved: list[MemoryRecord] = []
        for record in self.store.load_session():
            if (
                record.kind != "failure"
                or record.status != "active"
                or record.content.get("action") != message.tool_name
            ):
                continue
            if record.related_paths and not success_paths.intersection(record.related_paths):
                continue
            record.content["resolution"] = sanitize_memory_text(
                message.diff_summary or _tool_result_text(message) or "later call succeeded",
                limit=400,
            )
            record.source_run_id = run_id
            resolved.append(self.store.update(record))
        return resolved

    def _active_task(self) -> MemoryRecord | None:
        return next(
            (
                record
                for record in self.store.load_session()
                if record.kind == "task" and record.status == "active"
            ),
            None,
        )


class MemoryRetriever:
    def __init__(self, *, store: MemoryStore, workspace_dir: str | Path) -> None:
        self.store = store
        self.workspace_dir = Path(workspace_dir)

    def retrieve(self, query: MemoryQuery) -> list[RetrievedMemory]:
        records = [*self.store.load_session(), *self.store.load_project()]
        query_terms = _terms(query.text)
        active_paths = {Path(path).as_posix() for path in query.active_paths}
        ranked: list[RetrievedMemory] = []
        for record in records:
            if record.status != "active":
                continue
            if (
                record.kind == "failure"
                and int(record.content.get("occurrence_count", 1)) < 2
                and not record.content.get("resolution")
            ):
                continue
            score = 0
            reasons: list[str] = []
            if record.kind == "task":
                score += 100
                reasons.append("task_memory")
            related = active_paths.intersection(record.related_paths)
            if related:
                score += 40
                reasons.append(f"related_path:{sorted(related)[0]}")
            record_terms = _terms(render_memory(record))
            keyword_matches = sorted(query_terms.intersection(record_terms))
            if keyword_matches:
                score += min(30, len(keyword_matches) * 10)
                reasons.append(f"keyword:{keyword_matches[0]}")
            if record.trust in {"verified", "observed"}:
                score += 20
                reasons.append(f"trust:{record.trust}")
            if record.scope == "project":
                score += 10
                reasons.append("project_memory")
            if record.trust == "model_claim":
                score -= 20
                reasons.append("model_claim_penalty")
            if score > 0:
                ranked.append(RetrievedMemory(record=record, score=score, reasons=reasons))

        ranked.sort(
            key=lambda item: (item.score, item.record.updated_at),
            reverse=True,
        )
        return _apply_kind_limits(ranked, query.limit)

    def validate_freshness(self) -> list[MemoryRecord]:
        return MemoryWriter(
            store=self.store,
            workspace_dir=self.workspace_dir,
        ).validate_freshness()

    def pinned_memory(self) -> str:
        return load_global_memory(self.workspace_dir)


def render_memory(record: MemoryRecord) -> str:
    content = record.content
    if record.kind == "task":
        parts = [f"Task goal: {content.get('goal', '')}"]
        for key, label in [
            ("constraints", "Constraints"),
            ("confirmed_findings", "Confirmed"),
            ("blocked_on", "Blocked"),
        ]:
            values = content.get(key)
            if isinstance(values, list) and values:
                parts.append(f"{label}: {'; '.join(str(item) for item in values[:5])}")
        if content.get("next_action"):
            parts.append(f"Next: {content['next_action']}")
        progress = content.get("task_progress")
        if isinstance(progress, dict):
            pending = progress.get("pending_steps")
            if isinstance(pending, list) and pending:
                parts.append(f"Pending: {'; '.join(str(item) for item in pending[:5])}")
        return " | ".join(parts)
    if record.kind == "file":
        return (
            f"File {content.get('path', '')}: {content.get('summary', '')} "
            f"(hash={next(iter(record.source_hashes.values()), '')[:12]})"
        ).strip()
    if record.kind == "failure":
        return (
            f"Failure lesson: {content.get('action', '')} -> "
            f"{content.get('failure_signature', '')}; "
            f"cause={content.get('cause') or 'unknown'}; "
            f"resolution={content.get('resolution') or 'not confirmed'}"
        )
    if record.kind == "decision":
        return f"Decision: {content.get('decision', '')}; rationale={content.get('rationale', '')}"
    return f"Project knowledge: {content.get('knowledge', '')}"


def sanitize_memory_text(text: str, *, limit: int) -> str:
    safe = text
    for pattern in _SECRET_PATTERNS:
        safe = pattern.sub("[REDACTED]", safe)
    safe = safe.replace("\x00", "").strip()
    return safe[:limit]


def load_global_memory(workspace_dir: str | Path) -> str:
    """Load user-maintained pinned `.codepilot/MEMORY.md`."""
    path = Path(workspace_dir) / ".codepilot" / "MEMORY.md"
    return _read_memory_file(path)


def save_global_memory(workspace_dir: str | Path, content: str) -> None:
    """Save pinned MEMORY.md. Automatic memory never calls this function."""
    path = Path(workspace_dir) / ".codepilot" / "MEMORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    logger.info("global memory saved chars=%d", len(content))


def _read_memory_file(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8").strip()
        if text:
            logger.debug("loaded memory file=%s chars=%d", path, len(text))
        return text
    except Exception as exc:
        logger.warning("failed to read memory file=%s: %s", path, exc)
        return ""


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


def _tool_result_text(message: ToolResultMessage) -> str:
    return "".join(
        str(getattr(block, "text", ""))
        for block in message.content
        if getattr(block, "text", "")
    ).strip()


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[\w./-]{2,}", text, flags=re.UNICODE)
    }


def _apply_kind_limits(
    ranked: list[RetrievedMemory],
    total_limit: int,
) -> list[RetrievedMemory]:
    limits = {"task": 1, "file": 3, "failure": 2, "decision": 2, "project": 3}
    counts: dict[str, int] = {}
    selected: list[RetrievedMemory] = []
    for item in ranked:
        kind = item.record.kind
        if counts.get(kind, 0) >= limits.get(kind, total_limit):
            continue
        selected.append(item)
        counts[kind] = counts.get(kind, 0) + 1
        if len(selected) >= total_limit:
            break
    return selected


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
    "MemoryRetriever",
    "MemoryScope",
    "MemoryStatus",
    "MemoryStore",
    "MemoryTrust",
    "MemoryWriter",
    "RetrievedMemory",
    "load_global_memory",
    "render_memory",
    "sanitize_memory_text",
    "save_global_memory",
]
