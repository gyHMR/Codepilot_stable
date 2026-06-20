from __future__ import annotations

"""Write structured memories from user prompts, tool results, and run results."""

import uuid
from pathlib import Path
from typing import Any

from codepilot.protocols import AgentRunResult, ToolResultMessage
from codepilot.tools.sandbox import file_state_for_path

from .files import sanitize_memory_text
from .records import MemoryRecord, MemoryStatus, utc_now_iso
from .store import MemoryStore


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
                existing.content["last_seen_at"] = utc_now_iso()
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
                    "last_seen_at": utc_now_iso(),
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


def _new_memory_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


def _tool_result_text(message: ToolResultMessage) -> str:
    return "".join(
        str(getattr(block, "text", ""))
        for block in message.content
        if getattr(block, "text", "")
    ).strip()


__all__ = ["MemoryWriter"]
