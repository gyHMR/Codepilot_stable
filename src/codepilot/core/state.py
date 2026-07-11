from __future__ import annotations

"""Mechanical facts collected during one agent run."""

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from codepilot.protocols import (
    AgentRunCounters,
    RunSignalsSummary,
    RunSignalsVerificationStatus,
    RunVerification,
    RunVerificationStatus,
    ToolCall,
    ToolResultMessage,
)


_PLAN_EXPLORATION_TOOL_NAMES = frozenset(
    {
        "ls",
        "find",
        "read",
        "grep",
        "workspace_status",
        "list_exploration_agents",
        "dispatch_exploration",
    }
)


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


@dataclass
class RunState:
    """Run-local execution facts.

    This state is intentionally not a task judge. It only records facts that the
    runner and RunGuard can observe directly.
    """

    run_id: str
    session_id: str | None
    counters: AgentRunCounters = field(default_factory=AgentRunCounters)
    workspace_changed: bool = False
    affected_paths: set[str] = field(default_factory=set)
    verification: list[RunVerification] = field(default_factory=list)
    verification_status: RunSignalsVerificationStatus = "unknown"
    last_error: dict[str, Any] | None = None
    approval_required: bool = False
    tool_unavailable: bool = False
    cancelled: bool = False
    last_tool_fingerprint: str | None = None
    repeated_tool_calls: int = 0
    last_truncated_read_fingerprint: str | None = None
    repeated_truncated_reads: int = 0
    truncated_read_recovery_sent: bool = False
    seen_tool_call_ids: set[str] = field(default_factory=set)
    pending_approval_tool_call_ids: set[str] = field(default_factory=set)
    qualified_plan_failure_item_id: str | None = None
    qualified_plan_failure_count: int = 0
    plan_exploration_tool_call_ids: set[str] = field(default_factory=set)

    def has_plan_exploration_evidence(self) -> bool:
        return bool(self.plan_exploration_tool_call_ids)

    def has_repeated_call(
        self,
        tool_calls: list[ToolCall],
        *,
        limit: int,
    ) -> bool:
        if limit <= 0:
            return False
        for tool_call in tool_calls:
            fingerprint = json.dumps(
                [tool_call.name, tool_call.arguments],
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            if fingerprint == self.last_tool_fingerprint:
                self.repeated_tool_calls += 1
            else:
                self.last_tool_fingerprint = fingerprint
                self.repeated_tool_calls = 1
            if self.repeated_tool_calls > limit:
                return True
        return False

    def truncated_read_recovery_instruction(
        self,
        results: list[ToolResultMessage],
    ) -> str | None:
        fingerprint, metadata = _truncated_read_fingerprint(results)
        if fingerprint is None:
            return None
        if fingerprint == self.last_truncated_read_fingerprint:
            self.repeated_truncated_reads += 1
        else:
            self.last_truncated_read_fingerprint = fingerprint
            self.repeated_truncated_reads = 1
            self.truncated_read_recovery_sent = False
        if self.repeated_truncated_reads < 2 or self.truncated_read_recovery_sent:
            return None
        self.truncated_read_recovery_sent = True
        path = metadata.get("path") or "the same file"
        next_offset = metadata.get("next_offset")
        return (
            "The previous read result was truncated and the same file range was read again. "
            "Do not repeat truncated reads. Use grep/find to locate relevant text, or continue "
            f"with offset/limit pagination from next_offset={next_offset} for {path}."
        )

    def collect_tool_results(self, results: list[ToolResultMessage]) -> None:
        for result in results:
            tool_call_id = result.tool_call_id.strip()
            if tool_call_id:
                if tool_call_id not in self.seen_tool_call_ids:
                    self.counters.tool_calls += 1
                    self.seen_tool_call_ids.add(tool_call_id)
            else:
                self.counters.tool_calls += 1
            self.affected_paths.update(result.affected_paths)
            if result.workspace_changed:
                self.workspace_changed = True
                self.verification_status = "stale"
            if result.status == "approval_required":
                if tool_call_id:
                    self.pending_approval_tool_call_ids.add(tool_call_id)
            elif tool_call_id:
                self.pending_approval_tool_call_ids.discard(tool_call_id)
            if result.status == "cancelled":
                self.cancelled = True
                self.verification_status = "cancelled"
            if result.error_code == "tool_not_found":
                self.tool_unavailable = True
            if result.is_error:
                self.last_error = _tool_error(result)
            if (
                result.status == "success"
                and not result.is_error
                and result.tool_name in _PLAN_EXPLORATION_TOOL_NAMES
            ):
                evidence_id = result.tool_call_id or (
                    f"{result.tool_name}:{len(self.plan_exploration_tool_call_ids)}"
                )
                self.plan_exploration_tool_call_ids.add(evidence_id)
            if result.verification:
                status = _verification_status(result.verification.get("status"))
                self.verification.append(
                    RunVerification(
                        tool_call_id=result.tool_call_id,
                        tool_name=result.tool_name,
                        status=status,
                        command=_optional_str(result.verification.get("command")),
                        exit_code=_optional_int(result.verification.get("exit_code")),
                        summary=str(result.verification.get("summary", "")),
                    )
                )
                self.verification_status = _signal_verification_status(status)
        self.approval_required = bool(self.pending_approval_tool_call_ids)

    def observe_plan_execution(
        self,
        results: list[ToolResultMessage],
        *,
        in_progress_item_id: str | None,
    ) -> None:
        if in_progress_item_id is None:
            self.reset_plan_failures()
            return
        if self.qualified_plan_failure_item_id != in_progress_item_id:
            self.qualified_plan_failure_item_id = in_progress_item_id
            self.qualified_plan_failure_count = 0
        if any(_is_successful_verification(result) for result in results):
            self.qualified_plan_failure_count = 0
            return
        if any(_is_qualified_plan_failure(result) for result in results):
            self.qualified_plan_failure_count += 1

    def reset_plan_failures(self) -> None:
        self.qualified_plan_failure_item_id = None
        self.qualified_plan_failure_count = 0

    def summary(self) -> RunSignalsSummary:
        return RunSignalsSummary(
            workspace_changed=self.workspace_changed,
            affected_paths=sorted(self.affected_paths),
            verification_status=self.verification_status,
            last_error=self.last_error,
            approval_required=self.approval_required,
            tool_unavailable=self.tool_unavailable,
            cancelled=self.cancelled,
            counters=AgentRunCounters(
                model_attempts=self.counters.model_attempts,
                tool_iterations=self.counters.tool_iterations,
                tool_calls=self.counters.tool_calls,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "counters": {
                "model_attempts": self.counters.model_attempts,
                "tool_iterations": self.counters.tool_iterations,
                "tool_calls": self.counters.tool_calls,
            },
            "workspace_changed": self.workspace_changed,
            "affected_paths": sorted(self.affected_paths),
            "verification": [asdict(item) for item in self.verification],
            "verification_status": self.verification_status,
            "last_error": dict(self.last_error) if isinstance(self.last_error, dict) else None,
            "approval_required": self.approval_required,
            "tool_unavailable": self.tool_unavailable,
            "cancelled": self.cancelled,
            "last_tool_fingerprint": self.last_tool_fingerprint,
            "repeated_tool_calls": self.repeated_tool_calls,
            "last_truncated_read_fingerprint": self.last_truncated_read_fingerprint,
            "repeated_truncated_reads": self.repeated_truncated_reads,
            "truncated_read_recovery_sent": self.truncated_read_recovery_sent,
            "seen_tool_call_ids": sorted(self.seen_tool_call_ids),
            "pending_approval_tool_call_ids": sorted(self.pending_approval_tool_call_ids),
            "qualified_plan_failure_item_id": self.qualified_plan_failure_item_id,
            "qualified_plan_failure_count": self.qualified_plan_failure_count,
            "plan_exploration_tool_call_ids": sorted(self.plan_exploration_tool_call_ids),
        }

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        run_id: str,
        session_id: str | None,
    ) -> "RunState":
        if not isinstance(value, dict):
            return cls(run_id=run_id, session_id=session_id)
        counters = value.get("counters")
        counter_data = counters if isinstance(counters, dict) else {}
        state = cls(
            run_id=run_id,
            session_id=session_id,
            counters=AgentRunCounters(
                model_attempts=_non_negative_int(counter_data.get("model_attempts")),
                tool_iterations=_non_negative_int(counter_data.get("tool_iterations")),
                tool_calls=_non_negative_int(counter_data.get("tool_calls")),
            ),
            workspace_changed=bool(value.get("workspace_changed")),
            affected_paths=set(_string_list(value.get("affected_paths"))),
            verification=_verification_list(value.get("verification")),
            verification_status=_signal_status(value.get("verification_status")),
            last_error=(
                dict(value.get("last_error"))
                if isinstance(value.get("last_error"), dict)
                else None
            ),
            approval_required=bool(value.get("approval_required")),
            tool_unavailable=bool(value.get("tool_unavailable")),
            cancelled=bool(value.get("cancelled")),
            last_tool_fingerprint=_optional_text(value.get("last_tool_fingerprint")),
            repeated_tool_calls=_non_negative_int(value.get("repeated_tool_calls")),
            last_truncated_read_fingerprint=_optional_text(
                value.get("last_truncated_read_fingerprint")
            ),
            repeated_truncated_reads=_non_negative_int(value.get("repeated_truncated_reads")),
            truncated_read_recovery_sent=bool(value.get("truncated_read_recovery_sent")),
            seen_tool_call_ids=set(_string_list(value.get("seen_tool_call_ids"))),
            pending_approval_tool_call_ids=set(
                _string_list(value.get("pending_approval_tool_call_ids"))
            ),
            qualified_plan_failure_item_id=_optional_text(
                value.get("qualified_plan_failure_item_id")
            ),
            qualified_plan_failure_count=_non_negative_int(
                value.get("qualified_plan_failure_count")
            ),
            plan_exploration_tool_call_ids=set(
                _string_list(value.get("plan_exploration_tool_call_ids"))
            ),
        )
        state.approval_required = state.approval_required or bool(
            state.pending_approval_tool_call_ids
        )
        return state


def _tool_error(result: ToolResultMessage) -> dict[str, Any]:
    return {
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "status": result.status,
        "error_code": result.error_code,
    }


def _is_successful_verification(result: ToolResultMessage) -> bool:
    verification = result.verification
    return isinstance(verification, dict) and verification.get("status") == "passed"


def _is_qualified_plan_failure(result: ToolResultMessage) -> bool:
    verification = result.verification
    if isinstance(verification, dict) and verification.get("status") == "failed":
        return True
    if not result.is_error:
        return False
    code = str(result.error_code or "").strip().lower()
    if not code:
        return False
    ignored_prefixes = ("invalid_", "tool_not_found", "permission", "approval", "cancel")
    return not code.startswith(ignored_prefixes)


def _truncated_read_fingerprint(
    results: list[ToolResultMessage],
) -> tuple[str | None, dict[str, Any]]:
    for result in results:
        if result.tool_name != "read":
            continue
        metadata = dict(result.metadata)
        quality = metadata.get("output_quality")
        truncated = bool(metadata.get("truncated"))
        if isinstance(quality, dict):
            truncated = truncated or bool(quality.get("truncated"))
        if not truncated:
            continue
        paths = metadata.get("read_paths")
        path = str(paths[0]) if isinstance(paths, list) and paths else None
        path = path or _optional_text(metadata.get("path")) or _optional_text(metadata.get("read_path"))
        if not path:
            continue
        start = metadata.get("actual_start_line", metadata.get("start_line"))
        next_offset = metadata.get(
            "next_offset",
            metadata.get("actual_end_line", metadata.get("end_line")),
        )
        metadata.setdefault("path", path)
        return f"{path}:{start}:{next_offset}", metadata
    return None, {}


def _verification_status(value: object) -> RunVerificationStatus:
    if value in {"passed", "failed", "cancelled", "unknown"}:
        return cast(RunVerificationStatus, value)
    return "unknown"


def _signal_verification_status(status: RunVerificationStatus) -> RunSignalsVerificationStatus:
    if status in {"passed", "failed", "cancelled"}:
        return cast(RunSignalsVerificationStatus, status)
    return "unknown"


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _non_negative_int(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _optional_text(item)
        if text is not None:
            result.append(text)
    return result


def _verification_list(value: object) -> list[RunVerification]:
    if not isinstance(value, list):
        return []
    result: list[RunVerification] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        result.append(
            RunVerification(
                tool_call_id=_optional_text(item.get("tool_call_id")) or "",
                tool_name=_optional_text(item.get("tool_name")) or "",
                status=_verification_status(item.get("status")),
                command=_optional_str(item.get("command")),
                exit_code=_optional_int(item.get("exit_code")),
                summary=str(item.get("summary") or ""),
            )
        )
    return result


def _signal_status(value: object) -> RunSignalsVerificationStatus:
    if value in {"unknown", "passed", "failed", "cancelled", "stale"}:
        return cast(RunSignalsVerificationStatus, value)
    return "unknown"


__all__ = ["RunState", "new_run_id"]
