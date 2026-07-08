from __future__ import annotations

"""Mechanical facts collected during one agent run."""

import json
import uuid
from dataclasses import dataclass, field
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
        self.counters.tool_calls += len(results)
        for result in results:
            self.affected_paths.update(result.affected_paths)
            if result.workspace_changed:
                self.workspace_changed = True
                self.verification_status = "stale"
            if result.status == "approval_required":
                self.approval_required = True
            if result.status == "cancelled":
                self.cancelled = True
                self.verification_status = "cancelled"
            if result.error_code == "tool_not_found":
                self.tool_unavailable = True
            if result.is_error:
                self.last_error = _tool_error(result)
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


def _tool_error(result: ToolResultMessage) -> dict[str, Any]:
    return {
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "status": result.status,
        "error_code": result.error_code,
    }


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


__all__ = ["RunState", "new_run_id"]
