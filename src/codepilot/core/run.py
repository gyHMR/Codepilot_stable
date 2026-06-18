from __future__ import annotations

"""Mutable state collected while one Agent Run is executing."""

import json
import uuid
from dataclasses import dataclass, field
from typing import cast

from codepilot.protocols import (
    AgentRunCounters,
    AgentRunResult,
    AgentRunStatus,
    AgentRunStopReason,
    AssistantMessage,
    ErrorInfo,
    RunVerification,
    RunVerificationStatus,
    ToolCall,
    ToolResultMessage,
)

from .types import AgentMessage


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


@dataclass
class RunState:
    run_id: str
    session_id: str | None
    counters: AgentRunCounters = field(default_factory=AgentRunCounters)
    affected_paths: set[str] = field(default_factory=set)
    workspace_changed: bool = False
    verification: list[RunVerification] = field(default_factory=list)
    last_tool_fingerprint: str | None = None
    repeated_tool_calls: int = 0

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

    def collect_tool_results(self, results: list[ToolResultMessage]) -> None:
        self.counters.tool_calls += len(results)
        for result in results:
            self.affected_paths.update(result.affected_paths)
            if result.workspace_changed:
                self.workspace_changed = True
            if result.verification:
                verification = result.verification
                self.verification.append(
                    RunVerification(
                        tool_call_id=result.tool_call_id,
                        tool_name=result.tool_name,
                        status=_verification_status(verification.get("status")),
                        command=_optional_str(verification.get("command")),
                        exit_code=_optional_int(verification.get("exit_code")),
                        summary=str(verification.get("summary", "")),
                    )
                )

    def result(
        self,
        *,
        status: AgentRunStatus,
        stop_reason: AgentRunStopReason,
        messages: list[AgentMessage],
        final_message: AssistantMessage | None,
        error: ErrorInfo | None = None,
    ) -> AgentRunResult:
        return AgentRunResult(
            run_id=self.run_id,
            session_id=self.session_id,
            status=status,
            stop_reason=stop_reason,
            counters=self.counters,
            messages=list(messages),
            final_message=final_message,
            error=error,
            affected_paths=sorted(self.affected_paths),
            workspace_changed=self.workspace_changed,
            verification=list(self.verification),
        )


def _verification_status(value: object) -> RunVerificationStatus:
    if value in {"passed", "failed", "cancelled", "unknown"}:
        return cast(RunVerificationStatus, value)
    return "unknown"


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
