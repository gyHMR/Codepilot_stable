from __future__ import annotations

"""Pure rules for Runtime tool-approval recovery.

RuntimeService owns the in-memory approval table and the user-facing
transaction.  This module owns the small, testable rules that are independent
from session orchestration: extracting pending approvals from a run result and
turning approval decisions or tool results into normalized ToolResultMessage
objects.
"""

import time
from dataclasses import dataclass

from codepilot.protocols import (
    AgentRunResult,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
)
from codepilot.tools import AgentToolResult

APPROVAL_DECISION_APPROVE_ALIASES = frozenset(
    {"approve", "approved", "allow", "yes", "y"}
)
APPROVAL_DECISION_DENY_ALIASES = frozenset(
    {"deny", "denied", "reject", "rejected", "no", "n"}
)


@dataclass(frozen=True)
class PendingApproval:
    """A resumable approval paired with the assistant ToolCall that produced it."""

    approval_id: str
    session_id: str
    run_id: str
    assistant_message: AssistantMessage
    tool_call: ToolCall
    reason: str = ""

    def __post_init__(self) -> None:
        _require_text(self.approval_id, "approval_id")
        _require_text(self.session_id, "session_id")
        _require_text(self.run_id, "run_id")
        _require_text(self.tool_call.id, "tool_call.id")


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")


def normalize_approval_decision(decision: object) -> str | None:
    """Normalize user-facing approval text to the runtime action vocabulary.

    The runtime approval transaction only has two actions: ``approve`` and
    ``deny``.  Interface-specific wording may vary, so aliases are accepted at
    this boundary and invalid values return ``None`` for the service layer to
    translate into its public error type.
    """

    value = str(decision).strip().lower() if decision is not None else ""
    if value in APPROVAL_DECISION_APPROVE_ALIASES:
        return "approve"
    if value in APPROVAL_DECISION_DENY_ALIASES:
        return "deny"
    return None


def denied_tool_result(approval: PendingApproval) -> ToolResultMessage:
    """Build the tool result inserted when the user rejects a pending approval."""

    return ToolResultMessage(
        tool_call_id=approval.tool_call.id,
        tool_name=approval.tool_call.name,
        content=[TextContent(text="Tool execution denied by user")],
        details={
            "status": "denied",
            "reason": "user_denied",
            "approval_id": approval.approval_id,
        },
        is_error=True,
        status="denied",
        approved=False,
        approval_id=approval.approval_id,
        error_code="user_denied",
        timestamp=int(time.time() * 1000),
        metadata={
            "approval_resume": {
                "approval_id": approval.approval_id,
                "decision": "denied",
            }
        },
    )


def to_tool_result_message(result: AgentToolResult) -> ToolResultMessage:
    """Convert a direct AgentToolResult into the message shape stored in context."""

    return ToolResultMessage(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        content=list(result.content),
        details=result.details,
        is_error=result.is_error,
        status=result.status,
        approved=result.approved,
        approval_id=result.approval_id,
        error_code=result.error_code,
        exit_code=result.exit_code,
        affected_paths=list(result.affected_paths),
        workspace_changed=result.workspace_changed,
        diff_summary=result.diff_summary,
        verification=dict(result.verification) if result.verification else None,
        timestamp=int(time.time() * 1000),
        metadata=dict(result.metadata),
    )


def build_pending_approvals(
    session_id: str,
    result: AgentRunResult,
) -> list[PendingApproval]:
    """Extract approval requests that can be resumed later by RuntimeService.

    A pending approval is valid only when the ``approval_required`` tool result
    can be paired with the assistant ``ToolCall`` that requested it.  Orphaned
    tool results are ignored because Runtime would not know which tool and
    arguments to execute after the user approves.
    """

    if result.status != "waiting_approval":
        return []

    assistant_by_tool_call: dict[str, AssistantMessage] = {}
    tool_calls: dict[str, ToolCall] = {}
    for message in result.messages:
        if not isinstance(message, AssistantMessage):
            continue
        for block in message.content:
            if isinstance(block, ToolCall) and block.id:
                assistant_by_tool_call[block.id] = message
                tool_calls[block.id] = block

    approvals: list[PendingApproval] = []
    for message in result.messages:
        if (
            not isinstance(message, ToolResultMessage)
            or message.status != "approval_required"
            or not message.approval_id
        ):
            continue
        tool_call = tool_calls.get(message.tool_call_id)
        assistant = assistant_by_tool_call.get(message.tool_call_id)
        if tool_call is None or assistant is None:
            continue
        reason = ""
        if isinstance(message.details, dict):
            raw_reason = message.details.get("reason") or message.details.get("policy_reason")
            reason = raw_reason if isinstance(raw_reason, str) else ""
        approvals.append(
            PendingApproval(
                approval_id=message.approval_id,
                session_id=session_id,
                run_id=result.run_id,
                assistant_message=assistant,
                tool_call=tool_call,
                reason=reason,
            )
        )
    return approvals


__all__ = [
    "PendingApproval",
    "build_pending_approvals",
    "denied_tool_result",
    "normalize_approval_decision",
    "to_tool_result_message",
]
