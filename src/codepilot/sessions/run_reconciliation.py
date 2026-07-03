from __future__ import annotations

# 新手导读：run reconciliation 负责把运行结果、事件和会话事实源对齐。
# 关注点：它帮助恢复或审计时判断哪些 run 已经完整落盘。

"""Reconcile session-owned evidence into core Agent run results.

The core Agent loop reports what happened while it was actively running.  Some
session workflows, such as resuming a user-approved tool call, perform work just
before the follow-up Agent run starts.  This module keeps that evidence merging
explicit and testable instead of hiding it inside ``AgentSession`` lifecycle
plumbing.
"""

from dataclasses import replace
from typing import cast

from codepilot.protocols import (
    AgentRunResult,
    Message,
    RunVerification,
    RunVerificationStatus,
    ToolResultMessage,
)


def verification_from_approved_tool_result(
    result: ToolResultMessage,
) -> RunVerification | None:
    """Build run-level verification evidence from an approved tool result."""

    if not result.verification:
        return None
    raw_status = result.verification.get("status")
    status: RunVerificationStatus = (
        cast(RunVerificationStatus, raw_status)
        if raw_status in {"passed", "failed", "cancelled", "unknown"}
        else "unknown"
    )
    raw_command = result.verification.get("command")
    raw_exit_code = result.verification.get("exit_code")
    return RunVerification(
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        status=status,
        command=raw_command if isinstance(raw_command, str) else None,
        exit_code=(
            raw_exit_code
            if isinstance(raw_exit_code, int) and not isinstance(raw_exit_code, bool)
            else None
        ),
        summary=str(result.verification.get("summary", "")),
    )


def merge_approved_tool_result(
    result: AgentRunResult,
    approved_tool_result: ToolResultMessage,
) -> AgentRunResult:
    """Merge approval-time tool evidence into the resumed run result.

    Approval recovery has two phases: Runtime executes the previously deferred
    tool, then Session resumes the model.  The resumed model run does not know
    that approval-time execution should count as part of the same user-visible
    run, so Session adds the tool result, affected paths, workspace flag, and
    verification evidence before persisting the final ``AgentRunResult``.
    """

    messages = list(result.messages)
    already_present = any(
        isinstance(message, ToolResultMessage)
        and message.tool_call_id == approved_tool_result.tool_call_id
        and message.approval_id == approved_tool_result.approval_id
        for message in messages
    )
    if not already_present:
        messages.insert(0, approved_tool_result)

    counters = replace(
        result.counters,
        tool_calls=result.counters.tool_calls + (0 if already_present else 1),
    )
    affected_paths = sorted(
        {
            *result.affected_paths,
            *[
                str(path)
                for path in approved_tool_result.affected_paths
                if str(path)
            ],
        }
    )
    verification = list(result.verification)
    approved_verification = verification_from_approved_tool_result(
        approved_tool_result
    )
    if approved_verification is not None:
        duplicate_verification = any(
            item.tool_call_id == approved_verification.tool_call_id
            and item.tool_name == approved_verification.tool_name
            and item.command == approved_verification.command
            for item in verification
        )
        if not duplicate_verification:
            verification.append(approved_verification)

    return AgentRunResult(
        run_id=result.run_id,
        session_id=result.session_id,
        status=result.status,
        stop_reason=result.stop_reason,
        counters=counters,
        messages=messages,
        final_message=result.final_message,
        error=result.error,
        affected_paths=affected_paths,
        workspace_changed=bool(
            result.workspace_changed or approved_tool_result.workspace_changed is True
        ),
        verification=verification,
        task=result.task,
    )


def replace_pending_tool_result(
    messages: list[Message],
    replacement: ToolResultMessage,
    *,
    approval_id: str | None = None,
) -> tuple[list[Message], bool]:
    """Replace the pending approval result that belongs to ``replacement``.

    Runtime may know the explicit approval id, but older or synthetic flows can
    only identify the pending result by tool call id.  Keep both rules in one
    function so live context rewriting and future persistence tools agree on
    what "the approval result being resolved" means.
    """

    updated = list(messages)
    for index, message in enumerate(updated):
        if not isinstance(message, ToolResultMessage):
            continue
        approval_matches = (
            bool(approval_id)
            and message.approval_id == approval_id
            and message.status == "approval_required"
        )
        call_matches = (
            message.tool_call_id == replacement.tool_call_id
            and message.status == "approval_required"
        )
        if not approval_matches and not call_matches:
            continue
        updated[index] = replacement
        return updated, True
    return updated, False


__all__ = [
    "merge_approved_tool_result",
    "replace_pending_tool_result",
    "verification_from_approved_tool_result",
]
