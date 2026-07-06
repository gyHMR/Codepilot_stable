from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from codepilot.protocols import (
    AgentEvent,
    AssistantMessage,
    Message,
    Tool,
    ToolCall,
    ToolHookContextSnapshot,
    ToolResultMessage,
)
from codepilot.tools.ports import ToolInvocation, ToolObservation, ToolPort

from .contracts import WorkspaceEffects


async def execute_tool_turn(
    *,
    run_id: str,
    session_id: str | None = None,
    assistant_message: AssistantMessage | None = None,
    messages: list[Message] | None = None,
    system_prompt: str = "",
    available_tools: list[Tool] | None = None,
    task_signal: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tools: ToolPort,
    tool_calls: list[ToolCall],
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> list[ToolObservation]:
    """Execute the tool calls requested by one assistant message."""

    observations: list[ToolObservation] = []
    context = ToolHookContextSnapshot(
        run_id=run_id,
        session_id=session_id,
        system_prompt=system_prompt,
        messages=tuple(messages or ()),
        tools=tuple(available_tools or ()),
        task_signal=dict(task_signal or {}),
        metadata=dict(metadata or {}),
    )
    for tool_call in tool_calls:
        if emit is not None:
            emit(
                {
                    "type": "tool_execution_start",
                    "toolCallId": tool_call.id,
                    "toolName": tool_call.name,
                    "args": dict(tool_call.arguments),
                    "arguments": dict(tool_call.arguments),
                }
            )
        observation = await tools.execute(
            ToolInvocation(
                run_id=run_id,
                tool_call_id=tool_call.id,
                name=tool_call.name,
                arguments=dict(tool_call.arguments),
                assistant_message=assistant_message,
                context=context,
            )
        )
        observations.append(observation)
        if emit is not None:
            emit(_tool_end_event(observation))
        if observation.status == "approval_required" and observation.interruption is not None:
            break
    return observations


def approval_observations(
    observations: list[ToolObservation],
) -> list[ToolObservation]:
    return [
        observation
        for observation in observations
        if observation.status == "approval_required" and observation.interruption is not None
    ]


def to_tool_result_message(
    observation: ToolObservation,
    *,
    approval_id: str | None = None,
    approved: bool = True,
) -> ToolResultMessage:
    metadata = dict(observation.metadata)
    resolved_approval_id = approval_id or metadata.get("approval_id")
    status = observation.status if observation.status else "success"
    return ToolResultMessage(
        tool_call_id=observation.tool_call_id,
        tool_name=observation.name,
        content=list(observation.content),
        status=status,  # type: ignore[arg-type]
        is_error=status != "success",
        approved=approved,
        approval_id=str(resolved_approval_id) if resolved_approval_id else None,
        affected_paths=list(observation.affected_paths),
        workspace_changed=observation.workspace_changed,
        verification=_verification_payload(observation),
        details=metadata.get("details"),
        metadata=metadata,
    )


def workspace_effects(observations: list[ToolObservation]) -> WorkspaceEffects:
    paths: list[str] = []
    changed = False
    for observation in observations:
        paths.extend(observation.affected_paths)
        changed = changed or observation.workspace_changed
    return WorkspaceEffects(affected_paths=tuple(dict.fromkeys(paths)), changed=changed)


def merge_workspace_effects(
    first: WorkspaceEffects,
    second: WorkspaceEffects,
) -> WorkspaceEffects:
    paths = tuple(dict.fromkeys([*first.affected_paths, *second.affected_paths]))
    return WorkspaceEffects(affected_paths=paths, changed=first.changed or second.changed)


def verification(observations: list[ToolObservation]) -> list[Any]:
    items: list[Any] = []
    for observation in observations:
        items.extend(observation.verification)
    return items


def _verification_payload(observation: ToolObservation) -> dict[str, Any] | None:
    if not observation.verification:
        return None
    items = [asdict(item) for item in observation.verification]
    if len(items) == 1:
        return items[0]
    status = "unknown"
    if any(item.get("status") == "failed" for item in items):
        status = "failed"
    elif any(item.get("status") == "passed" for item in items):
        status = "passed"
    return {"status": status, "items": items}


def _tool_end_event(observation: ToolObservation) -> AgentEvent:
    metadata = dict(observation.metadata)
    approval_id = metadata.get("approval_id")
    approved = metadata.get("approved")
    if approved is None:
        approved = observation.status == "success"
    details = metadata.get("details")
    error_reason = metadata.get("error_code")
    if error_reason is None and isinstance(details, dict):
        error_reason = details.get("reason") or details.get("status")
    return {
        "type": "tool_execution_end",
        "toolCallId": observation.tool_call_id,
        "toolName": observation.name,
        "status": observation.status,
        "isError": observation.status != "success",
        "approved": approved,
        "approvalId": str(approval_id) if approval_id else None,
        "errorReason": str(error_reason) if error_reason else None,
        "affectedPaths": list(observation.affected_paths),
        "workspaceChanged": observation.workspace_changed,
        "verification": list(observation.verification),
        "result": {
            "content": list(observation.content),
            "metadata": metadata,
        },
    }
