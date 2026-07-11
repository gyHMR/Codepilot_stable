from __future__ import annotations

import asyncio
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
from codepilot.tools.contracts import (
    ToolCatalogView,
    ToolInvocation,
    ToolMetadata,
    ToolObservation,
    ToolPort,
)

from .contracts import WorkspaceEffects


async def execute_tool_turn(
    *,
    run_id: str,
    session_id: str | None = None,
    assistant_message: AssistantMessage | None = None,
    messages: list[Message] | None = None,
    system_prompt: str = "",
    available_tools: list[Tool] | None = None,
    current_mode: str = "build",
    run_signals: dict[str, Any] | None = None,
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
        run_signals=dict(run_signals or {}),
        metadata=dict(metadata or {}),
    )
    metadata_by_name = _catalog_metadata(tools, current_mode)
    index = 0
    while index < len(tool_calls):
        batch = _next_concurrent_batch(tool_calls, index, metadata_by_name)
        if len(batch) > 1:
            invocations = [
                _invocation_for(
                    tool_call,
                    run_id=run_id,
                    current_mode=current_mode,
                    assistant_message=assistant_message,
                    context=context,
                )
                for tool_call in batch
            ]
            for tool_call in batch:
                _emit_tool_start(emit, tool_call)
            batch_observations = list(
                await asyncio.gather(*(tools.execute(invocation) for invocation in invocations))
            )
            observations.extend(batch_observations)
            for observation in batch_observations:
                if emit is not None:
                    emit(_tool_end_event(observation))
            if approval_observations(batch_observations):
                break
            index += len(batch)
            continue

        tool_call = tool_calls[index]
        _emit_tool_start(emit, tool_call)
        observation = await tools.execute(
            _invocation_for(
                tool_call,
                run_id=run_id,
                current_mode=current_mode,
                assistant_message=assistant_message,
                context=context,
            )
        )
        observations.append(observation)
        if emit is not None:
            emit(_tool_end_event(observation))
        if observation.status == "approval_required" and observation.interruption is not None:
            break
        index += 1
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
        error_code=_optional_text(metadata.get("error_code")),
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
    interruption = observation.interruption
    approval_id = metadata.get("approval_id")
    if approval_id is None and interruption is not None:
        approval_id = interruption.approval_id
    approved = metadata.get("approved")
    if approved is None:
        approved = observation.status == "success"
    details = metadata.get("details")
    error_reason = metadata.get("error_code")
    if error_reason is None and isinstance(details, dict):
        error_reason = details.get("reason") or details.get("status")
    return {
        "type": _tool_event_type(observation),
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
        "reason": interruption.reason if interruption is not None else None,
        "riskLevel": (
            str(getattr(interruption.risk, "level", "unknown"))
            if interruption is not None
            else None
        ),
        "result": {
            "content": list(observation.content),
            "metadata": metadata,
        },
    }


def _tool_event_type(observation: ToolObservation) -> str:
    if observation.status == "success":
        return "tool_completed"
    if observation.status in {"approval_required", "denied", "cancelled"}:
        return "tool_interrupted"
    return "tool_failed"


def _invocation_for(
    tool_call: ToolCall,
    *,
    run_id: str,
    current_mode: str,
    assistant_message: AssistantMessage | None,
    context: ToolHookContextSnapshot,
) -> ToolInvocation:
    return ToolInvocation(
        run_id=run_id,
        tool_call_id=tool_call.id,
        name=tool_call.name,
        arguments=dict(tool_call.arguments),
        current_mode=current_mode,
        assistant_message=assistant_message,
        context=context,
    )


def _emit_tool_start(
    emit: Callable[[dict[str, Any]], None] | None,
    tool_call: ToolCall,
) -> None:
    if emit is None:
        return
    emit(
        {
            "type": "tool_started",
            "toolCallId": tool_call.id,
            "toolName": tool_call.name,
            "args": dict(tool_call.arguments),
            "arguments": dict(tool_call.arguments),
        }
    )


def _catalog_metadata(tools: ToolPort, current_mode: str) -> dict[str, ToolMetadata]:
    try:
        catalog = tools.catalog(current_mode)
    except TypeError:
        catalog = tools.catalog()
    if not isinstance(catalog, ToolCatalogView):
        return {}
    return {item.metadata.name: item.metadata for item in catalog.items}


def _next_concurrent_batch(
    tool_calls: list[ToolCall],
    start: int,
    metadata_by_name: dict[str, ToolMetadata],
) -> list[ToolCall]:
    first = tool_calls[start]
    if not _can_run_concurrently(first, metadata_by_name):
        return [first]
    batch = [first]
    for tool_call in tool_calls[start + 1 :]:
        if not _can_run_concurrently(tool_call, metadata_by_name):
            break
        batch.append(tool_call)
    return batch


def _can_run_concurrently(
    tool_call: ToolCall,
    metadata_by_name: dict[str, ToolMetadata],
) -> bool:
    metadata = metadata_by_name.get(tool_call.name)
    return bool(
        metadata is not None
        and metadata.read_only
        and metadata.concurrency_safe
        and not metadata.exclusive
    )


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
