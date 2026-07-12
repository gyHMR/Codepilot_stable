from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codepilot.protocols import AgentEvent, ToolCall, ToolResultMessage
from codepilot.tools.contracts import ToolExecutionRequest, ToolPort
from codepilot.tools.registry import ToolCatalogSnapshot
from codepilot.tools.results import ToolResult, to_tool_result_message as project_tool_result_message

from .contracts import WorkspaceEffects


async def execute_tool_turn(
    *,
    run_id: str,
    session_id: str,
    current_mode: str,
    tools: ToolPort,
    tool_calls: list[ToolCall],
    catalog_snapshot: ToolCatalogSnapshot,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> list[ToolResult]:
    """Execute one model tool-call batch through the immutable catalog it observed."""

    entries = {item.spec.name: item for item in catalog_snapshot.entries}
    requests = [
        ToolExecutionRequest(
            run_id=run_id,
            session_id=session_id,
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=dict(tool_call.arguments),
            mode=_tool_mode(current_mode),
            registration_id=(
                entries[tool_call.name].registration_id
                if tool_call.name in entries
                else "registration_missing"
            ),
        )
        for tool_call in tool_calls
    ]
    for tool_call in tool_calls:
        _emit_tool_start(emit, tool_call)
    results = list(await tools.execute_batch(requests))
    for result in results:
        if emit is not None:
            emit(tool_end_event(result))
    return results


def approval_results(results: list[ToolResult]) -> list[ToolResult]:
    return [item for item in results if item.status == "approval_required"]


def to_tool_result_message(result: ToolResult) -> ToolResultMessage:
    return project_tool_result_message(result)


def workspace_effects(results: list[ToolResult]) -> WorkspaceEffects:
    paths: list[str] = []
    changed = False
    for result in results:
        for effect in result.effects:
            if effect.resource.uri.startswith("workspace:///"):
                path = effect.resource.uri.removeprefix("workspace:///") or "."
                paths.append(path)
            if effect.kind in {"filesystem_write", "filesystem_delete"}:
                changed = True
    return WorkspaceEffects(
        affected_paths=tuple(dict.fromkeys(paths)),
        changed=changed,
    )


def merge_workspace_effects(
    first: WorkspaceEffects,
    second: WorkspaceEffects,
) -> WorkspaceEffects:
    paths = tuple(dict.fromkeys([*first.affected_paths, *second.affected_paths]))
    return WorkspaceEffects(affected_paths=paths, changed=first.changed or second.changed)


def verification(results: list[ToolResult]) -> list[Any]:
    items: list[Any] = []
    for result in results:
        value = result.data.get("verification")
        if isinstance(value, (list, tuple)):
            items.extend(value)
        elif value is not None:
            items.append(value)
    return items


def tool_end_event(result: ToolResult) -> AgentEvent:
    approval = result.approval
    error = result.error
    effects = workspace_effects([result])
    return {
        "type": _tool_event_type(result),
        "toolCallId": result.tool_call_id,
        "toolName": result.tool_name,
        "status": result.status,
        "isError": result.status not in {"success", "approval_required", "user_input_required"},
        "approved": result.status not in {"approval_required", "denied"},
        "approvalId": approval.approval_id if approval is not None else None,
        "errorReason": error.code if error is not None else None,
        "affectedPaths": list(effects.affected_paths),
        "workspaceChanged": effects.changed,
        "reason": approval.reason if approval is not None else None,
        "riskLevel": approval.risk if approval is not None else None,
        "result": {
            "content": list(result.content),
            "data": dict(result.data),
            "effects": [
                {
                    "kind": effect.kind,
                    "resource": effect.resource.uri,
                    "operation": effect.operation,
                    "status": effect.status,
                    "certainty": effect.certainty,
                }
                for effect in result.effects
            ],
            "registrationId": result.registration_id,
            "outputValidation": result.output_validation,
            "contentTrust": result.content_trust,
        },
    }


def _tool_event_type(result: ToolResult) -> str:
    if result.status == "success":
        return "tool_completed"
    if result.status in {
        "approval_required",
        "user_input_required",
        "denied",
        "cancelled",
        "interrupted",
    }:
        return "tool_interrupted"
    return "tool_failed"


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
        }
    )


def _tool_mode(mode: str) -> str:
    return "plan" if mode in {"read", "plan"} else "execute"


__all__ = [
    "approval_results",
    "execute_tool_turn",
    "merge_workspace_effects",
    "to_tool_result_message",
    "tool_end_event",
    "verification",
    "workspace_effects",
]
