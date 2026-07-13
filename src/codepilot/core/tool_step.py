from __future__ import annotations

from collections.abc import Callable
from typing import Any

from codepilot.protocols import AgentEvent, ToolCall, tool_mode_for_run_mode
from codepilot.tools.contracts import ToolExecutionRequest, ToolPort
from codepilot.tools.registry import ToolCatalogSnapshot
from codepilot.tools.results import ToolResult, workspace_effect_summary

from .contracts import WorkspaceEffects


async def execute_tool_turn(
    *,
    run_id: str,
    session_id: str,
    current_mode: str,
    tools: ToolPort,
    tool_calls: list[ToolCall],
    catalog_snapshot: ToolCatalogSnapshot,
    deadline_at_ms: int | None = None,
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
            raw_arguments=tool_call.raw_arguments,
            argument_parse_error=(
                str(tool_call.metadata.get("argument_parse_error"))
                if tool_call.metadata.get("argument_parse_error")
                else None
            ),
            mode=tool_mode_for_run_mode(current_mode),
            registration_id=(
                entries[tool_call.name].registration_id
                if tool_call.name in entries
                else "registration_missing"
            ),
            deadline_at_ms=deadline_at_ms,
        )
        for tool_call in tool_calls
    ]
    results = list(await tools.execute_batch(requests))
    for result in results:
        if result.error is None or result.error.code != "tool.batch.interrupted":
            matching_call = next(
                (item for item in tool_calls if item.id == result.tool_call_id),
                None,
            )
            if matching_call is not None:
                _emit_tool_start(emit, matching_call)
        if emit is not None:
            emit(tool_end_event(result))
    return results


def approval_results(results: list[ToolResult]) -> list[ToolResult]:
    return [item for item in results if item.status == "approval_required"]


def workspace_effects(results: list[ToolResult]) -> WorkspaceEffects:
    paths: list[str] = []
    changed = False
    for result in results:
        uris, result_changed = workspace_effect_summary(result.effects)
        paths.extend(uri.removeprefix("workspace:///") or "." for uri in uris)
        changed = changed or result_changed
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
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "status": result.status,
        "is_error": result.status not in {"success", "approval_required", "user_input_required"},
        "approved": result.status not in {"approval_required", "denied"},
        "approval_id": approval.approval_id if approval is not None else None,
        "error_reason": error.code if error is not None else None,
        "affected_paths": list(effects.affected_paths),
        "workspace_changed": effects.changed,
        "reason": approval.reason if approval is not None else None,
        "risk_level": approval.risk if approval is not None else None,
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
            "registration_id": result.registration_id,
            "output_validation": result.output_validation,
            "content_trust": result.content_trust,
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
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.name,
            "args": dict(tool_call.arguments),
        }
    )


__all__ = [
    "approval_results",
    "execute_tool_turn",
    "merge_workspace_effects",
    "tool_end_event",
    "verification",
    "workspace_effects",
]
