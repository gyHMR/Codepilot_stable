from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import inspect
from typing import Any

from codepilot.protocols import PLAN_TOOL_NAMES, ToolCall, ToolResultMessage, tool_mode_for_run_mode
from codepilot.tools.contracts import (
    ToolBatchPreparation,
    ToolExecutionRequest,
)
from codepilot.tools.registry import ToolCatalogSnapshot
from codepilot.tools.results import (
    TextContent,
    ToolError,
    ToolResult,
    to_tool_result_message,
    workspace_effect_summary,
)

from .contracts import CorePorts, CoreRunInput, ExecuteTools
from .commands import CoreCommand, core_command_from_mapping
from .commands import CommandResult
from .errors import CoreInvariantError


@dataclass(frozen=True)
class PreparedCoreToolBatch:
    calls: tuple[ToolCall, ...]
    preparation: ToolBatchPreparation


def prepare_core_tool_batch(
    input: CoreRunInput,
    ports: CorePorts,
    decision: ExecuteTools,
    catalog_snapshot: ToolCatalogSnapshot | None,
) -> PreparedCoreToolBatch:
    if ports.tools is None:
        return PreparedCoreToolBatch(
            calls=decision.calls,
            preparation=ToolBatchPreparation(
                results=unavailable_tool_results(decision.calls)
            ),
        )
    entries = {
        item.spec.name: item
        for item in (catalog_snapshot.entries if catalog_snapshot is not None else ())
    }
    requests = tuple(
        ToolExecutionRequest(
            run_id=input.run_id,
            session_id=input.session_id,
            tool_call_id=call.id,
            tool_name=call.name,
            arguments=dict(call.arguments),
            raw_arguments=call.raw_arguments,
            argument_parse_error=(
                str(call.metadata.get("argument_parse_error"))
                if call.metadata.get("argument_parse_error")
                else None
            ),
            mode=tool_mode_for_run_mode(input.mode),
            registration_id=(
                entries[call.name].registration_id
                if call.name in entries
                else "registration_missing"
            ),
        )
        for call in decision.calls
    )
    return PreparedCoreToolBatch(
        calls=decision.calls,
        preparation=ports.tools.prepare_batch(requests),
    )


async def execute_core_tool_batch(
    ports: CorePorts,
    prepared: PreparedCoreToolBatch,
) -> tuple[ToolResult, ...]:
    if ports.tools is None or prepared.preparation.batch_id is None:
        raise RuntimeError("Prepared Tool batch is not executable")
    for call in prepared.calls:
        await _emit_core_live_event(
            ports,
            {
                "type": "tool_started",
                "tool_call_id": call.id,
                "tool_name": call.name,
                "args": dict(call.arguments),
            },
        )
    results = await ports.tools.execute_prepared(prepared.preparation.batch_id)
    for result in results:
        await _emit_core_live_event(ports, tool_end_event(result))
    return tuple(results)


def project_final_tool_messages(
    results: tuple[ToolResult, ...] | list[ToolResult],
) -> tuple[ToolResultMessage, ...]:
    return tuple(
        to_tool_result_message(result)
        for result in results
        if result.status not in {"approval_required", "user_input_required"}
    )


def project_core_commands(
    results: tuple[ToolResult, ...] | list[ToolResult],
) -> tuple[CoreCommand, ...]:
    """Accept command payloads only from Core-owned Plan tool names."""

    commands: list[CoreCommand] = []
    for result in results:
        if result.status != "success" or result.tool_name not in PLAN_TOOL_NAMES:
            continue
        raw = result.data.get("core_command")
        if not isinstance(raw, Mapping):
            raise CoreInvariantError(
                f"Plan tool returned no Core command: {result.tool_name}"
            )
        try:
            command = core_command_from_mapping(raw)
        except (TypeError, ValueError) as exc:
            raise CoreInvariantError(
                f"Plan tool returned an invalid Core command: {result.tool_name}"
            ) from exc
        if command.command_id != result.tool_call_id:
            raise CoreInvariantError(
                f"Plan command id does not match ToolCall: {result.tool_call_id}"
            )
        commands.append(command)
    return tuple(commands)


def project_core_command_results(
    results: tuple[ToolResult, ...] | list[ToolResult],
    command_results: tuple[CommandResult, ...] | list[CommandResult],
) -> tuple[ToolResult, ...]:
    """Expose Reducer acceptance or rejection in the model-facing ToolResult."""

    by_id = {item.command_id: item for item in command_results}
    projected: list[ToolResult] = []
    for result in results:
        command_result = by_id.get(result.tool_call_id)
        if command_result is None:
            projected.append(result)
            continue
        message = (
            "Core applied the Plan command."
            if command_result.status == "applied"
            else f"Core rejected the Plan command: {command_result.reason}."
        )
        data = {
            **dict(result.data),
            "core_command_result": {
                "status": command_result.status,
                "reason": command_result.reason,
            },
        }
        projected.append(
            ToolResult(
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                status=(
                    result.status
                    if command_result.status == "applied"
                    else "error"
                ),
                content=(*result.content, TextContent(message)),
                data=data,
                effects=result.effects,
                error=(
                    result.error
                    if command_result.status == "applied"
                    else ToolError(
                        code=command_result.reason,
                        kind="validation",
                        message=message,
                        retryable=True,
                    )
                ),
                approval=result.approval,
                interaction=result.interaction,
                registration_id=result.registration_id,
                output_validation=result.output_validation,
                content_trust=result.content_trust,
            )
        )
    return tuple(projected)


def interrupted_tool_results(
    calls: tuple[ToolCall, ...] | list[ToolCall],
    *,
    code: str,
    message: str,
) -> tuple[ToolResult, ...]:
    return tuple(
        ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            status="interrupted",
            content=(TextContent(message),),
            error=ToolError(
                code=code,
                kind="interrupted",
                message=message,
                retryable=True,
            ),
            registration_id="core_settlement",
        )
        for call in calls
    )


def unavailable_tool_results(
    calls: tuple[ToolCall, ...] | list[ToolCall],
) -> tuple[ToolResult, ...]:
    return tuple(
        ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            status="error",
            content=(TextContent("No ToolExecutionPort is available for this run."),),
            error=ToolError(
                code="tool_not_found",
                kind="unavailable",
                message=f"Tool is unavailable: {call.name}",
                retryable=False,
            ),
            registration_id="registration_missing",
        )
        for call in calls
    )


async def _emit_core_live_event(ports: CorePorts, event: dict[str, Any]) -> None:
    if ports.live_events is None:
        return
    try:
        value = ports.live_events(event)
        if inspect.isawaitable(value):
            await value
    except Exception:
        return


def tool_end_event(result: ToolResult) -> dict[str, Any]:
    approval = result.approval
    error = result.error
    uris, workspace_changed = workspace_effect_summary(result.effects)
    affected_paths = [
        uri.removeprefix("workspace:///") or "."
        for uri in uris
    ]
    return {
        "type": _tool_event_type(result),
        "tool_call_id": result.tool_call_id,
        "tool_name": result.tool_name,
        "status": result.status,
        "is_error": result.status
        not in {"success", "approval_required", "user_input_required"},
        "approved": result.status not in {"approval_required", "denied"},
        "approval_id": approval.approval_id if approval is not None else None,
        "error_reason": error.code if error is not None else None,
        "affected_paths": affected_paths,
        "workspace_changed": workspace_changed,
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


__all__ = [
    "execute_core_tool_batch",
    "interrupted_tool_results",
    "PreparedCoreToolBatch",
    "prepare_core_tool_batch",
    "project_core_command_results",
    "project_core_commands",
    "project_final_tool_messages",
    "tool_end_event",
    "unavailable_tool_results",
]
