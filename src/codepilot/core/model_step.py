from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from codepilot.llm.ports import (
    LLMCompleted,
    LLMCorrelation,
    LLMFailed,
    LLMReasoningDelta,
    LLMRequest,
    LLMTextDelta,
    LLMToolCallDelta,
)
from codepilot.protocols import AssistantMessage, Message, Tool, Usage, tool_mode_for_run_mode
from codepilot.tools.codecs import json_value
from codepilot.tools.registry import ToolCatalogSnapshot

from .contracts import CoreDirective, CorePorts, CoreRunInput
from .observations import ModelObservation
from .state import FailureRecord


@dataclass(frozen=True)
class ModelActionResult:
    observation: ModelObservation
    usage: Usage | None = None
    catalog_snapshot: ToolCatalogSnapshot | None = None


async def call_model_once(
    input: CoreRunInput,
    ports: CorePorts,
    messages: tuple[Message, ...],
    directive: CoreDirective,
    *,
    observation_id: str,
) -> ModelActionResult:
    """Prepare context and perform exactly one model action."""

    snapshot = _core_tool_catalog(input, ports)
    tools = _tools_from_snapshot(snapshot)
    request_data: dict[str, Any] = {
        **dict(input.context_seed),
        "run_id": input.run_id,
        "mode": input.mode,
        "model": input.model,
        "messages": list(messages),
        "tools": tools,
        "directive": {
            "code": directive.code,
            "constraints": list(directive.constraints),
            "evidence_refs": list(directive.evidence_refs),
        },
        "context": dict(input.context_seed),
    }
    prepared = ports.context.prepare(request_data)
    if inspect.isawaitable(prepared):
        prepared = await prepared
    if prepared is not None:
        if not isinstance(prepared, Mapping):
            raise TypeError("ContextPort.prepare must return a mapping or None")
        request_data.update(dict(prepared))

    request = LLMRequest(
        model=request_data.get("model", input.model),
        messages=tuple(request_data.get("messages", messages)),
        system_prompt=str(request_data.get("system_prompt", "")),
        tools=tuple(request_data.get("tools", tools)),
        correlation=LLMCorrelation(
            run_id=input.run_id,
            session_id=_optional_text(input.context_seed.get("session_id")) or "",
        ),
    )
    assistant: AssistantMessage | None = None
    usage = None
    async for event in ports.model.stream(request):
        if isinstance(event, LLMFailed):
            return ModelActionResult(
                observation=ModelObservation(
                    observation_id=observation_id,
                    status="failed",
                    error=_model_failure(event.error, observation_id),
                ),
                catalog_snapshot=snapshot,
            )
        if isinstance(event, LLMTextDelta):
            await _emit_core_live_event(
                ports,
                {
                    "type": "message_update",
                    "assistant_message_event": {
                        "type": "text_delta",
                        "delta": event.text,
                    },
                },
            )
            continue
        if isinstance(event, LLMReasoningDelta):
            await _emit_core_live_event(
                ports,
                {
                    "type": "message_update",
                    "assistant_message_event": {
                        "type": "reasoning_delta",
                        "delta": event.text,
                    },
                },
            )
            continue
        if isinstance(event, LLMToolCallDelta):
            await _emit_core_live_event(
                ports,
                {
                    "type": "message_update",
                    "assistant_message_event": {
                        "type": "tool_call_delta",
                        "toolCall": event.tool_call,
                    },
                },
            )
            continue
        if isinstance(event, LLMCompleted):
            assistant = event.message
            usage = event.usage

    if assistant is None:
        return ModelActionResult(
            observation=ModelObservation(
                observation_id=observation_id,
                status="failed",
                error=FailureRecord(
                    code="llm.stream_incomplete",
                    source="model",
                    message="Model stream ended without a completion event",
                    recoverable=False,
                    evidence_refs=(observation_id,),
                ),
            ),
            catalog_snapshot=snapshot,
        )
    return ModelActionResult(
        observation=ModelObservation(
            observation_id=observation_id,
            message=assistant,
        ),
        usage=usage,
        catalog_snapshot=snapshot,
    )


def _core_tool_catalog(
    input: CoreRunInput,
    ports: CorePorts,
) -> ToolCatalogSnapshot | None:
    if ports.tools is None or bool(input.context_seed.get("suppress_tools", False)):
        return None
    return ports.tools.catalog_snapshot(mode=tool_mode_for_run_mode(input.mode))


def _tools_from_snapshot(snapshot: ToolCatalogSnapshot | None) -> list[Tool]:
    if snapshot is None:
        return []
    return [
        Tool(
            name=item.spec.name,
            description=item.spec.description,
            parameters=json_value(item.spec.input_schema),
        )
        for item in snapshot.entries
    ]


def _model_failure(error: object, observation_id: str) -> FailureRecord:
    if isinstance(error, Mapping):
        code = _optional_text(error.get("code")) or "model.call_failed"
        message = _optional_text(error.get("message")) or code
        recoverable = bool(error.get("retryable", False))
    else:
        code = _optional_text(getattr(error, "code", None)) or "model.call_failed"
        message = _optional_text(getattr(error, "message", None)) or str(error)
        recoverable = bool(getattr(error, "retryable", False))
    return FailureRecord(
        code=code,
        source="model",
        message=message or code,
        recoverable=recoverable,
        evidence_refs=(observation_id,),
    )


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


async def _emit_core_live_event(
    ports: CorePorts,
    event: Mapping[str, object],
) -> None:
    if ports.live_events is None:
        return
    try:
        result = ports.live_events(event)
        if inspect.isawaitable(result):
            await result
    except Exception:
        return


__all__ = ["ModelActionResult", "call_model_once"]
