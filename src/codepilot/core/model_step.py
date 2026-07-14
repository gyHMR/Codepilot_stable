"""封装一次模型调用的请求构造、事件归并和失败分类。"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass

from codepilot.llm.ports import (
    LLMCompleted,
    LLMCorrelation,
    LLMFailed,
    LLMReasoningDelta,
    LLMRequest,
    LLMTextDelta,
    LLMToolCallDelta,
)
from codepilot.protocols import AssistantMessage, Message, Usage, tool_mode_for_run_mode
from codepilot.tools.registry import ToolCatalogSnapshot

from .contracts import (
    ContextPrepareRequest,
    ContextPurpose,
    CoreContextView,
    CoreDirective,
    CorePorts,
    CoreRunInput,
    ModelPurpose,
    PreparedModelContext,
)
from .observations import ModelObservation
from .state import CoreState, FailureRecord


@dataclass(frozen=True)
class ModelActionResult:
    """一次模型调用归并后的消息、用量和错误结果。"""
    observation: ModelObservation
    usage: Usage | None = None
    catalog_snapshot: ToolCatalogSnapshot | None = None


async def call_model_once(
    input: CoreRunInput,
    ports: CorePorts,
    messages: tuple[Message, ...],
    state: CoreState,
    purpose: ModelPurpose,
    directive: CoreDirective,
    *,
    observation_id: str,
) -> ModelActionResult:
    """Prepare context and perform exactly one model action."""

    snapshot = _core_tool_catalog(input, ports)
    prepared = ports.context.prepare(
        ContextPrepareRequest(
            session_id=input.session_id,
            run_id=input.run_id,
            purpose=_context_purpose(purpose),
            directive=_directive_text(directive),
            messages=messages,
            core_view=CoreContextView.from_state(state, input.mode),
            model=input.model,
            tool_catalog=snapshot,
            seed=input.context_seed,
        )
    )
    if inspect.isawaitable(prepared):
        prepared = await prepared
    if not isinstance(prepared, PreparedModelContext):
        raise TypeError("ContextPreparationPort.prepare must return PreparedModelContext")

    request = LLMRequest(
        model=input.model,
        messages=prepared.messages,
        system_prompt=prepared.system_prompt,
        tools=prepared.tools,
        correlation=LLMCorrelation(
            run_id=input.run_id,
            session_id=input.session_id,
            purpose=_context_purpose(purpose),
        ),
    )
    assistant: AssistantMessage | None = None
    usage = None
    attempts = 1
    async for event in ports.model.stream(request):
        if isinstance(event, LLMFailed):
            return ModelActionResult(
                observation=ModelObservation(
                    observation_id=observation_id,
                    status="failed",
                    error=_model_failure(event.error, observation_id),
                    purpose=purpose,
                    attempts=event.attempts,
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
            attempts = event.attempts

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
                purpose=purpose,
                attempts=attempts,
            ),
            catalog_snapshot=snapshot,
        )
    return ModelActionResult(
        observation=ModelObservation(
            observation_id=observation_id,
            message=assistant,
            purpose=purpose,
            attempts=attempts,
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


def _context_purpose(purpose: ModelPurpose) -> ContextPurpose:
    if purpose == "verification":
        return "verification"
    if purpose == "final_response":
        return "finalization"
    return "reasoning"


def _directive_text(directive: CoreDirective) -> str:
    lines = [directive.code]
    lines.extend(directive.constraints)
    if directive.evidence_refs:
        lines.append("evidence: " + ", ".join(directive.evidence_refs))
    return "\n".join(lines)


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
