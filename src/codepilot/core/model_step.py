from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from codepilot.llm.ports import (
    LLMCompleted,
    LLMCorrelation,
    LLMFailed,
    LLMReasoningDelta,
    LLMRequest,
    LLMTextDelta,
    LLMToolCallDelta,
)
from codepilot.protocols import (
    AssistantMessage,
    Message,
    TextContent,
    ThinkingContent,
    Tool,
    ToolResultMessage,
    Usage,
    UserMessage,
    tool_mode_for_run_mode,
)
from codepilot.tools.codecs import json_value
from codepilot.tools.registry import ToolCatalogSnapshot

from .context_preflight import TOOL_RESULT_MAX_CHARS, prepare_messages_for_model
from .contracts import AgentLoopInput, AgentLoopPorts, AgentMessage


@dataclass(frozen=True)
class ModelTurnResult:
    message: AssistantMessage
    usage: Usage | None = None
    error: Any = None
    catalog_snapshot: ToolCatalogSnapshot | None = None


async def run_model_turn(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    messages: list[Any],
    *,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> ModelTurnResult:
    """Ask the model for the next assistant message."""

    catalog_snapshot = tool_catalog_snapshot_for_request(input, ports)
    if ports.model is None:
        return ModelTurnResult(
            message=AssistantMessage(content=[TextContent(text=input.user_prompt or "")]),
            catalog_snapshot=catalog_snapshot,
        )

    assistant: AssistantMessage | None = None
    usage = None
    request = await build_model_request(
        input,
        ports,
        messages,
        catalog_snapshot=catalog_snapshot,
    )
    async for event in ports.model.stream(request):
        if isinstance(event, LLMFailed):
            return ModelTurnResult(
                message=AssistantMessage(content=[TextContent(text="")]),
                error=event.error,
                catalog_snapshot=catalog_snapshot,
            )
        if isinstance(event, LLMTextDelta):
            _emit_llm_delta(
                emit,
                event_type="text_delta",
                payload={"delta": event.text},
            )
            continue
        if isinstance(event, LLMReasoningDelta):
            _emit_llm_delta(
                emit,
                event_type="reasoning_delta",
                payload={"delta": event.text},
            )
            continue
        if isinstance(event, LLMToolCallDelta):
            _emit_llm_delta(
                emit,
                event_type="tool_call_delta",
                payload={"toolCall": event.tool_call},
            )
            continue
        if isinstance(event, LLMCompleted):
            assistant = event.message
            usage = event.usage

    return ModelTurnResult(
        message=assistant or AssistantMessage(content=[TextContent(text="")]),
        usage=usage,
        catalog_snapshot=catalog_snapshot,
    )


def _emit_llm_delta(
    emit: Callable[[dict[str, Any]], None] | None,
    *,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    if emit is None:
        return
    emit(
        {
            "type": "message_update",
            "assistant_message_event": {
                "type": event_type,
                **payload,
            },
        }
    )


async def build_model_request(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    messages: list[Any],
    *,
    catalog_snapshot: ToolCatalogSnapshot | None = None,
) -> LLMRequest:
    snapshot = catalog_snapshot or tool_catalog_snapshot_for_request(input, ports)
    tools = tool_catalog_for_request(input, ports, catalog_snapshot=snapshot)
    request_data: dict[str, Any] = {
        "run_id": input.run_id,
        "session_id": input.correlation.session_id or "",
        "mode": input.mode,
        "model": input.model,
        "messages": list(messages),
        "system_prompt": str(input.context.get("system_prompt", "")),
        "tools": list(tools),
        "context": dict(input.context),
    }
    if ports.context is not None:
        prepared = ports.context.prepare(request_data)
        if inspect.isawaitable(prepared):
            prepared = await prepared
        if isinstance(prepared, dict):
            request_data.update(prepared)
    request_data["system_prompt"] = _append_synthetic_control(
        str(request_data.get("system_prompt", "")),
        request_data.get("context"),
    )
    preflight = prepare_messages_for_model(
        list(request_data.get("messages", messages)),
        tool_result_max_chars=TOOL_RESULT_MAX_CHARS,
    )
    request_data["messages"] = preflight.messages
    report = request_data.get("context_report")
    if isinstance(report, dict):
        runner_preflight = preflight.report.to_dict()
        report["runner_preflight"] = runner_preflight
        recorder = getattr(ports.context, "record_preflight", None)
        if callable(recorder):
            value = recorder(runner_preflight, run_id=input.run_id)
            if inspect.isawaitable(value):
                await value
    return LLMRequest(
        model=request_data.get("model", input.model),
        messages=tuple(request_data.get("messages", messages)),
        system_prompt=str(request_data.get("system_prompt", "")),
        tools=tuple(request_data.get("tools", tools)),
        correlation=LLMCorrelation(
            run_id=input.run_id,
            session_id=input.correlation.session_id or "",
        ),
    )


def _append_synthetic_control(
    system_prompt: str,
    context: object,
) -> str:
    if not isinstance(context, dict):
        return system_prompt
    control = context.get("synthetic_control")
    if not isinstance(control, dict):
        return system_prompt
    instruction = _control_text(control, "instruction")
    if not instruction:
        return system_prompt
    section = "\n".join(
        [
            "## Synthetic Control",
            f"Source: {_control_text(control, 'source') or 'runner'}",
            f"Kind: {_control_text(control, 'kind') or 'runner_control'}",
            f"Scope: {_control_text(control, 'scope') or 'summary_only'}",
            "Lifetime: this model call only",
            "This is not a user request. Do not expand task scope from it.",
            f"Instruction: {instruction}",
        ]
    )
    if section in system_prompt:
        return system_prompt
    if system_prompt.strip():
        return f"{system_prompt.rstrip()}\n\n{section}"
    return section


def _control_text(control: dict[str, object], key: str) -> str:
    value = control.get(key)
    return value.strip() if isinstance(value, str) else ""


def tool_catalog_snapshot_for_request(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
) -> ToolCatalogSnapshot | None:
    if bool(input.context.get("suppress_tools", False)):
        return None
    if ports.tools is None:
        return None
    return ports.tools.catalog_snapshot(mode=tool_mode_for_run_mode(input.mode))


def tool_catalog_for_request(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    *,
    catalog_snapshot: ToolCatalogSnapshot | None = None,
) -> list[Tool]:
    snapshot = catalog_snapshot or tool_catalog_snapshot_for_request(input, ports)
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


def convert_to_llm(
    messages: list[AgentMessage],
    *,
    strip_thinking: bool = False,
    thinking_to_text: bool = False,
    tool_result_max_chars: int = TOOL_RESULT_MAX_CHARS,
) -> list[Message]:
    """Normalize core messages before handing them to an LLM provider."""

    result: list[Message] = []
    for msg in messages:
        converted = _convert_single(
            msg,
            strip_thinking=strip_thinking,
            thinking_to_text=thinking_to_text,
        )
        if converted is not None:
            result.append(converted)
    return prepare_messages_for_model(
        result,
        tool_result_max_chars=tool_result_max_chars,
    ).messages


def _convert_single(
    msg: AgentMessage,
    *,
    strip_thinking: bool,
    thinking_to_text: bool,
) -> Message | None:
    if isinstance(msg, UserMessage):
        return msg
    if isinstance(msg, AssistantMessage):
        return _process_assistant(
            msg,
            strip_thinking=strip_thinking,
            thinking_to_text=thinking_to_text,
        )
    if isinstance(msg, ToolResultMessage):
        return msg
    return None


def _process_assistant(
    msg: AssistantMessage,
    *,
    strip_thinking: bool,
    thinking_to_text: bool,
) -> AssistantMessage:
    if not strip_thinking and not thinking_to_text:
        return msg

    new_content = []
    for block in msg.content:
        if isinstance(block, ThinkingContent):
            if strip_thinking:
                continue
            if thinking_to_text and block.thinking:
                new_content.append(
                    TextContent(text=f"[thinking]\n{block.thinking}\n[/thinking]")
                )
                continue
        new_content.append(block)
    if not new_content:
        new_content = [TextContent(text="(no content)")]

    return AssistantMessage(
        role=msg.role,
        content=new_content,
        api=msg.api,
        provider=msg.provider,
        model=msg.model,
        usage=msg.usage,
        stop_reason=msg.stop_reason,
        response_id=msg.response_id,
        error_message=msg.error_message,
        error_info=msg.error_info,
        timestamp=msg.timestamp,
        metadata=dict(msg.metadata),
    )
