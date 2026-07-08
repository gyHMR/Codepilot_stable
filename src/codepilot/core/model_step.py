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
)
from codepilot.tools.contracts import ToolCatalogView

from .context_preflight import TOOL_RESULT_MAX_CHARS, prepare_messages_for_model
from .contracts import AgentLoopInput, AgentLoopPorts, AgentMessage


@dataclass(frozen=True)
class ModelTurnResult:
    message: AssistantMessage
    usage: Usage | None = None
    error: Any = None


async def run_model_turn(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    messages: list[Any],
    *,
    emit: Callable[[dict[str, Any]], None] | None = None,
) -> ModelTurnResult:
    """Ask the model for the next assistant message."""

    if ports.model is None:
        return ModelTurnResult(
            message=AssistantMessage(content=[TextContent(text=input.user_prompt or "")])
        )

    assistant: AssistantMessage | None = None
    usage = None
    request = await build_model_request(input, ports, messages)
    async for event in ports.model.stream(request):
        if isinstance(event, LLMFailed):
            return ModelTurnResult(
                message=AssistantMessage(content=[TextContent(text="")]),
                error=event.error,
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
            "assistantMessageEvent": {
                "type": event_type,
                **payload,
            },
        }
    )


async def build_model_request(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    messages: list[Any],
) -> LLMRequest:
    tools = tool_catalog_for_request(input, ports)
    request_data: dict[str, Any] = {
        "run_id": input.run_id,
        "session_id": input.correlation.session_id or "",
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


def tool_catalog_for_request(input: AgentLoopInput, ports: AgentLoopPorts) -> list[Tool]:
    if ports.tools is not None:
        catalog = ports.tools.catalog(input.mode)
        if catalog:
            return [_as_tool(item) for item in _catalog_items(catalog)]
    return [_as_tool(item) for item in input.tools]


def _catalog_items(catalog: Any) -> list[Any]:
    if isinstance(catalog, ToolCatalogView):
        return list(catalog.tools)
    if isinstance(catalog, dict):
        value = catalog.get("tools")
        if isinstance(value, (list, tuple)):
            return list(value)
        return [catalog]
    if isinstance(catalog, (list, tuple)):
        return list(catalog)
    return [catalog]


def _as_tool(item: Any) -> Tool:
    if isinstance(item, Tool):
        return item
    if isinstance(item, str):
        return Tool(name=item, description=item, parameters={})
    if hasattr(item, "to_spec"):
        spec = item.to_spec()
        if isinstance(spec, Tool):
            return spec
    if isinstance(item, dict):
        name = str(item.get("name") or item.get("id") or "")
        description = str(item.get("description") or name)
        return Tool(
            name=name,
            description=description,
            parameters=dict(item.get("parameters") or item.get("input_schema") or {}),
        )
    name = str(getattr(item, "name"))
    return Tool(
        name=name,
        description=str(getattr(item, "description", "") or name),
        parameters=dict(getattr(item, "parameters", {}) or {}),
    )


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
