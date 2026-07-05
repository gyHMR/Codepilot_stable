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
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    Tool,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from codepilot.tools.ports import ToolCatalogView

from .contracts import AgentLoopInput, AgentLoopPorts, AgentMessage


TOOL_RESULT_MAX_CHARS = 30_000
TOOL_RESULT_TRUNCATION_NOTICE = "\n...<content truncated>..."


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
    return LLMRequest(
        model=request_data.get("model", input.model),
        messages=tuple(request_data.get("messages", messages)),
        system_prompt=_system_prompt_with_task_context(request_data),
        tools=tuple(request_data.get("tools", tools)),
        correlation=LLMCorrelation(
            run_id=input.run_id,
            session_id=input.correlation.session_id or "",
        ),
    )


def _system_prompt_with_task_context(request_data: dict[str, Any]) -> str:
    system_prompt = str(request_data.get("system_prompt", ""))
    current_task = request_data.get("current_task")
    if not isinstance(current_task, str) or not current_task.strip():
        context = request_data.get("context")
        if isinstance(context, dict):
            current_task = context.get("current_task")
    if not isinstance(current_task, str) or not current_task.strip():
        return system_prompt
    if not system_prompt:
        return current_task
    return f"{system_prompt}\n\n{current_task}"


def tool_catalog_for_request(input: AgentLoopInput, ports: AgentLoopPorts) -> list[Tool]:
    if ports.tools is not None:
        catalog = ports.tools.catalog()
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
            tool_result_max_chars=tool_result_max_chars,
        )
        if converted is not None:
            result.append(converted)
    return _ensure_valid_sequence(result)


def _convert_single(
    msg: AgentMessage,
    *,
    strip_thinking: bool,
    thinking_to_text: bool,
    tool_result_max_chars: int,
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
        return _process_tool_result(msg, max_chars=tool_result_max_chars)
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


def _process_tool_result(msg: ToolResultMessage, *, max_chars: int) -> ToolResultMessage:
    total_chars = sum(len(b.text) for b in msg.content if isinstance(b, TextContent))
    if total_chars <= max_chars:
        return msg

    new_content = []
    remaining = max_chars
    for block in msg.content:
        if isinstance(block, TextContent):
            if remaining <= 0:
                continue
            if len(block.text) > remaining:
                new_content.append(
                    TextContent(text=block.text[:remaining] + TOOL_RESULT_TRUNCATION_NOTICE)
                )
                remaining = 0
            else:
                new_content.append(block)
                remaining -= len(block.text)
        elif isinstance(block, ImageContent):
            new_content.append(block)

    return ToolResultMessage(
        role=msg.role,
        tool_call_id=msg.tool_call_id,
        tool_name=msg.tool_name,
        content=new_content,
        status=msg.status,
        is_error=msg.is_error,
        approved=msg.approved,
        approval_id=msg.approval_id,
        error_code=msg.error_code,
        exit_code=msg.exit_code,
        affected_paths=list(msg.affected_paths),
        workspace_changed=msg.workspace_changed,
        diff_summary=msg.diff_summary,
        verification=dict(msg.verification) if msg.verification else None,
        details=msg.details,
        timestamp=msg.timestamp,
        metadata=dict(msg.metadata),
    )


def _ensure_valid_sequence(messages: list[Message]) -> list[Message]:
    if not messages:
        return messages
    result: list[Message] = []
    for msg in messages:
        if not result:
            result.append(msg)
            continue
        prev = result[-1]
        if isinstance(prev, AssistantMessage) and isinstance(msg, AssistantMessage):
            continue
        result.append(msg)
    return result
