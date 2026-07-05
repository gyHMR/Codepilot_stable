from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

from codepilot.llm.ports import LLMCompleted, LLMFailed, LLMRequest
from codepilot.protocols import AssistantMessage, TextContent, Tool, Usage

from .contracts import AgentLoopInput, AgentLoopPorts


@dataclass(frozen=True)
class ModelTurnResult:
    message: AssistantMessage
    usage: Usage | None = None
    error: Any = None


async def run_model_turn(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    messages: list[Any],
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
        if isinstance(event, LLMCompleted):
            assistant = event.message
            usage = event.usage

    return ModelTurnResult(
        message=assistant or AssistantMessage(content=[TextContent(text="")]),
        usage=usage,
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
        correlation={
            "run_id": input.run_id,
            "session_id": input.correlation.session_id or "",
        },
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
