from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _model(**capability_overrides):
    from codepilot.protocols import Model, ModelCapabilities

    capabilities = ModelCapabilities(**capability_overrides)
    return Model(
        id="contract-model",
        name="Contract Model",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=capabilities.reasoning,
        input=["text", "image"] if capabilities.vision else ["text"],
        context_window=1000,
        max_tokens=100,
        capabilities=capabilities,
    )


def test_core_rejects_unmanaged_tool_by_default() -> None:
    asyncio.run(_run_unmanaged_tool_case())


async def _run_unmanaged_tool_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    executed = False

    async def raw_execute(_tool_call_id, _params, _signal=None, _on_update=None):
        nonlocal executed
        executed = True
        return AgentToolResult(content=[TextContent(text="unsafe")])

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            stream.end(AssistantMessage(content=[TextContent(text="done")]))
        else:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="call_raw", name="raw", arguments={})],
                    stop_reason="toolUse",
                )
            )
        return stream

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="run raw")],
        context=AgentContext(
            system_prompt="",
            messages=[],
            tools=[
                AgentTool(
                    name="raw",
                    label="Raw",
                    description="Unmanaged raw tool",
                    parameters={},
                    execute=raw_execute,
                )
            ],
        ),
        config=AgentLoopConfig(model=_model(), convert_to_llm=lambda items: items),
        emit=events.append,
        stream_fn=fake_stream,
    )
    messages = result.messages

    assert not executed
    tool_end = next(event for event in events if event["type"] == "tool_execution_end")
    assert tool_end["status"] == "denied"
    assert tool_end["errorReason"] == "unmanaged_tool"
    result_message = next(message for message in messages if isinstance(message, ToolResultMessage))
    assert result_message.status == "denied"


def test_llm_runner_applies_model_capabilities() -> None:
    asyncio.run(_run_capability_case())


async def _run_capability_case() -> None:
    from codepilot.core.events import AgentEventEmitter
    from codepilot.core.llm_runner import LLMStreamRunner
    from codepilot.core.types import AgentContext, AgentLoopConfig
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, ImageContent, TextContent, UserMessage
    from codepilot.tools import AgentTool

    async def raw_execute(*_args):
        raise AssertionError("tool should not execute")

    captured: dict[str, Any] = {}

    async def fake_stream(_model, context, options):
        captured["context"] = context
        captured["options"] = options
        stream = AssistantMessageEventStream()
        stream.end(AssistantMessage(content=[TextContent(text="ok")]))
        return stream

    events: list[dict[str, Any]] = []
    model = _model(tools=False, reasoning=False, system_prompt=False)
    runner = LLMStreamRunner(
        config=AgentLoopConfig(
            model=model,
            convert_to_llm=lambda items: items,
            reasoning="high",
        ),
        emitter=AgentEventEmitter(events.append, run_id="run_capabilities"),
        stream_fn=fake_stream,
    )
    await runner.stream_assistant_response(
        AgentContext(
            system_prompt="hidden",
            messages=[UserMessage(content="hello")],
            tools=[
                AgentTool(
                    name="raw",
                    label="Raw",
                    description="Raw",
                    parameters={},
                    execute=raw_execute,
                )
            ],
        )
    )

    assert captured["context"].tools == []
    assert captured["context"].system_prompt is None
    assert captured["options"].reasoning is None

    called = False

    async def should_not_stream(*_args):
        nonlocal called
        called = True
        raise AssertionError("vision capability check should stop before provider call")

    vision_events: list[dict[str, Any]] = []
    vision_runner = LLMStreamRunner(
        config=AgentLoopConfig(
            model=_model(vision=False),
            convert_to_llm=lambda items: items,
        ),
        emitter=AgentEventEmitter(vision_events.append, run_id="run_vision"),
        stream_fn=should_not_stream,
    )
    response = await vision_runner.stream_assistant_response(
        AgentContext(
            system_prompt="",
            messages=[UserMessage(content=[ImageContent(data="abc")])],
        )
    )

    assert not called
    assert response.stop_reason == "error"
    assert response.error_info is not None
    assert response.error_info.code == "llm.unsupported_capability"
    error_event = next(event for event in vision_events if event["type"] == "error")
    assert error_event["source"] == "llm"
    assert error_event["retryable"] is False


def test_non_streaming_model_uses_complete_path() -> None:
    asyncio.run(_run_non_streaming_case())


async def _run_non_streaming_case() -> None:
    from codepilot.core.events import AgentEventEmitter
    from codepilot.core.llm_runner import LLMStreamRunner
    from codepilot.core.types import AgentContext, AgentLoopConfig
    from codepilot.protocols import AssistantMessage, TextContent, UserMessage

    complete_called = False

    async def fake_complete(_model, _context, _options):
        nonlocal complete_called
        complete_called = True
        return AssistantMessage(content=[TextContent(text="complete")])

    events: list[dict[str, Any]] = []
    runner = LLMStreamRunner(
        config=AgentLoopConfig(
            model=_model(streaming=False),
            convert_to_llm=lambda items: items,
        ),
        emitter=AgentEventEmitter(events.append, run_id="run_complete"),
        complete_fn=fake_complete,
    )
    context = AgentContext(
        system_prompt="",
        messages=[UserMessage(content="hello")],
    )
    response = await runner.stream_assistant_response(context)

    assert complete_called
    assert response.content[0].text == "complete"
    assert context.messages[-1] is response
    assert [event["type"] for event in events] == ["message_start", "message_end"]
