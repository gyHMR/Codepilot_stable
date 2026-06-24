from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import asyncio


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _test_model():
    from codepilot.protocols import Model

    return Model(
        id="test-model",
        name="Test Model",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=1000,
        max_tokens=100,
    )


def _managed_tool(name: str, execute):
    from codepilot.tools import AgentTool, ToolMetadata, ToolRegistry, ToolRuntime

    registry = ToolRegistry()
    registry.register(
        AgentTool(
            name=name,
            label=name.title(),
            description=f"Test tool: {name}",
            parameters={"type": "object", "properties": {}},
            execute=execute,
        ),
        metadata=ToolMetadata(
            name=name,
            category="test",
            read_only=True,
            concurrency_safe=True,
            exclusive=False,
            requires_approval=False,
            risk_level="low",
            resource_scope=("test",),
        ),
    )
    return ToolRuntime(registry).as_agent_tools()[0]


def test_agent_loop_records_tool_events_with_schema() -> None:
    asyncio.run(_run_agent_loop_observability_case())


def test_agent_loop_propagates_tool_result_status_to_events() -> None:
    asyncio.run(_run_agent_loop_tool_status_case())


def test_agent_loop_converts_after_tool_hook_exception_to_tool_error() -> None:
    asyncio.run(_run_agent_loop_after_tool_hook_error_case())


def test_agent_loop_stops_at_max_tool_iterations() -> None:
    asyncio.run(_run_agent_loop_max_tool_iterations_case())


async def _run_agent_loop_observability_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
    from codepilot.observability import summarize_events, validate_agent_event
    from codepilot.tools import AgentToolResult

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        has_tool_result = any(isinstance(message, ToolResultMessage) for message in context.messages)
        if has_tool_result:
            stream.end(AssistantMessage(content=[TextContent(text="done")], stop_reason="stop"))
        else:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="call_1", name="echo", arguments={"text": "hello"})],
                    stop_reason="toolUse",
                )
            )
        return stream

    async def echo_tool(_tool_call_id: str, params: dict[str, Any], _signal=None, on_update=None):
        if on_update:
            on_update(AgentToolResult(content=[TextContent(text="working")], details={"phase": "update"}))
        return AgentToolResult(content=[TextContent(text=params["text"])], details={"ok": True})

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="use echo")],
        context=AgentContext(
            system_prompt="",
            messages=[],
            tools=[_managed_tool("echo", echo_tool)],
        ),
        config=AgentLoopConfig(model=_test_model(), convert_to_llm=lambda m: m, tool_execution="sequential", session_id="s1"),
        emit=events.append,
        stream_fn=fake_stream,
    )
    messages = result.messages

    event_types = [event["type"] for event in events]
    assert "tool_execution_start" in event_types
    assert "tool_execution_update" in event_types
    assert "tool_execution_end" in event_types
    assert event_types[-1] == "agent_end"
    assert any(isinstance(message, ToolResultMessage) for message in messages)
    assert all(validate_agent_event(event) == [] for event in events)

    summary = summarize_events(events)
    assert summary["tool_calls"] == 1
    assert summary["tool_errors"] == 0
    assert summary["run_count"] == 1


async def _run_agent_loop_tool_status_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
    from codepilot.observability import summarize_events, validate_agent_event
    from codepilot.tools import AgentToolResult

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        has_tool_result = any(isinstance(message, ToolResultMessage) for message in context.messages)
        if has_tool_result:
            stream.end(AssistantMessage(content=[TextContent(text="blocked")], stop_reason="stop"))
        else:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="call_1", name="write", arguments={"path": "a.txt"})],
                    stop_reason="toolUse",
                )
            )
        return stream

    async def blocked_tool(_tool_call_id: str, params: dict[str, Any], _signal=None, on_update=None):
        _ = params, on_update
        return AgentToolResult(
            content=[TextContent(text="Tool blocked")],
            details={"reason": "read_only_mode"},
            is_error=True,
            status="denied",
            approved=False,
        )

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="write file")],
        context=AgentContext(
            system_prompt="",
            messages=[],
            tools=[_managed_tool("write", blocked_tool)],
        ),
        config=AgentLoopConfig(
            model=_test_model(),
            convert_to_llm=lambda m: m,
            tool_execution="sequential",
            session_id="s1",
        ),
        emit=events.append,
        stream_fn=fake_stream,
    )
    messages = result.messages

    tool_end = next(event for event in events if event["type"] == "tool_execution_end")
    assert tool_end["isError"] is True
    assert tool_end["status"] == "denied"
    assert tool_end["approved"] is False
    assert tool_end["approvalId"] is None
    assert tool_end["errorReason"] == "read_only_mode"
    assert any(
        isinstance(message, ToolResultMessage)
        and message.is_error
        and message.status == "denied"
        for message in messages
    )
    assert all(validate_agent_event(event) == [] for event in events)

    summary = summarize_events(events)
    assert summary["tool_calls"] == 1
    assert summary["tool_errors"] == 1


async def _run_agent_loop_after_tool_hook_error_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
    from codepilot.observability import validate_agent_event
    from codepilot.tools import AgentToolResult

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        has_tool_result = any(isinstance(message, ToolResultMessage) for message in context.messages)
        if has_tool_result:
            stream.end(AssistantMessage(content=[TextContent(text="done")], stop_reason="stop"))
        else:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="call_1", name="echo", arguments={"text": "hello"})],
                    stop_reason="toolUse",
                )
            )
        return stream

    async def echo_tool(_tool_call_id: str, params: dict[str, Any], _signal=None, on_update=None):
        _ = on_update
        return AgentToolResult(content=[TextContent(text=params["text"])], details={"ok": True})

    def broken_after_hook(_ctx, _signal=None):
        raise RuntimeError("hook exploded")

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="use echo")],
        context=AgentContext(
            system_prompt="",
            messages=[],
            tools=[_managed_tool("echo", echo_tool)],
        ),
        config=AgentLoopConfig(
            model=_test_model(),
            convert_to_llm=lambda m: m,
            tool_execution="sequential",
            session_id="s1",
            after_tool_call=broken_after_hook,
        ),
        emit=events.append,
        stream_fn=fake_stream,
    )

    assert result.status == "waiting_user"
    assert result.stop_reason == "task_incomplete"
    assert result.task is not None
    assert result.task.completion_satisfied is False
    tool_end = next(event for event in events if event["type"] == "tool_execution_end")
    assert tool_end["isError"] is True
    assert tool_end["status"] == "error"
    assert tool_end["errorReason"] == "after_tool_hook_error"
    result_message = next(message for message in result.messages if isinstance(message, ToolResultMessage))
    assert result_message.error_code == "after_tool_hook_error"
    assert all(validate_agent_event(event) == [] for event in events)


async def _run_agent_loop_max_tool_iterations_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, UserMessage
    from codepilot.observability import validate_agent_event
    from codepilot.tools import AgentToolResult

    async def fake_stream(_model, _context, _options):
        stream = AssistantMessageEventStream()
        stream.end(
            AssistantMessage(
                content=[ToolCall(id="call_loop", name="echo", arguments={"text": "loop"})],
                stop_reason="toolUse",
            )
        )
        return stream

    async def echo_tool(_tool_call_id: str, params: dict[str, Any], _signal=None, on_update=None):
        _ = on_update
        return AgentToolResult(content=[TextContent(text=params["text"])], details={"ok": True})

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="loop")],
        context=AgentContext(
            system_prompt="",
            messages=[],
            tools=[_managed_tool("echo", echo_tool)],
        ),
        config=AgentLoopConfig(
            model=_test_model(),
            convert_to_llm=lambda m: m,
            tool_execution="sequential",
            session_id="s1",
            max_tool_iterations=1,
        ),
        emit=events.append,
        stream_fn=fake_stream,
    )
    messages = result.messages

    assert any(event["type"] == "error" and event["error"] == "run.max_iterations" for event in events)
    assert events[-1]["type"] == "agent_end"
    assert any(
        isinstance(message, AssistantMessage) and message.stop_reason == "max_iterations"
        for message in messages
    )
    assert len([event for event in events if event["type"] == "tool_execution_start"]) == 1
    assert all(validate_agent_event(event) == [] for event in events)
