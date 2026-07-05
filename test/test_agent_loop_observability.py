from __future__ import annotations

import asyncio
from typing import Any, Callable


def test_agent_loop_records_tool_events_with_schema() -> None:
    asyncio.run(_run_agent_loop_observability_case())


def test_agent_loop_propagates_tool_result_status_to_events() -> None:
    asyncio.run(_run_agent_loop_tool_status_case())


def test_agent_loop_preserves_tool_adapter_errors_in_events() -> None:
    asyncio.run(_run_agent_loop_tool_adapter_error_case())


def test_agent_loop_stops_at_max_tool_iterations() -> None:
    asyncio.run(_run_agent_loop_max_tool_iterations_case())


async def _run_agent_loop_observability_case() -> None:
    from codepilot.core.contracts import AgentLoopInput, AgentLoopLimits, AgentLoopPorts, RunCorrelation
    from codepilot.core.loop import run_agent_loop
    from codepilot.llm.ports import LLMCompleted, ModelDescriptor
    from codepilot.observability import event_to_record, summarize_events, validate_run_event
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
    from codepilot.tools.ports import ToolObservation

    model = _ScriptedModel(
        lambda request, calls: AssistantMessage(
            content=[TextContent(text="done")]
            if any(isinstance(message, ToolResultMessage) for message in request.messages)
            else [ToolCall(id="call_1", name="echo", arguments={"text": "hello"})],
            stop_reason="stop" if calls > 1 else "toolUse",
        )
    )
    tools = _StaticToolPort(
        ToolObservation(
            tool_call_id="call_1",
            name="echo",
            status="success",
            content=(TextContent(text="hello"),),
            metadata={"details": {"ok": True}},
        )
    )

    outcome = await run_agent_loop(
        _loop_input("run_obs", prompt="use echo"),
        AgentLoopPorts(model=model, tools=tools),
    )

    event_types = [event["type"] for event in outcome.events]
    assert "tool_execution_start" in event_types
    assert "tool_execution_end" in event_types
    assert event_types[-1] == "agent_end"
    assert any(isinstance(message, ToolResultMessage) for message in outcome.new_messages)
    records = [record for event in outcome.events if (record := event_to_record(event))]
    assert all(validate_run_event(record) == [] for record in records)

    summary = summarize_events(records)
    assert summary["event_counts"]["tool_call_finished"] == 1
    assert summary["event_counts"]["run_finished"] == 1


async def _run_agent_loop_tool_status_case() -> None:
    from codepilot.core.contracts import AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.observability import event_to_record, summarize_events, validate_run_event
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
    from codepilot.tools.ports import ToolObservation

    model = _ScriptedModel(
        lambda request, calls: AssistantMessage(
            content=[TextContent(text="blocked")]
            if any(isinstance(message, ToolResultMessage) for message in request.messages)
            else [ToolCall(id="call_1", name="write", arguments={"path": "a.txt"})],
            stop_reason="stop" if calls > 1 else "toolUse",
        )
    )
    tools = _StaticToolPort(
        ToolObservation(
            tool_call_id="call_1",
            name="write",
            status="denied",
            content=(TextContent(text="Tool blocked"),),
            metadata={"error_code": "read_only_mode", "approved": False},
        )
    )

    outcome = await run_agent_loop(
        _loop_input("run_status", prompt="write file"),
        AgentLoopPorts(model=model, tools=tools),
    )

    tool_end = next(event for event in outcome.events if event["type"] == "tool_execution_end")
    assert tool_end["isError"] is True
    assert tool_end["status"] == "denied"
    assert tool_end["approved"] is False
    assert tool_end["approvalId"] is None
    assert tool_end["errorReason"] == "read_only_mode"
    assert any(
        isinstance(message, ToolResultMessage)
        and message.is_error
        and message.status == "denied"
        for message in outcome.new_messages
    )
    records = [record for event in outcome.events if (record := event_to_record(event))]
    assert all(validate_run_event(record) == [] for record in records)

    summary = summarize_events(records)
    assert summary["event_counts"]["tool_call_finished"] == 1


async def _run_agent_loop_tool_adapter_error_case() -> None:
    from codepilot.core.contracts import AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.observability import event_to_record, validate_run_event
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
    from codepilot.tools.ports import ToolObservation

    model = _ScriptedModel(
        lambda request, calls: AssistantMessage(
            content=[TextContent(text="done")]
            if any(isinstance(message, ToolResultMessage) for message in request.messages)
            else [ToolCall(id="call_1", name="echo", arguments={"text": "hello"})],
            stop_reason="stop" if calls > 1 else "toolUse",
        )
    )
    tools = _StaticToolPort(
        ToolObservation(
            tool_call_id="call_1",
            name="echo",
            status="error",
            content=(TextContent(text="hook exploded"),),
            metadata={"error_code": "after_tool_hook_error"},
        )
    )

    outcome = await run_agent_loop(
        _loop_input("run_tool_adapter_error", prompt="use echo"),
        AgentLoopPorts(model=model, tools=tools),
    )

    tool_end = next(event for event in outcome.events if event["type"] == "tool_execution_end")
    assert tool_end["isError"] is True
    assert tool_end["status"] == "error"
    assert tool_end["errorReason"] == "after_tool_hook_error"
    result_message = next(
        message for message in outcome.new_messages if isinstance(message, ToolResultMessage)
    )
    assert result_message.metadata["error_code"] == "after_tool_hook_error"
    records = [record for event in outcome.events if (record := event_to_record(event))]
    assert all(validate_run_event(record) == [] for record in records)


async def _run_agent_loop_max_tool_iterations_case() -> None:
    from codepilot.core.contracts import AgentLoopLimits, AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.observability import event_to_record, validate_run_event
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall
    from codepilot.tools.ports import ToolObservation

    model = _ScriptedModel(
        lambda _request, _calls: AssistantMessage(
            content=[ToolCall(id="call_loop", name="echo", arguments={"text": "loop"})],
            stop_reason="toolUse",
        )
    )
    tools = _StaticToolPort(
        ToolObservation(
            tool_call_id="call_loop",
            name="echo",
            status="success",
            content=(TextContent(text="loop"),),
        )
    )

    outcome = await run_agent_loop(
        _loop_input(
            "run_max_iterations",
            prompt="loop",
            limits=AgentLoopLimits(max_tool_iterations=1, max_model_turns=3),
        ),
        AgentLoopPorts(model=model, tools=tools),
    )

    assert any(
        event["type"] == "error" and event["error"]["code"] == "run.max_iterations"
        for event in outcome.events
    )
    assert outcome.events[-1]["type"] == "agent_end"
    assert outcome.status == "failed"
    assert outcome.stop_reason == "max_iterations"
    records = [record for event in outcome.events if (record := event_to_record(event))]
    assert all(validate_run_event(record) == [] for record in records)
    assert isinstance(outcome.final_message, AssistantMessage)
    assert outcome.final_message.stop_reason == "max_iterations"
    assert len([event for event in outcome.events if event["type"] == "tool_execution_start"]) == 1


def _loop_input(
    run_id: str,
    *,
    prompt: str,
    limits: Any | None = None,
):
    from codepilot.core.contracts import AgentLoopInput, AgentLoopLimits, RunCorrelation
    from codepilot.llm.ports import ModelDescriptor

    return AgentLoopInput(
        run_id=run_id,
        correlation=RunCorrelation(session_id="s1"),
        user_prompt=prompt,
        model=ModelDescriptor(provider="unit-test", model_id="test-model"),
        limits=limits or AgentLoopLimits(max_model_turns=4),
    )


class _ScriptedModel:
    def __init__(self, factory: Callable[[Any, int], Any]) -> None:
        self._factory = factory
        self._calls = 0

    async def stream(self, request):
        from codepilot.llm.ports import LLMCompleted

        self._calls += 1
        yield LLMCompleted(message=self._factory(request, self._calls))


class _StaticToolPort:
    def __init__(self, observation):
        self._observation = observation

    def catalog(self):
        return {"tools": [self._observation.name]}

    async def execute(self, invocation):
        from dataclasses import replace

        return replace(
            self._observation,
            tool_call_id=invocation.tool_call_id,
            name=invocation.name,
        )
