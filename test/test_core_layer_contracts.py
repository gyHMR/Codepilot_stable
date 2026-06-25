from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest


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


def test_before_tool_hook_block_is_a_policy_denial_not_execution_error() -> None:
    asyncio.run(_run_before_tool_hook_block_case())


def test_parallel_tool_scheduling_requires_explicit_safe_non_exclusive_metadata() -> None:
    from codepilot.core.tool_coordinator import can_schedule_tool_in_parallel
    from codepilot.protocols import TextContent, ToolMetadata
    from codepilot.tools import AgentTool, AgentToolResult

    async def execute(_tool_call_id, _params, _signal=None, _on_update=None):
        return AgentToolResult(content=[TextContent(text="ok")])

    def metadata(*, concurrency_safe: bool, exclusive: bool) -> ToolMetadata:
        return ToolMetadata(
            name="demo",
            category="unit",
            read_only=concurrency_safe,
            concurrency_safe=concurrency_safe,
            exclusive=exclusive,
            requires_approval=False,
            risk_level="low",
            resource_scope=("workspace",),
        )

    def tool(tool_metadata: ToolMetadata | None) -> AgentTool:
        return AgentTool(
            name="demo",
            label="Demo",
            description="Demo tool",
            parameters={},
            execute=execute,
            runtime_managed=True,
            metadata=tool_metadata,
        )

    assert can_schedule_tool_in_parallel(
        tool(metadata(concurrency_safe=True, exclusive=False))
    )
    assert not can_schedule_tool_in_parallel(
        tool(metadata(concurrency_safe=True, exclusive=True))
    )
    assert not can_schedule_tool_in_parallel(
        tool(metadata(concurrency_safe=False, exclusive=False))
    )
    assert not can_schedule_tool_in_parallel(tool(None))


async def _run_before_tool_hook_block_case() -> None:
    from codepilot.core import (
        AgentContext,
        AgentLoopConfig,
        BeforeToolCallResult,
        run_agent_loop,
    )
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    executed = False

    async def managed_execute(_tool_call_id, _params, _signal=None, _on_update=None):
        nonlocal executed
        executed = True
        return AgentToolResult(content=[TextContent(text="unsafe")])

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            stream.end(AssistantMessage(content=[TextContent(text="blocked")]))
        else:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="call_write", name="write", arguments={})],
                    stop_reason="toolUse",
                )
            )
        return stream

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="write")],
        context=AgentContext(
            system_prompt="",
            messages=[],
            tools=[
                AgentTool(
                    name="write",
                    label="Write",
                    description="Write file",
                    parameters={},
                    execute=managed_execute,
                    runtime_managed=True,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=_model(),
            convert_to_llm=lambda items: items,
            before_tool_call=lambda _ctx, _signal=None: BeforeToolCallResult(
                block=True,
                reason="workspace policy denies writes",
            ),
        ),
        emit=events.append,
        stream_fn=fake_stream,
    )

    assert not executed
    tool_end = next(event for event in events if event["type"] == "tool_execution_end")
    assert tool_end["status"] == "denied"
    assert tool_end["errorReason"] == "before_tool_call_blocked"
    result_message = next(message for message in result.messages if isinstance(message, ToolResultMessage))
    assert result_message.status == "denied"
    assert result_message.error_code == "before_tool_call_blocked"
    assert result_message.details == {
        "reason": "before_tool_call_blocked",
        "status": "denied",
        "message": "workspace policy denies writes",
    }


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


def test_event_emitter_normalizes_envelope_and_rejects_unknown_event_type() -> None:
    asyncio.run(_run_event_emitter_contract_case())


async def _run_event_emitter_contract_case() -> None:
    from codepilot.core.events import AgentEventEmitter

    events: list[dict[str, Any]] = []
    emitter = AgentEventEmitter(events.append, run_id=" run_events ", session_id=" session_1 ")

    await emitter.emit({"type": "turn_start", "runId": "caller_override"})
    await emitter.emit({"type": "message_end", "message": "payload"})

    assert events[0]["runId"] == "run_events"
    assert events[0]["sessionId"] == "session_1"
    assert events[0]["turnId"] == 1
    assert events[0]["eventId"] == "run_events:1"
    assert events[1]["turnId"] == 1
    assert events[1]["eventId"] == "run_events:2"

    with pytest.raises(ValueError, match="runtime event type"):
        await emitter.emit({"type": "unknown_event"})

    with pytest.raises(ValueError, match="event type"):
        await emitter.emit({})


def test_core_context_and_loop_config_validate_loop_boundaries() -> None:
    from codepilot.core.types import AgentContext, AgentLoopConfig
    from codepilot.protocols import UserMessage

    model = _model()
    messages = [UserMessage(content="hello")]
    recovery_projection = {"task_id": "task_1", "nested": {"step": "s1"}}
    context = AgentContext(
        system_prompt="rules",
        messages=messages,
        task_recovery_projection=recovery_projection,
        task_signal={"action": "continue"},
    )
    messages.append(UserMessage(content="mutated"))
    recovery_projection["nested"] = {"step": "mutated"}

    assert context.messages == [UserMessage(content="hello")]
    assert context.task_recovery_projection == {"task_id": "task_1", "nested": {"step": "s1"}}
    assert context.task_signal == {"action": "continue"}

    with pytest.raises(TypeError, match="messages"):
        AgentContext(system_prompt="rules", messages="not a list")  # type: ignore[arg-type]

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda items: items,
        tool_execution=" sequential ",
        max_tool_iterations=1,
        max_tool_calls_per_turn=None,
        repeated_tool_call_limit=0,
        max_model_retries=0,
        retry_base_delay_ms=0,
        session_id=" session_1 ",
    )

    assert config.tool_execution == "sequential"
    assert config.max_tool_calls_per_turn is None
    assert config.repeated_tool_call_limit == 0
    assert config.session_id == "session_1"

    with pytest.raises(TypeError, match="model"):
        AgentLoopConfig(model="demo", convert_to_llm=lambda items: items)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="convert_to_llm"):
        AgentLoopConfig(model=model, convert_to_llm=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="tool_execution"):
        AgentLoopConfig(model=model, convert_to_llm=lambda items: items, tool_execution="batched")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_tool_iterations"):
        AgentLoopConfig(model=model, convert_to_llm=lambda items: items, max_tool_iterations=0)

    with pytest.raises(ValueError, match="max_tool_calls_per_turn"):
        AgentLoopConfig(model=model, convert_to_llm=lambda items: items, max_tool_calls_per_turn=0)

    with pytest.raises(TypeError, match="retry_enabled"):
        AgentLoopConfig(model=model, convert_to_llm=lambda items: items, retry_enabled="yes")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="reasoning"):
        AgentLoopConfig(model=model, convert_to_llm=lambda items: items, reasoning=True)  # type: ignore[arg-type]


def test_agent_options_and_state_validate_agent_boundaries() -> None:
    from codepilot.core import AgentOptions
    from codepilot.core.types import AgentState
    from codepilot.protocols import UserMessage
    from codepilot.tools import AgentTool

    model = _model()
    tool = AgentTool(
        name="echo",
        label="Echo",
        description="Echo input",
        parameters={},
        execute=lambda *_args, **_kwargs: None,
    )
    messages = [UserMessage(content="hello")]
    tools = [tool]
    task_recovery_projection = {"task_id": "task_1", "nested": {"step": "s1"}}

    options = AgentOptions(
        model=model,
        system_prompt="rules",
        tools=tools,
        messages=messages,
        thinking_level=" high ",
        tool_execution=" sequential ",
        max_tool_iterations=1,
        max_tool_calls_per_turn=None,
        repeated_tool_call_limit=0,
        max_model_retries=0,
        retry_base_delay_ms=0,
        session_id=" session_1 ",
        task_recovery_projection=task_recovery_projection,
    )
    tools.append(tool)
    messages.append(UserMessage(content="mutated"))
    task_recovery_projection["nested"] = {"step": "mutated"}

    assert options.thinking_level == "high"
    assert options.tool_execution == "sequential"
    assert options.session_id == "session_1"
    assert options.messages == [UserMessage(content="hello")]
    assert options.tools == [tool]
    assert options.task_recovery_projection == {"task_id": "task_1", "nested": {"step": "s1"}}

    state_messages = [UserMessage(content="state")]
    state = AgentState(
        system_prompt="rules",
        model=model,
        thinking_level=" xhigh ",
        tools=[tool],
        messages=state_messages,
        pending_tool_calls={" call_1 ", ""},
        error=" failed ",
    )
    state_messages.append(UserMessage(content="mutated"))

    assert state.thinking_level == "xhigh"
    assert state.messages == [UserMessage(content="state")]
    assert state.pending_tool_calls == {"call_1"}
    assert state.error == "failed"

    with pytest.raises(TypeError, match="model"):
        AgentOptions(model="demo")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="convert_to_llm"):
        AgentOptions(model=model, convert_to_llm=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="thinking_level"):
        AgentOptions(model=model, thinking_level="extreme")

    with pytest.raises(ValueError, match="tool_execution"):
        AgentOptions(model=model, tool_execution="batched")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_tool_iterations"):
        AgentOptions(model=model, max_tool_iterations=0)

    with pytest.raises(ValueError, match="max_tool_calls_per_turn"):
        AgentOptions(model=model, max_tool_calls_per_turn=0)

    with pytest.raises(TypeError, match="allow_unmanaged_tools"):
        AgentOptions(model=model, allow_unmanaged_tools="yes")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="thinking_level"):
        AgentState(system_prompt="", model=model, thinking_level="fast")

    with pytest.raises(TypeError, match="is_streaming"):
        AgentState(system_prompt="", model=model, is_streaming="yes")  # type: ignore[arg-type]


def test_llm_runner_prepares_context_after_current_task_injection() -> None:
    asyncio.run(_run_prepare_context_sees_current_task_case())


async def _run_prepare_context_sees_current_task_case() -> None:
    from codepilot.core.events import AgentEventEmitter
    from codepilot.core.llm_runner import LLMStreamRunner
    from codepilot.core.types import (
        AgentContext,
        AgentLoopConfig,
        ContextPreparationRequest,
        PreparedAgentContext,
    )
    from codepilot.protocols import AssistantMessage, ContextReport, TextContent, UserMessage

    captured: dict[str, str] = {}

    def prepare_context(context: AgentContext, _request: ContextPreparationRequest) -> PreparedAgentContext:
        captured["system_prompt"] = context.system_prompt
        return PreparedAgentContext(
            system_prompt=context.system_prompt,
            messages=context.messages,
            tools=context.tools,
            report=ContextReport(
                context_id="ctx_1",
                repository_fingerprint="fp",
                total_budget_tokens=100,
                estimated_tokens_before=1,
                estimated_tokens_after=1,
            ),
        )

    async def complete(_model, _context, _options):
        return AssistantMessage(content=[TextContent(text="done")])

    events: list[dict[str, Any]] = []
    runner = LLMStreamRunner(
        config=AgentLoopConfig(
            model=_model(streaming=False),
            convert_to_llm=lambda items: items,
            prepare_context=prepare_context,
        ),
        emitter=AgentEventEmitter(events.append, run_id="run_prepare_context"),
        complete_fn=complete,
    )

    await runner.stream_assistant_response(
        AgentContext(
            system_prompt="Base rules",
            messages=[UserMessage(content="continue")],
            current_task="## Current Task\nImplement the next verified step.",
        )
    )

    assert "## Current Task" in captured["system_prompt"]
    assert captured["system_prompt"].count("## Current Task") == 1
    assert any(event["type"] == "context_prepared" for event in events)
