from __future__ import annotations

"""Core agent loop orchestration.

This module keeps the high-level loop readable:
prompt/follow-up messages -> LLM stream -> optional tool batch -> repeat.
Detailed event schema handling, LLM stream handling, and tool execution
coordination live in sibling modules.
"""

from typing import Any

from codepilot.llm.types import AssistantMessage, TextContent, ToolCall

from .events import AgentEventEmitter, maybe_await
from .llm_runner import LLMStreamRunner, StreamFn
from .tool_coordinator import ToolCallCoordinator
from .types import AgentContext, AgentEventSink, AgentLoopConfig, AgentMessage


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Run a new agent turn from user prompts."""

    emitter = AgentEventEmitter(emit, session_id=config.session_id)
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=[*context.messages, *prompts],
        tools=context.tools,
    )

    await emitter.emit({"type": "agent_start"})
    await emitter.emit({"type": "turn_start"})
    for prompt in prompts:
        await emitter.emit({"type": "message_start", "message": prompt})
        await emitter.emit({"type": "message_end", "message": prompt})

    await _run_loop(
        current_context,
        new_messages,
        config,
        emitter,
        signal=signal,
        stream_fn=stream_fn,
    )
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """Continue an existing agent run from the current context."""

    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if isinstance(context.messages[-1], AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")

    emitter = AgentEventEmitter(emit, session_id=config.session_id)
    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await emitter.emit({"type": "agent_start"})
    await emitter.emit({"type": "turn_start"})

    await _run_loop(
        current_context,
        new_messages,
        config,
        emitter,
        signal=signal,
        stream_fn=stream_fn,
    )
    return new_messages


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emitter: AgentEventEmitter,
    *,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> None:
    llm_runner = LLMStreamRunner(
        config=config,
        emitter=emitter,
        stream_fn=stream_fn,
    )
    tool_coordinator = ToolCallCoordinator(config=config, emitter=emitter)

    first_iteration = True
    tool_iterations = 0
    pending_messages = await _drain(config.get_steering_messages)

    while True:
        has_more_tool_calls = True

        while has_more_tool_calls or pending_messages:
            if first_iteration:
                first_iteration = False
            else:
                await emitter.emit({"type": "turn_start"})

            if pending_messages:
                await _inject_messages(pending_messages, current_context, new_messages, emitter)
                pending_messages = []

            assistant = await llm_runner.stream_assistant_response(
                current_context,
                signal=signal,
            )
            new_messages.append(assistant)

            if assistant.stop_reason in {"error", "aborted", "max_iterations"}:
                await emitter.emit(
                    {"type": "turn_end", "message": assistant, "toolResults": []}
                )
                await emitter.emit({"type": "agent_end", "messages": new_messages})
                return

            tool_calls = [
                content for content in assistant.content if isinstance(content, ToolCall)
            ]
            has_more_tool_calls = bool(tool_calls)
            tool_results = []

            if has_more_tool_calls:
                if tool_iterations >= config.max_tool_iterations:
                    await _stop_for_max_tool_iterations(
                        assistant,
                        new_messages,
                        emitter,
                        max_tool_iterations=config.max_tool_iterations,
                    )
                    return

                tool_iterations += 1
                tool_results = await tool_coordinator.execute_batch(
                    current_context,
                    assistant,
                    signal=signal,
                )
                current_context.messages.extend(tool_results)
                new_messages.extend(tool_results)

            await emitter.emit(
                {"type": "turn_end", "message": assistant, "toolResults": tool_results}
            )
            pending_messages = await _drain(config.get_steering_messages)

        followups = await _drain(config.get_follow_up_messages)
        if followups:
            pending_messages = followups
            continue
        break

    await emitter.emit({"type": "agent_end", "messages": new_messages})


async def _inject_messages(
    messages: list[AgentMessage],
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    emitter: AgentEventEmitter,
) -> None:
    for message in messages:
        await emitter.emit({"type": "message_start", "message": message})
        await emitter.emit({"type": "message_end", "message": message})
        current_context.messages.append(message)
        new_messages.append(message)


async def _stop_for_max_tool_iterations(
    assistant: AssistantMessage,
    new_messages: list[AgentMessage],
    emitter: AgentEventEmitter,
    *,
    max_tool_iterations: int,
) -> None:
    message = f"Stopped after reaching max_tool_iterations={max_tool_iterations}"
    assistant.stop_reason = "max_iterations"
    assistant.error_message = message
    await emitter.emit(
        {
            "type": "error",
            "error": "max_tool_iterations",
            "message": message,
            "maxToolIterations": max_tool_iterations,
        }
    )
    await emitter.emit({"type": "turn_end", "message": assistant, "toolResults": []})
    await emitter.emit({"type": "agent_end", "messages": new_messages})


async def _drain(callback):
    if callback is None:
        return []
    return await maybe_await(callback())

