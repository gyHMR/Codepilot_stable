from __future__ import annotations

"""One complete Agent Run: prompt -> model attempts -> tools -> RunResult."""

import asyncio
from typing import Any

from codepilot.protocols import AssistantMessage, ToolCall, ToolResultMessage
from codepilot.protocols import (
    AgentEventSink,
    AgentRunResult,
    AgentRunStopReason,
    ErrorInfo,
)

from .events import AgentEventEmitter, maybe_await
from .llm_runner import LLMStreamRunner, StreamFn
from .run import RunState, new_run_id
from .tool_coordinator import ToolCallCoordinator
from .types import AgentContext, AgentLoopConfig, AgentMessage


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
    *,
    run_id: str | None = None,
) -> AgentRunResult:
    """Run a new user task and return its explicit result."""

    state = RunState(run_id=run_id or new_run_id(), session_id=config.session_id)
    emitter = AgentEventEmitter(
        emit,
        run_id=state.run_id,
        session_id=config.session_id,
    )
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

    return await _run_safely(
        current_context,
        new_messages,
        config,
        emitter,
        state,
        signal=signal,
        stream_fn=stream_fn,
    )


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: Any | None = None,
    stream_fn: StreamFn | None = None,
    *,
    run_id: str | None = None,
) -> AgentRunResult:
    """Continue an in-memory unfinished Run."""

    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if isinstance(context.messages[-1], AssistantMessage):
        raise ValueError("Cannot continue from message role: assistant")

    state = RunState(run_id=run_id or new_run_id(), session_id=config.session_id)
    emitter = AgentEventEmitter(
        emit,
        run_id=state.run_id,
        session_id=config.session_id,
    )
    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )
    await emitter.emit({"type": "agent_start"})
    await emitter.emit({"type": "turn_start"})

    return await _run_safely(
        current_context,
        new_messages,
        config,
        emitter,
        state,
        signal=signal,
        stream_fn=stream_fn,
    )


async def _run_safely(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emitter: AgentEventEmitter,
    state: RunState,
    *,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> AgentRunResult:
    try:
        return await _run_loop(
            current_context,
            new_messages,
            config,
            emitter,
            state,
            signal=signal,
            stream_fn=stream_fn,
        )
    except asyncio.CancelledError:
        return await _finish_run(
            emitter,
            state.result(
                status="aborted",
                stop_reason="aborted",
                messages=new_messages,
                final_message=_last_assistant(new_messages),
            ),
        )
    except Exception as exc:
        error = ErrorInfo(
            code="run.internal_error",
            message=str(exc),
            retryable=False,
            source="runtime",
            details={"exception": type(exc).__name__},
        )
        await emitter.emit(
            {
                "type": "error",
                "error": error.code,
                "message": error.message,
                "source": error.source,
                "code": error.code,
                "retryable": False,
                "errorInfo": error,
            }
        )
        return await _finish_run(
            emitter,
            state.result(
                status="failed",
                stop_reason="internal_error",
                messages=new_messages,
                final_message=_last_assistant(new_messages),
                error=error,
            ),
        )


async def _run_loop(
    current_context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    emitter: AgentEventEmitter,
    state: RunState,
    *,
    signal: Any | None,
    stream_fn: StreamFn | None,
) -> AgentRunResult:
    llm_runner = LLMStreamRunner(config=config, emitter=emitter, stream_fn=stream_fn)
    tool_coordinator = ToolCallCoordinator(config=config, emitter=emitter)
    first_iteration = True
    pending_messages = await _drain(config.get_steering_messages)
    model_retries = 0

    while True:
        has_more_tool_calls = True
        while has_more_tool_calls or pending_messages:
            if first_iteration:
                first_iteration = False
            else:
                await emitter.emit({"type": "turn_start"})

            if pending_messages:
                await _inject_messages(
                    pending_messages,
                    current_context,
                    new_messages,
                    emitter,
                )
                pending_messages = []

            state.counters.model_attempts += 1
            assistant = await llm_runner.stream_assistant_response(
                current_context,
                signal=signal,
            )
            new_messages.append(assistant)

            if assistant.stop_reason == "error":
                await emitter.emit(
                    {"type": "turn_end", "message": assistant, "toolResults": []}
                )
                error = assistant.error_info or ErrorInfo(
                    code="llm.unknown",
                    message=assistant.error_message or "Unknown model error",
                    retryable=False,
                    source="llm",
                )
                if (
                    config.retry_enabled
                    and error.retryable
                    and model_retries < config.max_model_retries
                ):
                    model_retries += 1
                    delay_ms = int(config.retry_base_delay_ms * (2 ** (model_retries - 1)))
                    await emitter.emit(
                        {
                            "type": "model_retry_start",
                            "attempt": model_retries,
                            "maxAttempts": config.max_model_retries + 1,
                            "delayMs": delay_ms,
                            "error": error,
                        }
                    )
                    if (
                        current_context.messages
                        and current_context.messages[-1] is assistant
                    ):
                        current_context.messages.pop()
                    await asyncio.sleep(delay_ms / 1000.0)
                    continue
                return await _finish_run(
                    emitter,
                    state.result(
                        status="failed",
                        stop_reason="model_error",
                        messages=new_messages,
                        final_message=assistant,
                        error=error,
                    ),
                )

            tool_calls = [
                content for content in assistant.content if isinstance(content, ToolCall)
            ]
            has_more_tool_calls = bool(tool_calls)
            tool_results: list[ToolResultMessage] = []

            if has_more_tool_calls:
                if state.has_repeated_call(
                    tool_calls,
                    limit=config.repeated_tool_call_limit,
                ):
                    return await _stop_with_error(
                        emitter,
                        state,
                        new_messages,
                        assistant,
                        code="run.repeated_tool_call",
                        message="Stopped after repeated identical tool calls",
                        stop_reason="repeated_tool_call",
                    )
                if state.counters.tool_iterations >= config.max_tool_iterations:
                    assistant.stop_reason = "max_iterations"
                    return await _stop_with_error(
                        emitter,
                        state,
                        new_messages,
                        assistant,
                        code="run.max_iterations",
                        message=(
                            "Stopped after reaching "
                            f"max_tool_iterations={config.max_tool_iterations}"
                        ),
                        stop_reason="max_iterations",
                    )

                state.counters.tool_iterations += 1
                tool_results = await tool_coordinator.execute_batch(
                    current_context,
                    assistant,
                    signal=signal,
                )
                state.collect_tool_results(tool_results)
                current_context.messages.extend(tool_results)
                new_messages.extend(tool_results)

            await emitter.emit(
                {"type": "turn_end", "message": assistant, "toolResults": tool_results}
            )

            if any(result.status == "approval_required" for result in tool_results):
                return await _finish_run(
                    emitter,
                    state.result(
                        status="waiting_approval",
                        stop_reason="approval_required",
                        messages=new_messages,
                        final_message=assistant,
                    ),
                )
            if any(result.status == "cancelled" for result in tool_results):
                return await _finish_run(
                    emitter,
                    state.result(
                        status="aborted",
                        stop_reason="aborted",
                        messages=new_messages,
                        final_message=assistant,
                    ),
                )
            pending_messages = await _drain(config.get_steering_messages)

        followups = await _drain(config.get_follow_up_messages)
        if followups:
            pending_messages = followups
            continue
        return await _finish_run(
            emitter,
            state.result(
                status="completed",
                stop_reason="final_answer",
                messages=new_messages,
                final_message=_last_assistant(new_messages),
            ),
        )


async def _stop_with_error(
    emitter: AgentEventEmitter,
    state: RunState,
    messages: list[AgentMessage],
    assistant: AssistantMessage,
    *,
    code: str,
    message: str,
    stop_reason: AgentRunStopReason,
) -> AgentRunResult:
    assistant.error_message = message
    error = ErrorInfo(
        code=code,
        message=message,
        retryable=False,
        source="runtime",
    )
    await emitter.emit(
        {
            "type": "error",
            "error": code,
            "message": message,
            "source": "runtime",
            "code": code,
            "retryable": False,
            "errorInfo": error,
        }
    )
    await emitter.emit({"type": "turn_end", "message": assistant, "toolResults": []})
    return await _finish_run(
        emitter,
        state.result(
            status="failed",
            stop_reason=stop_reason,
            messages=messages,
            final_message=assistant,
            error=error,
        ),
    )


async def _finish_run(
    emitter: AgentEventEmitter,
    result: AgentRunResult,
) -> AgentRunResult:
    await emitter.emit(
        {
            "type": "agent_end",
            "messages": result.messages,
            "status": result.status,
            "stopReason": result.stop_reason,
            "counters": result.counters,
            "result": result,
        }
    )
    return result


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


async def _drain(callback):
    if callback is None:
        return []
    return await maybe_await(callback())


def _last_assistant(messages: list[AgentMessage]) -> AssistantMessage | None:
    return next(
        (message for message in reversed(messages) if isinstance(message, AssistantMessage)),
        None,
    )
