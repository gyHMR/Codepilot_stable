from __future__ import annotations

"""一次完整的 Agent Run：用户提示 → 模型推理 → 工具执行 → 返回 RunResult。"""

import asyncio
from typing import Any

from codepilot.protocols import AssistantMessage, ToolCall, ToolResultMessage
from codepilot.protocols import (
    AgentEventSink,
    AgentRunResult,
    AgentRunStopReason,
    TaskSummary,
    ErrorInfo,
)

from .events import AgentEventEmitter, maybe_await
from .llm_runner import LLMStreamRunner, StreamFn
from .run_state import RunState, new_run_id
from .task_controller import TaskController
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
    """运行一个新的用户任务并返回结构化结果。"""

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
        current_task=context.current_task,
        recovered_task=context.recovered_task,
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
    """继续一个内存中尚未完成的 Run（例如需要执行工具调用后继续推理）。"""

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
        current_task=context.current_task,
        recovered_task=context.recovered_task,
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


# ── 内部函数 ────────────────────────────────────────────────────

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
    """核心执行循环：模型推理 → 工具执行 → 任务状态更新，直到任务完成或出错。"""
    llm_runner = LLMStreamRunner(config=config, emitter=emitter, stream_fn=stream_fn)
    tool_coordinator = ToolCallCoordinator(config=config, emitter=emitter)
    task_controller = TaskController() if config.task_control_enabled else None
    task = (
        task_controller.initialize(
            current_context.messages,
            recovered_task=current_context.recovered_task,
        )
        if task_controller is not None
        else None
    )
    if task_controller is not None and task is not None:
        await emitter.emit(
            {
                "type": "task_plan_created",
                "task": task_controller.event_payload(task),
            }
        )
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

            current_context.current_task = (
                task_controller.render_context(task)
                if task_controller is not None and task is not None
                else None
            )
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
                decision = (
                    task_controller.after_tool_results(task, state, tool_results)
                    if task_controller is not None and task is not None
                    else None
                )
                if task_controller is not None and task is not None and decision is not None:
                    await emitter.emit(
                        {
                            "type": "task_step_updated",
                            "task": task_controller.event_payload(task),
                        }
                    )
                    await emitter.emit(
                        {
                            "type": "task_decision",
                            "decision": {
                                "action": decision.action,
                                "reason": decision.reason,
                                "next_action": decision.next_action,
                            },
                            "task": task_controller.event_payload(task),
                        }
                    )
                current_context.messages.extend(tool_results)
                new_messages.extend(tool_results)
                if decision is not None and decision.action == "stop":
                    return await _stop_with_error(
                        emitter,
                        state,
                        new_messages,
                        assistant,
                        code="run.replan_limit",
                        message=decision.reason,
                        stop_reason="replan_limit",
                        task=(
                            task_controller.summarize(task)
                            if task_controller is not None and task is not None
                            else None
                        ),
                    )

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
                        task=(
                            task_controller.summarize(task)
                            if task_controller is not None and task is not None
                            else None
                        ),
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
                        task=(
                            task_controller.summarize(task)
                            if task_controller is not None and task is not None
                            else None
                        ),
                    ),
                )
            pending_messages = await _drain(config.get_steering_messages)

        if task_controller is not None and task is not None:
            completion = task_controller.check_completion(task, state)
            await emitter.emit(
                {
                    "type": "completion_checked",
                    "completion": {
                        "satisfied": completion.satisfied,
                        "reason": completion.reason,
                        "missing": list(completion.missing),
                        "can_continue": completion.can_continue,
                        "unverified": completion.unverified,
                    },
                    "task": task_controller.event_payload(task),
                }
            )
            if not completion.satisfied and completion.can_continue:
                pending_messages = [task_controller.completion_steering(completion)]
                continue

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
                task=(
                    task_controller.summarize(task)
                    if task_controller is not None and task is not None
                    else None
                ),
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
    task: TaskSummary | None = None,
) -> AgentRunResult:
    """因错误终止运行：记录错误信息，发射 error 事件，返回失败状态的 RunResult。"""
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
            task=task,
        ),
    )


async def _finish_run(
    emitter: AgentEventEmitter,
    result: AgentRunResult,
) -> AgentRunResult:
    """完成运行：发射 agent_end 事件并返回最终结果。"""
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
    """将引导/后续消息注入上下文：发射 message_start/end 事件后追加到消息列表。"""
    for message in messages:
        await emitter.emit({"type": "message_start", "message": message})
        await emitter.emit({"type": "message_end", "message": message})
        current_context.messages.append(message)
        new_messages.append(message)


async def _drain(callback):
    """排空回调队列：调用回调并返回结果列表，回调为 None 时返回空列表。"""
    if callback is None:
        return []
    return await maybe_await(callback())


def _last_assistant(messages: list[AgentMessage]) -> AssistantMessage | None:
    """从消息列表末尾反向查找最近一条助手消息。"""
    return next(
        (message for message in reversed(messages) if isinstance(message, AssistantMessage)),
        None,
    )
