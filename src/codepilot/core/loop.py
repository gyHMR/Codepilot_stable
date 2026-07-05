from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from codepilot.protocols import (
    AgentEvent,
    AgentRunCounters,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from codepilot.tools.ports import ToolObservation, ToolResumeDecision

from .contracts import (
    AgentLoopInput,
    AgentLoopOutcome,
    AgentLoopPorts,
    AgentResumeInput,
)
from .events import now_ms
from .model_turn import run_model_turn, tool_catalog_for_request
from .stopping import (
    completion_decision,
    completed_outcome,
    last_assistant,
    post_tool_decision,
    tool_call_signature,
    tool_execution_gate,
)
from .task_runtime import AgentTaskRuntime, with_task_context
from .tool_turn import (
    approval_observations,
    execute_tool_turn,
    to_tool_result_message,
    verification,
    workspace_effects,
)


async def run_agent_loop(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
) -> AgentLoopOutcome:
    recorder = _LoopEventRecorder(input, ports)
    recorder.emit({"type": "agent_start"})
    recorder.emit({"type": "turn_start"})

    if ports.model is None:
        assistant = AssistantMessage(content=[TextContent(text=input.user_prompt or "")])
        _emit_message(recorder, assistant)
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
        recorder.emit({"type": "agent_end", "status": "completed"})
        return completed_outcome(
            input.run_id,
            [assistant],
            assistant,
            recorder.events,
            model_attempts=0,
        )

    messages = list(input.messages)
    if input.user_prompt is not None:
        user_message = UserMessage(content=input.user_prompt)
        messages.append(user_message)
        _emit_message(recorder, user_message)
    task_runtime = AgentTaskRuntime.from_strategy(
        strategy=input.task_strategy,
        messages=messages,
        run_id=input.run_id,
        session_id=input.correlation.session_id,
    )
    if task_runtime is not None:
        recorder.emit(
            {
                "type": "task_plan_created",
                "task": task_runtime.event_payload(),
            }
        )

    return await _run_loop_body(
        input=input,
        ports=ports,
        recorder=recorder,
        messages=messages,
        task_runtime=task_runtime,
    )


async def resume_agent_loop(
    input: AgentResumeInput,
    ports: AgentLoopPorts,
) -> AgentLoopOutcome:
    if ports.tools is None:
        return AgentLoopOutcome(
            run_id=input.run_id,
            status="failed",
            stop_reason="missing_tool_port",
            error={"code": "core.missing_tool_port"},
        )
    if not input.approval_id or not input.decision:
        return AgentLoopOutcome(
            run_id=input.run_id,
            status="failed",
            stop_reason="missing_approval_decision",
            error={"code": "core.missing_approval_decision"},
        )

    loop_input = AgentLoopInput(
        run_id=input.run_id,
        correlation=input.correlation,
        messages=list(input.messages),
        context=input.context,
        model=input.model,
        tools=input.tools,
        task_strategy=input.task_strategy,
        limits=input.limits,
        retry_policy=input.retry_policy,
    )
    recorder = _LoopEventRecorder(loop_input, ports)
    recorder.emit({"type": "agent_start"})
    recorder.emit({"type": "turn_start"})
    recorder.emit(
        {
            "type": "tool_execution_start",
            "toolCallId": input.approval_id,
            "toolName": "approval_resume",
            "args": {"approval_id": input.approval_id, "decision": input.decision},
            "source": "approval_resume",
        }
    )

    observation = await ports.tools.resume(
        ToolResumeDecision(
            approval_id=input.approval_id,
            decision=input.decision,  # type: ignore[arg-type]
            reason=input.reason,
        )
    )
    recorder.emit(_resume_tool_end_event(input.approval_id, observation))
    tool_message = to_tool_result_message(
        observation,
        approval_id=input.approval_id,
        approved=input.decision == "approve",
    )
    messages = [*input.messages, tool_message]
    _emit_message(recorder, tool_message)
    task_runtime = AgentTaskRuntime.from_strategy(
        strategy=input.task_strategy,
        messages=messages,
        run_id=input.run_id,
        session_id=input.correlation.session_id,
    )
    if task_runtime is not None:
        task_runtime.after_tool_results([tool_message])
        recorder.emit(
            {
                "type": "task_step_updated",
                "task": task_runtime.event_payload(),
            }
        )

    return await _run_loop_body(
        input=loop_input,
        ports=ports,
        recorder=recorder,
        messages=messages,
        task_runtime=task_runtime,
        initial_new_messages=[tool_message],
        initial_observations=[observation],
        initial_tool_iterations=1,
        initial_tool_calls=1,
    )


async def _run_loop_body(
    *,
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    recorder: "_LoopEventRecorder",
    messages: list[Any],
    task_runtime: AgentTaskRuntime | None,
    initial_new_messages: list[ToolResultMessage] | None = None,
    initial_observations: list[ToolObservation] | None = None,
    initial_tool_iterations: int = 0,
    initial_tool_calls: int = 0,
) -> AgentLoopOutcome:
    new_messages: list[Any] = list(initial_new_messages or [])
    model_attempts = 0
    tool_iterations = initial_tool_iterations
    tool_calls_count = initial_tool_calls
    all_observations: list[ToolObservation] = list(initial_observations or [])
    usage = None
    max_model_turns = max(1, input.limits.max_model_turns)
    first_model_turn = True
    model_retries = 0
    last_tool_signature = None
    repeated_tool_signature_count = 0
    final_verification_grace_used = False

    for model_turn_index in range(max_model_turns):
        if first_model_turn:
            first_model_turn = False
        else:
            recorder.emit({"type": "turn_start"})

        while True:
            model_turn = await run_model_turn(
                _with_current_task(input, task_runtime),
                ports,
                messages,
            )
            model_attempts += 1
            if model_turn.error is None:
                break
            retry = _next_model_retry(input.retry_policy, model_retries)
            if retry is not None:
                model_retries += 1
                recorder.emit(
                    {
                        "type": "model_retry_start",
                        "attempt": model_retries,
                        "maxAttempts": _max_model_attempts(input.retry_policy),
                        "delayMs": retry,
                        "error": model_turn.error,
                    }
                )
                if retry > 0:
                    await asyncio.sleep(retry / 1000.0)
                continue
            recorder.emit({"type": "error", "error": model_turn.error})
            recorder.emit({"type": "turn_end", "message": None, "toolResults": []})
            recorder.emit({"type": "agent_end", "status": "failed"})
            return AgentLoopOutcome(
                run_id=input.run_id,
                status="failed",
                stop_reason="model_error",
                new_messages=list(new_messages),
                counters=AgentRunCounters(
                    model_attempts=model_attempts,
                    tool_iterations=tool_iterations,
                    tool_calls=tool_calls_count,
                ),
                workspace_effects=workspace_effects(all_observations),
                events=recorder.events,
                task=_task_summary(
                    task_runtime,
                    observations=all_observations,
                ),
                error=model_turn.error,
            )

        assistant = model_turn.message
        usage = model_turn.usage
        messages.append(assistant)
        new_messages.append(assistant)
        _emit_message(recorder, assistant)

        tool_calls = [block for block in assistant.content if isinstance(block, ToolCall)]
        if not tool_calls or ports.tools is None:
            task, check = _complete_task_if_needed(
                task_runtime,
                observations=all_observations,
            )
            if task is not None:
                recorder.emit(
                    {
                        "type": "completion_checked",
                        "task": task_runtime.event_payload() if task_runtime else {},
                        "completion_satisfied": task.completion_satisfied,
                        "completion_reason": task.completion_reason,
                    }
                )
                decision = completion_decision(check)
                if decision.should_continue:
                    steering = task_runtime.completion_steering(check) if task_runtime else None
                    if steering is not None and model_turn_index + 1 < max_model_turns:
                        messages.append(steering)
                        new_messages.append(steering)
                        _emit_message(recorder, steering)
                        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
                        continue
                    decision = completion_decision(_without_continuation_budget(check))
                if decision.should_stop:
                    recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
                    recorder.emit({"type": "agent_end", "status": decision.status})
                    return AgentLoopOutcome(
                        run_id=input.run_id,
                        status=decision.status or "waiting_user",
                        stop_reason=decision.stop_reason or "task_incomplete",
                        new_messages=list(new_messages),
                        final_message=assistant,
                        counters=AgentRunCounters(
                            model_attempts=model_attempts,
                            tool_iterations=tool_iterations,
                            tool_calls=tool_calls_count,
                        ),
                        usage=usage,
                        verification=verification(all_observations),
                        workspace_effects=workspace_effects(all_observations),
                        events=recorder.events,
                        task=task,
                    )
            recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
            recorder.emit({"type": "agent_end", "status": "completed"})
            return completed_outcome(
                input.run_id,
                list(new_messages),
                assistant,
                recorder.events,
                model_attempts=model_attempts,
                tool_iterations=tool_iterations,
                tool_calls=tool_calls_count,
                usage=usage,
                observations=all_observations,
                task=task,
            )

        if (
            input.limits.max_tool_calls is not None
            and tool_calls_count + len(tool_calls) > input.limits.max_tool_calls
        ):
            recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
            recorder.emit({"type": "agent_end", "status": "failed"})
            return AgentLoopOutcome(
                run_id=input.run_id,
                status="failed",
                stop_reason="tool_call_limit",
                new_messages=list(new_messages),
                final_message=assistant,
                counters=AgentRunCounters(
                    model_attempts=model_attempts,
                    tool_iterations=tool_iterations,
                    tool_calls=tool_calls_count,
                ),
                usage=usage,
                workspace_effects=workspace_effects(all_observations),
                events=recorder.events,
                task=_task_summary(
                    task_runtime,
                    observations=all_observations,
                ),
                )

        if (
            input.limits.max_tool_calls_per_turn is not None
            and len(tool_calls) > input.limits.max_tool_calls_per_turn
        ):
            error = {
                "code": "run.max_tool_calls_per_turn",
                "message": (
                    "Stopped after model requested "
                    f"{len(tool_calls)} tool calls in one turn "
                    f"(max={input.limits.max_tool_calls_per_turn})"
                ),
            }
            recorder.emit({"type": "error", "error": error})
            recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
            recorder.emit({"type": "agent_end", "status": "failed"})
            return AgentLoopOutcome(
                run_id=input.run_id,
                status="failed",
                stop_reason="tool_call_limit",
                new_messages=list(new_messages),
                final_message=assistant,
                counters=AgentRunCounters(
                    model_attempts=model_attempts,
                    tool_iterations=tool_iterations,
                    tool_calls=tool_calls_count,
                ),
                usage=usage,
                verification=verification(all_observations),
                workspace_effects=workspace_effects(all_observations),
                events=recorder.events,
                task=_task_snapshot(task_runtime),
                error=error,
            )

        signature = tool_call_signature(tool_calls)
        if signature == last_tool_signature:
            repeated_tool_signature_count += 1
        else:
            last_tool_signature = signature
            repeated_tool_signature_count = 1
        gate = tool_execution_gate(
            tool_iterations=tool_iterations,
            tool_calls=tool_calls,
            current_signature=signature,
            repeated_count=repeated_tool_signature_count,
            max_tool_iterations=input.limits.max_tool_iterations,
            repeated_tool_call_limit=input.limits.repeated_tool_call_limit,
        )
        if not gate.should_execute:
            if (
                gate.reason == "max_iterations"
                and not final_verification_grace_used
                and task_runtime is not None
                and task_runtime.needs_final_verification_grace(tool_calls)
            ):
                final_verification_grace_used = True
                recorder.emit(
                    {
                        "type": "tool_execution_grace",
                        "reason": "final_verification_at_iteration_limit",
                        "max_tool_iterations": input.limits.max_tool_iterations,
                        "tool_calls": [
                            {"id": call.id, "name": call.name}
                            for call in tool_calls
                        ],
                    }
                )
            else:
                error = {"code": gate.code, "message": gate.message}
                assistant.stop_reason = gate.stop_reason or assistant.stop_reason
                recorder.emit({"type": "error", "error": error})
                recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
                recorder.emit({"type": "agent_end", "status": "failed"})
                return AgentLoopOutcome(
                    run_id=input.run_id,
                    status="failed",
                    stop_reason=gate.stop_reason or gate.reason,
                    new_messages=list(new_messages),
                    final_message=assistant,
                    counters=AgentRunCounters(
                        model_attempts=model_attempts,
                        tool_iterations=tool_iterations,
                        tool_calls=tool_calls_count,
                    ),
                    usage=usage,
                    verification=verification(all_observations),
                    workspace_effects=workspace_effects(all_observations),
                    events=recorder.events,
                    task=_task_snapshot(task_runtime),
                    error=error,
                )

        observations = await execute_tool_turn(
            run_id=input.run_id,
            session_id=input.correlation.session_id,
            assistant_message=assistant,
            messages=list(messages),
            system_prompt=str(input.context.get("system_prompt", "")),
            available_tools=tool_catalog_for_request(input, ports),
            task_signal=_task_signal(input, task_runtime),
            metadata={
                "model_provider": input.model.provider,
                "model_id": input.model.model_id,
            },
            tools=ports.tools,
            tool_calls=tool_calls,
            emit=recorder.emit,
        )
        all_observations.extend(observations)
        tool_iterations += 1
        tool_calls_count += len(tool_calls)

        approvals = approval_observations(observations)
        if approvals:
            if task_runtime is not None:
                approval_messages = [
                    to_tool_result_message(
                        observation,
                        approval_id=(
                            observation.interruption.approval_id
                            if observation.interruption is not None
                            else None
                        ),
                        approved=False,
                    )
                    for observation in approvals
                ]
                task_runtime.after_tool_results(approval_messages)
                recorder.emit(
                    {
                        "type": "task_step_updated",
                        "task": task_runtime.event_payload(),
                    }
                )
            recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
            recorder.emit({"type": "agent_end", "status": "waiting_approval"})
            return AgentLoopOutcome(
                run_id=input.run_id,
                status="waiting_approval",
                stop_reason="approval_required",
                new_messages=list(new_messages),
                final_message=assistant,
                interruptions=[item.interruption for item in approvals if item.interruption],
                counters=AgentRunCounters(
                    model_attempts=model_attempts,
                    tool_iterations=tool_iterations,
                    tool_calls=tool_calls_count,
                ),
                usage=usage,
                verification=verification(all_observations),
                workspace_effects=workspace_effects(all_observations),
                events=recorder.events,
                task=_task_snapshot(task_runtime),
            )

        tool_messages = [to_tool_result_message(observation) for observation in observations]
        for message in tool_messages:
            messages.append(message)
            new_messages.append(message)
            _emit_message(recorder, message)
        if task_runtime is not None:
            task_decision = task_runtime.after_tool_results(tool_messages)
            recorder.emit(
                {
                    "type": "task_step_updated",
                    "task": task_runtime.event_payload(),
                }
            )
            recorder.emit(
                {
                    "type": "task_decision",
                    "decision": {
                        "action": task_decision.action,
                        "reason": task_decision.reason,
                        "next_action": task_decision.next_action,
                    },
                    "task": task_runtime.event_payload(),
                }
            )
            post_decision = post_tool_decision(tool_messages, task_decision)
            if post_decision.force_completion_check:
                task, check = _complete_task_if_needed(
                    task_runtime,
                    observations=all_observations,
                )
                recorder.emit(
                    {
                        "type": "completion_checked",
                        "task": task_runtime.event_payload(),
                        "completion_satisfied": task.completion_satisfied if task else False,
                        "completion_reason": task.completion_reason if task else "",
                    }
                )
                if task is not None and check is not None and completion_decision(check).should_stop:
                    final_decision = completion_decision(check)
                    recorder.emit({"type": "turn_end", "message": assistant, "toolResults": tool_messages})
                    recorder.emit({"type": "agent_end", "status": final_decision.status})
                    return AgentLoopOutcome(
                        run_id=input.run_id,
                        status=final_decision.status or "waiting_user",
                        stop_reason=final_decision.stop_reason or "task_incomplete",
                        new_messages=list(new_messages),
                        final_message=assistant,
                        counters=AgentRunCounters(
                            model_attempts=model_attempts,
                            tool_iterations=tool_iterations,
                            tool_calls=tool_calls_count,
                        ),
                        usage=usage,
                        verification=verification(all_observations),
                        workspace_effects=workspace_effects(all_observations),
                        events=recorder.events,
                        task=task,
                    )
                recorder.emit({"type": "turn_end", "message": assistant, "toolResults": tool_messages})
                recorder.emit({"type": "agent_end", "status": "completed"})
                return completed_outcome(
                    input.run_id,
                    list(new_messages),
                    assistant,
                    recorder.events,
                    model_attempts=model_attempts,
                    tool_iterations=tool_iterations,
                    tool_calls=tool_calls_count,
                    usage=usage,
                    observations=all_observations,
                    task=task,
                )
            if post_decision.should_stop:
                recorder.emit({"type": "turn_end", "message": assistant, "toolResults": tool_messages})
                recorder.emit({"type": "agent_end", "status": post_decision.status})
                return AgentLoopOutcome(
                    run_id=input.run_id,
                    status=post_decision.status or "waiting_user",
                    stop_reason=post_decision.stop_reason or "task_blocked",
                    new_messages=list(new_messages),
                    final_message=assistant,
                    counters=AgentRunCounters(
                        model_attempts=model_attempts,
                        tool_iterations=tool_iterations,
                        tool_calls=tool_calls_count,
                    ),
                    usage=usage,
                    verification=verification(all_observations),
                    workspace_effects=workspace_effects(all_observations),
                    events=recorder.events,
                    task=_task_snapshot(task_runtime),
                    error=post_decision.error,
                )
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": tool_messages})

    recorder.emit({"type": "agent_end", "status": "failed"})
    return AgentLoopOutcome(
        run_id=input.run_id,
        status="failed",
        stop_reason="max_model_turns",
        new_messages=list(new_messages),
        final_message=last_assistant(new_messages),
        counters=AgentRunCounters(
            model_attempts=model_attempts,
            tool_iterations=tool_iterations,
            tool_calls=tool_calls_count,
        ),
        usage=usage,
        verification=verification(all_observations),
        workspace_effects=workspace_effects(all_observations),
        events=recorder.events,
        task=_task_summary(
            task_runtime,
            observations=all_observations,
        ),
    )


class _LoopEventRecorder:
    def __init__(self, input: AgentLoopInput, ports: AgentLoopPorts) -> None:
        self._input = input
        self._ports = ports
        self.events: list[AgentEvent] = []
        self._event_seq = 0
        self._turn_id = 0

    def emit(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", "")).strip()
        if event_type == "turn_start":
            self._turn_id += 1
        self._event_seq += 1
        enriched = {
            **event,
            "type": event_type,
            "runId": self._input.run_id,
            "sessionId": self._input.correlation.session_id,
            "turnId": self._turn_id,
            "eventId": f"{self._input.run_id}:{self._event_seq}",
            "timestamp": now_ms(),
        }
        self.events.append(enriched)  # type: ignore[arg-type]
        if self._ports.events is not None:
            self._ports.events(enriched)


def _emit_message(recorder: _LoopEventRecorder, message: Any) -> None:
    recorder.emit({"type": "message_start", "message": message})
    recorder.emit({"type": "message_end", "message": message})


def _with_current_task(
    input: AgentLoopInput,
    task_runtime: AgentTaskRuntime | None,
) -> AgentLoopInput:
    if task_runtime is None:
        return input
    return replace(input, context=with_task_context(input.context, task_runtime))


def _complete_task_if_needed(
    task_runtime: AgentTaskRuntime | None,
    *,
    observations: list[ToolObservation],
) -> tuple[Any | None, Any | None]:
    if task_runtime is None:
        return None, None
    effects = workspace_effects(observations)
    task_runtime.merge_observation_summary(
        affected_paths=effects.affected_paths,
        workspace_changed=effects.changed,
        verification=verification(observations),
    )
    return task_runtime.complete()


def _task_summary(
    task_runtime: AgentTaskRuntime | None,
    *,
    observations: list[ToolObservation],
):
    task, _check = _complete_task_if_needed(
        task_runtime,
        observations=observations,
    )
    return task


def _task_snapshot(task_runtime: AgentTaskRuntime | None):
    if task_runtime is None:
        return None
    return task_runtime.snapshot()


def _next_model_retry(policy: dict[str, Any], retries_so_far: int) -> int | None:
    if not bool(policy.get("enabled", False)):
        return None
    max_retries = _non_negative_int(policy.get("max_retries"), default=0)
    if retries_so_far >= max_retries:
        return None
    base_delay_ms = _non_negative_int(policy.get("base_delay_ms"), default=0)
    return int(base_delay_ms * (2 ** retries_so_far))


def _max_model_attempts(policy: dict[str, Any]) -> int:
    return 1 + _non_negative_int(policy.get("max_retries"), default=0)


def _non_negative_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(0, value)


def _without_continuation_budget(check: Any) -> Any:
    return replace(check, can_continue=False)


def _task_signal(
    input: AgentLoopInput,
    task_runtime: AgentTaskRuntime | None,
) -> dict[str, Any]:
    if task_runtime is not None:
        return dict(task_runtime.event_payload())
    signal = input.context.get("task_signal")
    if isinstance(signal, dict):
        return dict(signal)
    return {}


def _resume_tool_end_event(
    approval_id: str,
    observation: ToolObservation,
) -> dict[str, Any]:
    metadata = dict(observation.metadata)
    metadata.setdefault("approval_id", approval_id)
    return {
        "type": "tool_execution_end",
        "toolCallId": observation.tool_call_id or approval_id,
        "toolName": observation.name or "approval_resume",
        "status": observation.status,
        "isError": observation.status != "success",
        "approved": observation.status == "success",
        "approvalId": approval_id,
        "errorReason": metadata.get("error_code"),
        "affectedPaths": list(observation.affected_paths),
        "workspaceChanged": observation.workspace_changed,
        "verification": list(observation.verification),
        "result": {
            "content": list(observation.content),
            "metadata": metadata,
        },
    }
