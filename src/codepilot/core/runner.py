from __future__ import annotations

"""Agent runner: model/tool loop with soft plan state and final-answer guard."""

import asyncio
import time
from dataclasses import replace
from typing import Any

from codepilot.protocols import (
    AgentEvent,
    AgentEventSink,
    AgentRunCounters,
    AgentRunStatus,
    AgentRunStopReason,
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    ensure_runtime_event_type,
)
from codepilot.tools.contracts import ToolInvocation, ToolObservation, ToolResumeDecision

from .contracts import (
    AgentLoopInput,
    AgentLoopOutcome,
    AgentLoopPorts,
    AgentResumeInput,
    PreparedContext,
    RetryPolicy,
    WorkspaceEffects,
)
from .model_step import ModelTurnResult, run_model_turn, tool_catalog_for_request
from .plan import (
    PlanState,
    PlanValidationError,
    apply_plan_update_metadata,
    ensure_run_mode,
    load_plan_state,
)
from .run_guard import RunGuard
from .state import RunState
from .tool_step import (
    approval_observations,
    execute_tool_turn,
    to_tool_result_message,
    verification,
    workspace_effects,
)


def now_ms() -> int:
    return int(time.time() * 1000)


async def maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


class AgentEventEmitter:
    """Small public helper for tests and adapters that need enriched events."""

    def __init__(
        self,
        sink: AgentEventSink,
        *,
        run_id: str,
        session_id: str | None = None,
    ) -> None:
        self._sink = sink
        self.run_id = _required_text(run_id, "run_id")
        self.session_id = _optional_text(session_id)
        self.turn_id = 0
        self._event_seq = 0

    async def emit(self, event: dict[str, Any]) -> None:
        event_type = ensure_runtime_event_type(event.get("type"))
        if event_type == "turn_start":
            self.turn_id += 1
        self._event_seq += 1
        enriched = {
            **event,
            "type": event_type,
            "runId": self.run_id,
            "sessionId": self.session_id,
            "turnId": self.turn_id,
            "eventId": f"{self.run_id}:{self._event_seq}",
            "timestamp": now_ms(),
        }
        await maybe_await(self._sink(enriched))  # type: ignore[arg-type]


async def run_agent_loop(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
) -> AgentLoopOutcome:
    recorder = _EventRecorder(input, ports)
    run_state = RunState(
        run_id=input.run_id,
        session_id=input.correlation.session_id,
    )
    plan_state = load_plan_state(input.plan_state)
    messages = list(input.messages)
    new_messages: list[Message] = []

    recorder.emit({"type": "agent_start"})
    recorder.emit({"type": "turn_start"})

    return await _drive_loop(
        input=input,
        ports=ports,
        recorder=recorder,
        messages=messages,
        new_messages=new_messages,
        run_state=run_state,
        plan_state=plan_state,
        observations=[],
        first_turn_started=True,
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
            signals=RunState(input.run_id, input.correlation.session_id).summary(),
            error={"code": "core.missing_tool_port"},
        )
    if not input.approval_id or not input.decision:
        return AgentLoopOutcome(
            run_id=input.run_id,
            status="failed",
            stop_reason="missing_approval_decision",
            signals=RunState(input.run_id, input.correlation.session_id).summary(),
            error={"code": "core.missing_approval_decision"},
        )

    loop_input = AgentLoopInput(
        run_id=input.run_id,
        correlation=input.correlation,
        messages=list(input.messages),
        context=input.context,
        model=input.model,
        tools=input.tools,
        mode=input.mode,
        plan_state=input.plan_state,
        limits=input.limits,
        retry_policy=input.retry_policy,
        event_start_seq=input.event_start_seq,
    )
    recorder = _EventRecorder(loop_input, ports)
    run_state = RunState(input.run_id, input.correlation.session_id)
    plan_state = load_plan_state(input.plan_state)
    messages = list(input.messages)
    new_messages: list[Message] = []

    recorder.emit({"type": "agent_start"})
    recorder.emit({"type": "turn_start"})
    tool_call_id = input.tool_call_id or input.approval_id
    tool_name = input.tool_name or "approval_resume"
    recorder.emit(
        {
            "type": "tool_started",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": {
                "approval_id": input.approval_id,
                "decision": input.decision,
                "tool_call_id": input.tool_call_id,
                "tool_name": input.tool_name,
            },
            "source": "approval_resume",
        }
    )

    observation = await _resume_tool_observation(input, ports)
    recorder.emit(_resume_tool_end_event(input.approval_id, observation))
    tool_message = to_tool_result_message(
        observation,
        approval_id=input.approval_id,
        approved=input.decision == "approve",
    )
    messages.append(tool_message)
    new_messages.append(tool_message)
    _emit_message(recorder, tool_message)
    run_state.collect_tool_results([tool_message])
    run_state.counters.tool_iterations += 1
    plan_state = _apply_plan_updates(
        plan_state,
        [tool_message],
        input=loop_input,
        recorder=recorder,
        objective=_objective_for_plan(loop_input, messages),
    )

    interruption = _interruption_after_tool_results(
        input=loop_input,
        recorder=recorder,
        assistant=last_assistant(messages),
        visible_tool_messages=[tool_message],
        new_messages=new_messages,
        run_state=run_state,
        plan_state=plan_state,
        observations=[observation],
        usage=None,
    )
    if interruption is not None:
        return interruption

    return await _drive_loop(
        input=loop_input,
        ports=ports,
        recorder=recorder,
        messages=messages,
        new_messages=new_messages,
        run_state=run_state,
        plan_state=plan_state,
        observations=[observation],
        first_turn_started=True,
    )


async def _resume_tool_observation(
    input: AgentResumeInput,
    ports: AgentLoopPorts,
) -> ToolObservation:
    approval_id = input.approval_id or ""
    if input.tool_call_id and input.tool_name:
        if input.decision == "deny":
            return ToolObservation(
                tool_call_id=input.tool_call_id,
                name=input.tool_name,
                status="denied",
                content=(
                    TextContent(text=input.reason or "Tool execution denied by user"),
                ),
                metadata={
                    "approval_id": approval_id,
                    "approved": False,
                    "error_code": "approval_denied",
                },
            )
        return await ports.tools.execute(
            ToolInvocation(
                run_id=input.run_id,
                tool_call_id=input.tool_call_id,
                name=input.tool_name,
                arguments=dict(input.arguments),
                current_mode=input.mode,
                source="approval_resume",
                assistant_message=last_assistant(input.messages),
            )
        )

    return await ports.tools.resume(
        ToolResumeDecision(
            approval_id=approval_id,
            decision=input.decision,  # type: ignore[arg-type]
            reason=input.reason,
        )
    )


async def _drive_loop(
    *,
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    recorder: "_EventRecorder",
    messages: list[Message],
    new_messages: list[Message],
    run_state: RunState,
    plan_state: PlanState | None,
    observations: list[ToolObservation],
    first_turn_started: bool,
) -> AgentLoopOutcome:
    usage = None
    max_model_turns = max(1, input.limits.max_model_turns)
    plan_summary_pending = False

    for turn_index in range(max_model_turns):
        if turn_index > 0 or not first_turn_started:
            recorder.emit({"type": "turn_start"})

        turn_input = (
            _with_plan_summary_context(input, plan_state)
            if plan_summary_pending
            else input
        )
        try:
            model_turn = await _model_turn_with_retries(
                input=turn_input,
                ports=ports,
                recorder=recorder,
                messages=messages,
                run_state=run_state,
                plan_state=plan_state,
            )
        except Exception:
            if not (plan_summary_pending and _is_pending_plan(plan_state)):
                raise
            assistant = _plan_summary_fallback_message(plan_state)
            messages.append(assistant)
            new_messages.append(assistant)
            _emit_message(recorder, assistant)
            return _plan_approval_pause_outcome(
                input=input,
                recorder=recorder,
                assistant=assistant,
                new_messages=new_messages,
                run_state=run_state,
                plan_state=plan_state,
                observations=observations,
                usage=usage,
            )
        if model_turn.error is not None:
            if plan_summary_pending and _is_pending_plan(plan_state):
                assistant = _plan_summary_fallback_message(plan_state)
                messages.append(assistant)
                new_messages.append(assistant)
                _emit_message(recorder, assistant)
                return _plan_approval_pause_outcome(
                    input=input,
                    recorder=recorder,
                    assistant=assistant,
                    new_messages=new_messages,
                    run_state=run_state,
                    plan_state=plan_state,
                    observations=observations,
                    usage=usage,
                )
            recorder.emit({"type": "error", "error": model_turn.error})
            recorder.emit({"type": "turn_end", "message": None, "toolResults": []})
            recorder.emit({"type": "agent_end", "status": "failed"})
            return _outcome(
                input=input,
                status="failed",
                stop_reason="model_error",
                new_messages=new_messages,
                final_message=None,
                run_state=run_state,
                plan_state=plan_state,
                observations=observations,
                events=recorder.events,
                usage=usage,
                error=model_turn.error,
            )

        assistant = model_turn.message
        usage = model_turn.usage
        messages.append(assistant)
        new_messages.append(assistant)
        _emit_message(recorder, assistant)

        tool_calls = [block for block in assistant.content if isinstance(block, ToolCall)]
        if not tool_calls:
            if plan_summary_pending and _is_pending_plan(plan_state):
                _mark_plan_summary_message(assistant, plan_state)
                return _plan_approval_pause_outcome(
                    input=input,
                    recorder=recorder,
                    assistant=assistant,
                    new_messages=new_messages,
                    run_state=run_state,
                    plan_state=plan_state,
                    observations=observations,
                    usage=usage,
                )
            outcome = _finish_or_steer(
                input=input,
                recorder=recorder,
                messages=messages,
                new_messages=new_messages,
                assistant=assistant,
                run_state=run_state,
                plan_state=plan_state,
                observations=observations,
                usage=usage,
                turn_index=turn_index,
                max_model_turns=max_model_turns,
            )
            if outcome is not None:
                return outcome
            continue

        if ports.tools is None:
            error = {
                "code": "core.missing_tool_port",
                "message": "Model requested tools but no ToolPort was provided",
            }
            recorder.emit({"type": "error", "error": error})
            recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
            recorder.emit({"type": "agent_end", "status": "failed"})
            return _outcome(
                input=input,
                status="failed",
                stop_reason="missing_tool_port",
                new_messages=new_messages,
                final_message=assistant,
                run_state=run_state,
                plan_state=plan_state,
                observations=observations,
                events=recorder.events,
                usage=usage,
                error=error,
            )

        limit_outcome = _tool_limit_outcome(
            input=input,
            recorder=recorder,
            assistant=assistant,
            tool_calls=tool_calls,
            new_messages=new_messages,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            usage=usage,
        )
        if limit_outcome is not None:
            return limit_outcome

        turn_observations = await execute_tool_turn(
            run_id=input.run_id,
            session_id=input.correlation.session_id,
            assistant_message=assistant,
            messages=list(messages),
            system_prompt=str(input.context.get("system_prompt", "")),
            available_tools=tool_catalog_for_request(input, ports),
            current_mode=input.mode,
            run_signals=_run_signal_payload(run_state),
            metadata={
                "model_provider": input.model.provider,
                "model_id": input.model.model_id,
            },
            tools=ports.tools,
            tool_calls=tool_calls,
            emit=recorder.emit,
        )
        observations.extend(turn_observations)
        run_state.counters.tool_iterations += 1

        tool_messages = _tool_messages_from_observations(turn_observations)
        run_state.collect_tool_results(tool_messages)
        visible_tool_messages = [
            message
            for message in tool_messages
            if message.status != "approval_required"
        ]
        for message in visible_tool_messages:
            messages.append(message)
            new_messages.append(message)
            _emit_message(recorder, message)

        plan_state = _apply_plan_updates(
            plan_state,
            visible_tool_messages,
            input=input,
            recorder=recorder,
            objective=_objective_for_plan(input, messages),
        )

        if input.mode == "plan" and _is_pending_plan(plan_state) and any(
            message.tool_name == "update_plan"
            and message.status == "success"
            and isinstance(message.metadata.get("plan_update"), dict)
            for message in visible_tool_messages
        ):
            plan_summary_pending = True

        recovery_instruction = run_state.truncated_read_recovery_instruction(visible_tool_messages)
        if recovery_instruction:
            steering = UserMessage(content=recovery_instruction)
            messages.append(steering)
            new_messages.append(steering)
            _emit_message(recorder, steering)

        interruption = _interruption_after_tool_results(
            input=input,
            recorder=recorder,
            assistant=assistant,
            visible_tool_messages=visible_tool_messages,
            new_messages=new_messages,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            usage=usage,
        )
        if interruption is not None:
            return interruption

        continuation_outcome = _tool_continuation_limit_outcome(
            input=input,
            recorder=recorder,
            turn_index=turn_index,
            max_model_turns=max_model_turns,
            visible_tool_messages=visible_tool_messages,
            new_messages=new_messages,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            usage=usage,
        )
        if continuation_outcome is not None:
            return continuation_outcome

        recorder.emit(
            {
                "type": "turn_end",
                "message": assistant,
                "toolResults": visible_tool_messages,
            }
        )

    recorder.emit({"type": "agent_end", "status": "failed"})
    return _outcome(
        input=input,
        status="failed",
        stop_reason="max_iterations",
        new_messages=new_messages,
        final_message=last_assistant(new_messages),
        run_state=run_state,
        plan_state=plan_state,
        observations=observations,
        events=recorder.events,
        usage=usage,
    )


async def _model_turn_with_retries(
    *,
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    recorder: "_EventRecorder",
    messages: list[Message],
    run_state: RunState,
    plan_state: PlanState | None,
) -> ModelTurnResult:
    retries = 0
    while True:
        result = await run_model_turn(
            _with_runtime_context(input, plan_state, run_state),
            ports,
            messages,
            emit=recorder.emit,
        )
        run_state.counters.model_attempts += 1
        if result.error is None:
            return result
        delay = _next_retry_delay(input.retry_policy, retries)
        if delay is None:
            return result
        retries += 1
        recorder.emit(
            {
                "type": "model_retry_start",
                "attempt": retries,
                "maxAttempts": 1 + input.retry_policy.max_retries,
                "delayMs": delay,
                "error": result.error,
            }
        )
        if delay > 0:
            await asyncio.sleep(delay / 1000.0)


def _finish_or_steer(
    *,
    input: AgentLoopInput,
    recorder: "_EventRecorder",
    messages: list[Message],
    new_messages: list[Message],
    assistant: AssistantMessage,
    run_state: RunState,
    plan_state: PlanState | None,
    observations: list[ToolObservation],
    usage: Any,
    turn_index: int,
    max_model_turns: int,
) -> AgentLoopOutcome | None:
    if input.mode == "plan":
        recorder.emit(
            {
                "type": "plan_clarification_required",
                "reason": "plan_mode_finished_without_published_plan",
            }
        )
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
        recorder.emit(
            {
                "type": "agent_end",
                "status": "waiting_user",
                "stopReason": "plan_clarification_required",
            }
        )
        return _outcome(
            input=input,
            status="waiting_user",
            stop_reason="plan_clarification_required",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            events=recorder.events,
            usage=usage,
        )
    decision = RunGuard().check(
        assistant=assistant,
        signals=run_state.summary(),
        mode=input.mode,
    )
    recorder.emit(
        {
            "type": "run_guard_checked",
            "decision": {
                "action": decision.action,
                "reason": decision.reason,
                "instruction": decision.instruction,
            },
            "signals": run_state.summary(),
        }
    )
    if decision.action == "completed":
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
        recorder.emit({"type": "agent_end", "status": "completed"})
        return _outcome(
            input=input,
            status="completed",
            stop_reason="final_answer",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            events=recorder.events,
            usage=usage,
        )
    if decision.action == "continue_with_instruction" and turn_index + 1 < max_model_turns:
        steering = UserMessage(content=decision.instruction)
        messages.append(steering)
        new_messages.append(steering)
        _emit_message(recorder, steering)
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
        return None
    status: AgentRunStatus = "failed" if decision.action == "stopped" else "waiting_user"
    recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
    recorder.emit({"type": "agent_end", "status": status})
    return _outcome(
        input=input,
        status=status,
        stop_reason=_guard_stop_reason(decision.reason),
        new_messages=new_messages,
        final_message=assistant,
        run_state=run_state,
        plan_state=plan_state,
        observations=observations,
        events=recorder.events,
        usage=usage,
    )


def _tool_limit_outcome(
    *,
    input: AgentLoopInput,
    recorder: "_EventRecorder",
    assistant: AssistantMessage,
    tool_calls: list[ToolCall],
    new_messages: list[Message],
    run_state: RunState,
    plan_state: PlanState | None,
    observations: list[ToolObservation],
    usage: Any,
) -> AgentLoopOutcome | None:
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
        return _outcome(
            input=input,
            status="failed",
            stop_reason="tool_call_limit",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            events=recorder.events,
            usage=usage,
            error=error,
        )

    if (
        input.limits.max_tool_calls is not None
        and run_state.counters.tool_calls + len(tool_calls) > input.limits.max_tool_calls
    ):
        error = {"code": "run.max_tool_calls", "message": "Tool call limit reached"}
        recorder.emit({"type": "error", "error": error})
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
        recorder.emit({"type": "agent_end", "status": "failed"})
        return _outcome(
            input=input,
            status="failed",
            stop_reason="tool_call_limit",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            events=recorder.events,
            usage=usage,
            error=error,
        )

    if (
        input.limits.max_tool_iterations >= 0
        and run_state.counters.tool_iterations >= input.limits.max_tool_iterations
    ):
        pause = max_iterations_pause_message(input.limits.max_tool_iterations)
        if new_messages and new_messages[-1] is assistant:
            new_messages.pop()
        new_messages.append(pause)
        _emit_message(recorder, pause)
        recorder.emit({"type": "turn_end", "message": pause, "toolResults": []})
        recorder.emit({"type": "agent_end", "status": "waiting_user"})
        return _outcome(
            input=input,
            status="waiting_user",
            stop_reason="max_iterations",
            new_messages=new_messages,
            final_message=pause,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            events=recorder.events,
            usage=usage,
        )

    if run_state.has_repeated_call(
        tool_calls,
        limit=input.limits.repeated_tool_call_limit,
    ):
        error = {"code": "run.repeated_tool_call", "message": "Repeated identical tool calls"}
        recorder.emit({"type": "error", "error": error})
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
        recorder.emit({"type": "agent_end", "status": "failed"})
        return _outcome(
            input=input,
            status="failed",
            stop_reason="repeated_tool_call",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            events=recorder.events,
            usage=usage,
            error=error,
        )

    return None


def _tool_continuation_limit_outcome(
    *,
    input: AgentLoopInput,
    recorder: "_EventRecorder",
    turn_index: int,
    max_model_turns: int,
    visible_tool_messages: list[ToolResultMessage],
    new_messages: list[Message],
    run_state: RunState,
    plan_state: PlanState | None,
    observations: list[ToolObservation],
    usage: Any,
) -> AgentLoopOutcome | None:
    if turn_index + 1 < max_model_turns:
        return None

    pause = model_turns_exhausted_pause_message(max_model_turns)
    new_messages.append(pause)
    _emit_message(recorder, pause)
    recorder.emit(
        {
            "type": "turn_end",
            "message": pause,
            "toolResults": visible_tool_messages,
        }
    )
    recorder.emit({"type": "agent_end", "status": "waiting_user"})
    return _outcome(
        input=input,
        status="waiting_user",
        stop_reason="max_iterations",
        new_messages=new_messages,
        final_message=pause,
        run_state=run_state,
        plan_state=plan_state,
        observations=observations,
        events=recorder.events,
        usage=usage,
    )


def _interruption_after_tool_results(
    *,
    input: AgentLoopInput,
    recorder: "_EventRecorder",
    assistant: AssistantMessage | None,
    visible_tool_messages: list[ToolResultMessage],
    new_messages: list[Message],
    run_state: RunState,
    plan_state: PlanState | None,
    observations: list[ToolObservation],
    usage: Any,
) -> AgentLoopOutcome | None:
    approvals = approval_observations(observations)
    if approvals:
        recorder.emit(
            {
                "type": "turn_end",
                "message": assistant,
                "toolResults": visible_tool_messages,
            }
        )
        recorder.emit({"type": "agent_end", "status": "waiting_approval"})
        return _outcome(
            input=input,
            status="waiting_approval",
            stop_reason="approval_required",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            events=recorder.events,
            usage=usage,
            interruptions=[item.interruption for item in approvals if item.interruption],
        )
    if any(message.status == "cancelled" for message in visible_tool_messages):
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": visible_tool_messages})
        recorder.emit({"type": "agent_end", "status": "aborted"})
        return _outcome(
            input=input,
            status="aborted",
            stop_reason="aborted",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            events=recorder.events,
            usage=usage,
        )
    if run_state.tool_unavailable:
        error = run_state.last_error or {"code": "tool_unavailable"}
        recorder.emit({"type": "error", "error": error})
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": visible_tool_messages})
        recorder.emit({"type": "agent_end", "status": "failed"})
        return _outcome(
            input=input,
            status="failed",
            stop_reason="tool_unavailable",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            events=recorder.events,
            usage=usage,
            error=error,
        )
    return None


def _plan_approval_pause_outcome(
    *,
    input: AgentLoopInput,
    recorder: "_EventRecorder",
    assistant: AssistantMessage,
    new_messages: list[Message],
    run_state: RunState,
    plan_state: PlanState | None,
    observations: list[ToolObservation],
    usage: Any,
) -> AgentLoopOutcome:
    _mark_plan_summary_message(assistant, plan_state)
    recorder.emit(
        {
            "type": "plan_approval_required",
            "plan": plan_state.to_dict() if plan_state is not None else None,
            "reason": "proposed_plan_waiting_for_user_approval",
        }
    )
    recorder.emit(
        {
            "type": "turn_end",
            "message": assistant,
            "toolResults": [],
        }
    )
    recorder.emit(
        {
            "type": "agent_end",
            "status": "waiting_user",
            "stopReason": "plan_approval_required",
        }
    )
    return _outcome(
        input=input,
        status="waiting_user",
        stop_reason="plan_approval_required",
        new_messages=new_messages,
        final_message=assistant,
        run_state=run_state,
        plan_state=plan_state,
        observations=observations,
        events=recorder.events,
        usage=usage,
    )


def _is_pending_plan(plan_state: PlanState | None) -> bool:
    return (
        plan_state is not None
        and plan_state.status == "proposed"
        and plan_state.approval_state == "pending"
    )


def _mark_plan_summary_message(
    message: AssistantMessage,
    plan_state: PlanState | None,
) -> None:
    message.metadata["message_kind"] = "plan_summary"
    if plan_state is not None:
        message.metadata["plan_id"] = plan_state.plan_id
        message.metadata["plan_revision"] = plan_state.revision


def _plan_summary_fallback_message(plan_state: PlanState) -> AssistantMessage:
    lines = [
        f"计划：{plan_state.objective}",
        plan_state.summary,
        *[
            (
                f"{index}. {item.step}\n"
                f"   动作：{item.details}\n"
                f"   验证：{item.verification}"
            )
            for index, item in enumerate(plan_state.items, start=1)
        ],
        "请使用 /approve 批准，使用 /reject 拒绝，或直接说明需要修改的内容。",
    ]
    message = AssistantMessage(
        content=[TextContent(text="\n".join(line for line in lines if line))],
        stop_reason="stop",
    )
    message.metadata["summary_fallback"] = True
    _mark_plan_summary_message(message, plan_state)
    return message


def _outcome(
    *,
    input: AgentLoopInput,
    status: AgentRunStatus,
    stop_reason: AgentRunStopReason | str,
    new_messages: list[Message],
    final_message: AssistantMessage | None,
    run_state: RunState,
    plan_state: PlanState | None,
    observations: list[ToolObservation],
    events: list[AgentEvent],
    usage: Any = None,
    error: Any = None,
    interruptions: list[Any] | None = None,
) -> AgentLoopOutcome:
    effects = workspace_effects(observations)
    if run_state.affected_paths:
        effects = WorkspaceEffects(
            affected_paths=tuple(sorted(run_state.affected_paths | set(effects.affected_paths))),
            changed=run_state.workspace_changed or effects.changed,
        )
    return AgentLoopOutcome(
        run_id=input.run_id,
        status=status,
        stop_reason=str(stop_reason),
        new_messages=list(new_messages),
        final_message=final_message,
        interruptions=list(interruptions or []),
        counters=AgentRunCounters(
            model_attempts=run_state.counters.model_attempts,
            tool_iterations=run_state.counters.tool_iterations,
            tool_calls=run_state.counters.tool_calls,
        ),
        usage=usage,
        verification=verification(observations) or list(run_state.verification),
        workspace_effects=effects,
        events=list(events),
        plan=_plan_summary_for_outcome(plan_state),
        signals=run_state.summary(),
        error=error,
    )


def _apply_plan_updates(
    plan_state: PlanState | None,
    tool_messages: list[ToolResultMessage],
    *,
    input: AgentLoopInput,
    recorder: "_EventRecorder",
    objective: str,
) -> PlanState | None:
    current = plan_state
    for message in tool_messages:
        try:
            next_state = apply_plan_update_metadata(
                current,
                message.metadata,
                mode=ensure_run_mode(input.mode),
                objective=objective,
                run_id=input.run_id,
            )
        except PlanValidationError as exc:
            recorder.emit(
                {
                    "type": "plan_state_warning",
                    "operation": "apply_update",
                    "message": str(exc),
                }
            )
            continue
        if next_state is current:
            continue
        event_type = _plan_event_type(current, next_state)
        current = next_state
        if current is not None:
            recorder.emit({"type": event_type, "plan": current.to_dict()})
    return current


def _tool_messages_from_observations(
    observations: list[ToolObservation],
) -> list[ToolResultMessage]:
    messages: list[ToolResultMessage] = []
    for observation in observations:
        if observation.status == "approval_required":
            approval_id = (
                observation.interruption.approval_id
                if observation.interruption is not None
                else None
            )
            messages.append(
                to_tool_result_message(
                    observation,
                    approval_id=approval_id,
                    approved=False,
                )
            )
            continue
        messages.append(to_tool_result_message(observation))
    return messages


def _with_runtime_context(
    input: AgentLoopInput,
    plan_state: PlanState | None,
    run_state: RunState,
) -> AgentLoopInput:
    values = {
        **dict(input.context),
        "run_signals": _run_signal_payload(run_state),
    }
    if plan_state is not None:
        values["plan_state"] = plan_state.to_dict()
    return replace(input, context=PreparedContext(values), plan_state=plan_state.to_dict() if plan_state else None)


def _with_plan_summary_context(
    input: AgentLoopInput,
    plan_state: PlanState | None,
) -> AgentLoopInput:
    values = {
        **dict(input.context),
        "checkpoint_phase": "plan_summary",
        "suppress_tools": True,
        "runtime_directive": (
            "The structured plan has been published. Summarize the canonical "
            "PlanState for the user without changing it, calling tools, or starting implementation."
        ),
    }
    if plan_state is not None:
        values["plan_state"] = plan_state.to_dict()
    return replace(
        input,
        context=PreparedContext(values),
        plan_state=plan_state.to_dict() if plan_state else None,
    )


def _run_signal_payload(run_state: RunState) -> dict[str, Any]:
    summary = run_state.summary()
    return {
        "workspace_changed": summary.workspace_changed,
        "affected_paths": list(summary.affected_paths),
        "verification_status": summary.verification_status,
        "last_error": summary.last_error,
        "approval_required": summary.approval_required,
        "tool_unavailable": summary.tool_unavailable,
        "cancelled": summary.cancelled,
        "counters": {
            "model_attempts": summary.counters.model_attempts,
            "tool_iterations": summary.counters.tool_iterations,
            "tool_calls": summary.counters.tool_calls,
        },
    }


def _next_retry_delay(policy: RetryPolicy, retries_so_far: int) -> int | None:
    if not policy.enabled or retries_so_far >= policy.max_retries:
        return None
    return int(policy.base_delay_ms * (2 ** retries_so_far))


def max_iterations_pause_message(max_tool_iterations: int) -> AssistantMessage:
    return AssistantMessage(
        content=[
            TextContent(
                text=(
                    "已暂停：连续工具调用达到本轮上限"
                    f"（max_tool_iterations={max_tool_iterations}）。"
                    "请回复“继续”或指定下一步。"
                )
            )
        ],
        stop_reason="max_iterations",
    )


def model_turns_exhausted_pause_message(max_model_turns: int) -> AssistantMessage:
    return AssistantMessage(
        content=[
            TextContent(
                text=(
                    "已暂停：模型轮次达到本轮上限"
                    f"（max_model_turns={max_model_turns}），"
                    "工具已执行，但还没有后续模型轮次总结结果。"
                    "请回复“继续”或指定下一步。"
                )
            )
        ],
        stop_reason="max_iterations",
    )


def last_assistant(messages: list[Message]) -> AssistantMessage | None:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            return message
    return None


def _objective_for_plan(input: AgentLoopInput, messages: list[Message]) -> str:
    if input.user_prompt:
        return input.user_prompt
    for message in reversed(messages):
        if isinstance(message, UserMessage):
            if isinstance(message.content, str):
                text = " ".join(message.content.strip().split())
                if text:
                    return text
            else:
                text = " ".join(
                    block.text.strip()
                    for block in message.content
                    if isinstance(block, TextContent) and block.text.strip()
                )
                if text:
                    return text
    return ""


def _guard_stop_reason(reason: str) -> AgentRunStopReason:
    if reason == "tool_unavailable":
        return "tool_unavailable"
    if reason == "cancelled":
        return "aborted"
    if reason == "approval_required":
        return "approval_required"
    return "run_guard"


def _plan_event_type(
    previous: PlanState | None,
    current: PlanState,
) -> str:
    old_status = previous.status if previous is not None else None
    if current.status == "proposed" and old_status != "proposed":
        return "plan_proposed"
    if current.status == "completed" and old_status != "completed":
        return "plan_completed"
    return "plan_updated"


def _plan_summary_for_outcome(plan_state: PlanState | None) -> Any:
    if plan_state is None:
        return None
    return plan_state.to_summary()


class _EventRecorder:
    def __init__(self, input: AgentLoopInput, ports: AgentLoopPorts) -> None:
        self._input = input
        self._ports = ports
        self.events: list[AgentEvent] = []
        self._turn_id = input.turn_start_seq
        self._event_seq = input.event_start_seq

    def emit(self, event: dict[str, Any]) -> None:
        event_type = ensure_runtime_event_type(event.get("type"))
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


def _emit_message(recorder: _EventRecorder, message: Message) -> None:
    recorder.emit({"type": "message_start", "message": message})
    recorder.emit({"type": "message_end", "message": message})


def _resume_tool_end_event(
    approval_id: str,
    observation: ToolObservation,
) -> dict[str, Any]:
    metadata = dict(observation.metadata)
    metadata.setdefault("approval_id", approval_id)
    return {
        "type": _tool_event_type(observation),
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


def _tool_event_type(observation: ToolObservation) -> str:
    if observation.status == "success":
        return "tool_completed"
    if observation.status in {"approval_required", "denied", "cancelled"}:
        return "tool_interrupted"
    return "tool_failed"


def _required_text(value: object, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{field_name} cannot be empty")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


__all__ = [
    "AgentEventEmitter",
    "last_assistant",
    "max_iterations_pause_message",
    "maybe_await",
    "resume_agent_loop",
    "run_agent_loop",
]
