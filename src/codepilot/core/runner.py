from __future__ import annotations

"""Agent runner: model/tool loop with soft plan state and final-answer guard."""

import asyncio
import time
from dataclasses import dataclass, replace
from typing import Any

from codepilot.protocols import (
    AgentEvent,
    AgentEventSink,
    AgentRunCounters,
    AgentRunStatus,
    AgentRunStopReason,
    AssistantMessage,
    CLOSE_PLAN_TOOL,
    Message,
    PROPOSE_PLAN_TOOL,
    TextContent,
    ToolCall,
    ToolResultMessage,
    ensure_runtime_event_type,
)
from codepilot.tools.results import ToolResult, to_tool_result_message
from codepilot.tools.security import ApprovalResponse

from .contracts import (
    AgentLoopInput,
    AgentLoopOutcome,
    AgentLoopPorts,
    AgentResumeInput,
    CoreRunBoundary,
    CoreWaitingRequest,
    PreparedContext,
    RetryPolicy,
    WorkspaceEffects,
)
from .model_step import ModelTurnResult, run_model_turn
from .plan import (
    PlanState,
    PlanValidationError,
    load_plan_state,
)
from .run_guard import RunGuard
from .state import RunState
from .tool_step import (
    approval_results,
    execute_tool_turn,
    tool_end_event,
    verification,
    workspace_effects,
)


def now_ms() -> int:
    return int(time.time() * 1000)


async def maybe_await(value: Any) -> Any:
    if asyncio.isfuture(value) or asyncio.iscoroutine(value):
        return await value
    return value


@dataclass(frozen=True)
class _SyntheticControlFrame:
    id: str
    kind: str
    scope: str
    instruction: str
    reason: str = ""
    source: str = "runner"
    expires_after_turns: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "scope": self.scope,
            "instruction": self.instruction,
            "reason": self.reason,
            "source": self.source,
            "expires_after_turns": self.expires_after_turns,
        }


class _BoundaryCommitter:
    def __init__(self, ports: AgentLoopPorts, new_messages: list[Message]) -> None:
        self._port = ports.state
        self._tools = ports.tools
        self._messages = new_messages
        self._committed_count = 0

    def tool_checkpoint_state(
        self,
        intent: dict[str, object] | None = None,
    ) -> dict[str, object] | None:
        if self._tools is None:
            return dict(intent) if intent else None
        checkpoint = getattr(self._tools, "checkpoint_state", None)
        if not callable(checkpoint):
            return dict(intent) if intent else None
        return checkpoint(intent=intent)

    async def commit(
        self,
        kind: str,
        run_state: RunState,
        *,
        waiting: CoreWaitingRequest | None = None,
        tool_recovery_state: dict[str, object] | None = None,
    ) -> None:
        await self.commit_state(
            kind,
            run_state.to_dict(),
            waiting=waiting,
            tool_recovery_state=tool_recovery_state,
        )

    async def commit_state(
        self,
        kind: str,
        core_state: dict[str, object],
        *,
        waiting: CoreWaitingRequest | None = None,
        tool_recovery_state: dict[str, object] | None = None,
    ) -> None:
        if self._port is None:
            self._committed_count = len(self._messages)
            return
        pending = tuple(self._messages[self._committed_count :])
        boundary = CoreRunBoundary(
            kind=kind,  # type: ignore[arg-type]
            core_state=core_state,
            new_messages=pending,
            waiting=waiting,
            tool_recovery_state=tool_recovery_state,
        )
        await maybe_await(self._port.commit(boundary))
        self._committed_count = len(self._messages)


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
    run_state = RunState.from_mapping(
        input.run_state,
        run_id=input.run_id,
        session_id=input.correlation.session_id,
    )
    plan_state = load_plan_state(input.plan_state)
    messages = list(input.messages)
    new_messages: list[Message] = []
    committer = _BoundaryCommitter(ports, new_messages)

    recorder.emit({"type": "agent_start"})
    recorder.emit({"type": "turn_start"})

    outcome = await _drive_loop(
        input=input,
        ports=ports,
        recorder=recorder,
        messages=messages,
        new_messages=new_messages,
        committer=committer,
        run_state=run_state,
        plan_state=plan_state,
        observations=[],
        first_turn_started=True,
    )
    await _commit_outcome_boundary(committer, outcome)
    return outcome


async def resume_agent_loop(
    input: AgentResumeInput,
    ports: AgentLoopPorts,
) -> AgentLoopOutcome:
    run_state = RunState.from_mapping(
        input.run_state,
        run_id=input.run_id,
        session_id=input.correlation.session_id,
    )
    new_messages: list[Message] = []
    committer = _BoundaryCommitter(ports, new_messages)
    if ports.tools is None:
        outcome = AgentLoopOutcome(
            run_id=input.run_id,
            status="failed",
            stop_reason="missing_tool_port",
            signals=run_state.summary(),
            run_state=run_state.to_dict(),
            error={"code": "core.missing_tool_port"},
        )
        await _commit_outcome_boundary(committer, outcome)
        return outcome
    if not input.approval_id or not input.decision:
        outcome = AgentLoopOutcome(
            run_id=input.run_id,
            status="failed",
            stop_reason="missing_approval_decision",
            signals=run_state.summary(),
            run_state=run_state.to_dict(),
            error={"code": "core.missing_approval_decision"},
        )
        await _commit_outcome_boundary(committer, outcome)
        return outcome

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
        turn_start_seq=input.turn_start_seq,
        run_state=input.run_state,
    )
    recorder = _EventRecorder(loop_input, ports)
    plan_state = load_plan_state(input.plan_state)
    messages = list(input.messages)

    recorder.emit({"type": "agent_start"})
    recorder.emit({"type": "turn_start"})
    challenge = ports.tools.approval_challenge(input.approval_id)
    tool_call_id = challenge.tool_call_id if challenge is not None else input.approval_id
    tool_name = challenge.tool_name if challenge is not None else "approval_resume"
    recorder.emit(
        {
            "type": "tool_started",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "args": {
                "approval_id": input.approval_id,
                "decision": input.decision,
            },
            "source": "approval_resume",
        }
    )

    await committer.commit(
        "before_tools",
        run_state,
        tool_recovery_state=committer.tool_checkpoint_state(
            {
                "approval_id": input.approval_id,
                "decision": input.decision,
                "source": "approval_resume",
            }
        ),
    )
    observation = await _resume_tool_observation(input, ports)
    recorder.emit(tool_end_event(observation))
    tool_message = to_tool_result_message(observation)
    messages.append(tool_message)
    new_messages.append(tool_message)
    _emit_message(recorder, tool_message)
    run_state.collect_tool_results([tool_message])
    run_state.observe_plan_execution(
        [tool_message],
        in_progress_item_id=_in_progress_plan_item_id(plan_state),
    )
    run_state.counters.tool_iterations += 1
    plan_state = _apply_plan_updates(
        plan_state,
        [observation],
        input=loop_input,
        recorder=recorder,
        qualified_failure_count=run_state.qualified_plan_failure_count,
    )
    await committer.commit("after_tools", run_state)

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
        await _commit_outcome_boundary(committer, interruption)
        return interruption

    outcome = await _drive_loop(
        input=loop_input,
        ports=ports,
        recorder=recorder,
        messages=messages,
        new_messages=new_messages,
        committer=committer,
        run_state=run_state,
        plan_state=plan_state,
        observations=[observation],
        first_turn_started=True,
    )
    await _commit_outcome_boundary(committer, outcome)
    return outcome


async def _resume_tool_observation(
    input: AgentResumeInput,
    ports: AgentLoopPorts,
) -> ToolResult:
    approval_id = input.approval_id or ""
    challenge = ports.tools.approval_challenge(approval_id)
    fingerprint = challenge.request_fingerprint if challenge is not None else "unknown"
    return await ports.tools.resume(
        ApprovalResponse(
            approval_id=approval_id,
            request_fingerprint=fingerprint,
            decision=input.decision,  # type: ignore[arg-type]
            scope="once",
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
    committer: _BoundaryCommitter,
    run_state: RunState,
    plan_state: PlanState | None,
    observations: list[ToolResult],
    first_turn_started: bool,
) -> AgentLoopOutcome:
    usage = None
    max_model_turns = max(1, input.limits.max_model_turns)
    synthetic_control: _SyntheticControlFrame | None = None
    plan_publish_attempted = False
    plan_closeout_attempted = False

    for turn_index in range(max_model_turns):
        if turn_index > 0 or not first_turn_started:
            recorder.emit({"type": "turn_start"})

        turn_input = input
        if synthetic_control is not None:
            turn_input = _with_synthetic_control(
                input,
                synthetic_control,
            )
            synthetic_control = None
        model_turn = await _model_turn_with_retries(
            input=turn_input,
            ports=ports,
            recorder=recorder,
            messages=messages,
            committer=committer,
            run_state=run_state,
            plan_state=plan_state,
        )
        if model_turn.error is not None:
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
        await committer.commit("after_model", run_state)

        tool_calls = [block for block in assistant.content if isinstance(block, ToolCall)]
        if not tool_calls:
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
                plan_publish_attempted=plan_publish_attempted,
                plan_closeout_attempted=plan_closeout_attempted,
            )
            if outcome is not None:
                if isinstance(outcome, _SyntheticControlFrame):
                    synthetic_control = outcome
                    if outcome.kind == "plan_publish_required":
                        plan_publish_attempted = True
                    elif outcome.kind == "plan_closeout":
                        plan_closeout_attempted = True
                    continue
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

        await committer.commit(
            "before_tools",
            run_state,
            tool_recovery_state=committer.tool_checkpoint_state(
                {
                    "tool_calls": [
                        {
                            "id": call.id,
                            "name": call.name,
                            "arguments": dict(call.arguments),
                        }
                        for call in tool_calls
                    ]
                }
            ),
        )
        turn_observations = await execute_tool_turn(
            run_id=input.run_id,
            session_id=input.correlation.session_id or "session_unknown",
            current_mode=input.mode,
            tools=ports.tools,
            tool_calls=tool_calls,
            catalog_snapshot=model_turn.catalog_snapshot,
            emit=recorder.emit,
        )
        observations.extend(turn_observations)
        run_state.counters.tool_iterations += 1

        tool_messages = _tool_messages_from_observations(turn_observations)
        run_state.collect_tool_results(tool_messages)
        run_state.observe_plan_execution(
            tool_messages,
            in_progress_item_id=_in_progress_plan_item_id(plan_state),
        )
        visible_tool_messages = list(tool_messages)
        for message in visible_tool_messages:
            messages.append(message)
            new_messages.append(message)
            _emit_message(recorder, message)

        plan_state = _apply_plan_updates(
            plan_state,
            turn_observations,
            input=input,
            recorder=recorder,
            qualified_failure_count=run_state.qualified_plan_failure_count,
        )
        run_state.observe_plan_execution(
            [],
            in_progress_item_id=_in_progress_plan_item_id(plan_state),
        )
        await committer.commit("after_tools", run_state)

        plan_ready_for_approval = _is_pending_plan(plan_state) and any(
            result.tool_name == PROPOSE_PLAN_TOOL
            and result.status == "success"
            and result.data.get("plan_operation") == "propose_plan"
            for result in turn_observations
        )

        recovery_instruction = run_state.truncated_read_recovery_instruction(visible_tool_messages)
        if recovery_instruction:
            synthetic_control = _synthetic_control(
                kind="truncated_read_recovery",
                scope="tool_recovery_only",
                instruction=recovery_instruction,
                reason="repeated_truncated_read",
            )

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

        if plan_ready_for_approval and plan_state is not None:
            approval_message = _plan_approval_message(plan_state)
            messages.append(approval_message)
            new_messages.append(approval_message)
            _emit_message(recorder, approval_message)
            return _plan_approval_pause_outcome(
                input=input,
                recorder=recorder,
                assistant=approval_message,
                new_messages=new_messages,
                run_state=run_state,
                plan_state=plan_state,
                observations=observations,
                usage=usage,
                tool_results=visible_tool_messages,
            )

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
    committer: _BoundaryCommitter,
    run_state: RunState,
    plan_state: PlanState | None,
) -> ModelTurnResult:
    retries = 0
    while True:
        await committer.commit("before_model", run_state)
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
    observations: list[ToolResult],
    usage: Any,
    turn_index: int,
    max_model_turns: int,
    plan_publish_attempted: bool,
    plan_closeout_attempted: bool,
) -> AgentLoopOutcome | _SyntheticControlFrame | None:
    if input.mode == "plan":
        if not plan_publish_attempted and turn_index + 1 < max_model_turns:
            recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
            return _synthetic_control(
                kind="plan_publish_required",
                scope="plan_protocol_only",
                instruction=(
                    "Runtime protocol state: the current mode is plan and no canonical Task Plan "
                    "has been published. No user approval has occurred, and publishing a plan is "
                    "not approval. Preserve the user's original request and every confirmed "
                    "constraint. If repository evidence is sufficient and the implementation plan "
                    "is ready, publish it now with propose_plan instead of presenting or seeking "
                    "approval for a prose draft. If a material user decision is still missing, ask "
                    "one concrete clarification question. Do not state or imply that the user "
                    "accepted the plan, do not claim that implementation has started, do not modify "
                    "the workspace, and do not change modes."
                ),
                reason="plan_mode_finished_without_published_plan",
            )
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
        plan_guard = _plan_closeout_guard(
            input=input,
            plan_state=plan_state,
            run_state=run_state,
            new_messages=new_messages,
        )
        if plan_guard is not None:
            reason, instruction = plan_guard
            if plan_closeout_attempted:
                recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
                recorder.emit({"type": "agent_end", "status": "waiting_user"})
                return _outcome(
                    input=input,
                    status="waiting_user",
                    stop_reason="plan_incomplete",
                    new_messages=new_messages,
                    final_message=assistant,
                    run_state=run_state,
                    plan_state=plan_state,
                    observations=observations,
                    events=recorder.events,
                    usage=usage,
                )
            recorder.emit(
                {
                    "type": "run_guard_checked",
                    "decision": {
                        "action": "continue_with_instruction",
                        "reason": reason,
                        "instruction": instruction,
                    },
                    "signals": run_state.summary(),
                }
            )
            if turn_index + 1 < max_model_turns:
                recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
                return _synthetic_control(
                    kind="plan_closeout",
                    scope="plan_closeout_only",
                    instruction=instruction,
                    reason=reason,
                )
            recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
            recorder.emit({"type": "agent_end", "status": "waiting_user"})
            return _outcome(
                input=input,
                status="waiting_user",
                stop_reason="run_guard",
                new_messages=new_messages,
                final_message=assistant,
                run_state=run_state,
                plan_state=plan_state,
                observations=observations,
                events=recorder.events,
                usage=usage,
            )
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
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
        return _synthetic_control_for_guard(decision.reason, decision.instruction)
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


def _plan_closeout_guard(
    *,
    input: AgentLoopInput,
    plan_state: PlanState | None,
    run_state: RunState,
    new_messages: list[Message],
) -> tuple[str, str] | None:
    if input.mode == "plan":
        return None
    if plan_state is None or plan_state.status != "active":
        return None
    if (
        plan_state.origin_mode == "plan"
        and run_state.counters.tool_calls == 0
    ):
        step = _first_unfinished_plan_step(plan_state)
        return (
            "plan_execution_not_started",
            (
                "当前有已批准的 active Task Plan，但本轮还没有执行任何计划步骤。"
                "不要继续读取、修改或运行命令；只用最终答复说明尚未开始执行、当前计划状态和下一步需要用户确认。"
                + (f" 第一个未完成步骤：{step}。" if step else "")
            ),
        )
    return (
        "plan_closeout_missing",
        (
            "当前 active Task Plan 需要收尾确认。请依据 completion criteria、步骤验证和实际工具结果"
            f"只调用 {CLOSE_PLAN_TOOL} 做状态收尾，不要读取、修改、运行命令或扩大任务。"
            "完成时将 status 设为 completed；明显尚未完成时将 status 设为 active，保留剩余步骤并说明阻塞。"
        ),
    )


def _first_unfinished_plan_step(plan_state: PlanState) -> str:
    for item in plan_state.items:
        if item.status != "completed":
            return item.step
    return ""


def _in_progress_plan_item_id(plan_state: PlanState | None) -> str | None:
    if plan_state is None or plan_state.status != "active":
        return None
    for item in plan_state.items:
        if item.status == "in_progress":
            return item.id
    return None


def _tool_limit_outcome(
    *,
    input: AgentLoopInput,
    recorder: "_EventRecorder",
    assistant: AssistantMessage,
    tool_calls: list[ToolCall],
    new_messages: list[Message],
    run_state: RunState,
    plan_state: PlanState | None,
    observations: list[ToolResult],
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
    observations: list[ToolResult],
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
    observations: list[ToolResult],
    usage: Any,
) -> AgentLoopOutcome | None:
    approvals = approval_results(observations)
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
            interruptions=[item.approval for item in approvals if item.approval is not None],
        )
    if any(result.status == "user_input_required" for result in observations):
        recorder.emit(
            {"type": "turn_end", "message": assistant, "toolResults": visible_tool_messages}
        )
        recorder.emit({"type": "agent_end", "status": "waiting_user"})
        return _outcome(
            input=input,
            status="waiting_user",
            stop_reason="user_input_required",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            plan_state=plan_state,
            observations=observations,
            events=recorder.events,
            usage=usage,
        )
    if any(result.status == "cancelled" for result in observations):
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
    observations: list[ToolResult],
    usage: Any,
    tool_results: list[ToolResultMessage] | None = None,
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
            "toolResults": list(tool_results or []),
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
    return plan_state is not None and plan_state.status == "proposed"


def _mark_plan_summary_message(
    message: AssistantMessage,
    plan_state: PlanState | None,
) -> None:
    message.metadata["message_kind"] = "plan_summary"
    if plan_state is not None:
        message.metadata["plan_id"] = plan_state.plan_id
        message.metadata["plan_revision"] = plan_state.revision


def _plan_approval_message(plan_state: PlanState) -> AssistantMessage:
    lines = [
        "已根据仓库探索生成待审批的代码修改方案。",
        f"用户原始请求：{plan_state.raw_user_request}",
        f"Build 执行目标：{plan_state.interpreted_goal}",
        f"任务理解：{plan_state.task_understanding}",
        f"当前实现与证据：{plan_state.current_implementation}",
        f"目标设计：{plan_state.target_design}",
        f"影响范围：{plan_state.impact_scope}",
        "风险与待确认项：",
        *[f"- {item}" for item in plan_state.risks_and_open_questions],
        f"验证方案：{plan_state.verification_plan}",
        f"摘要：{plan_state.summary}",
        "执行步骤（批准后由 build 模式执行，当前均未开始）：",
        *[
            (
                f"{index}. {item.step}\n"
                f"   修改：{item.details}\n"
                f"   验证：{item.verification}"
            )
            for index, item in enumerate(plan_state.items, start=1)
        ],
        "完成标准：",
        *[f"- {criterion}" for criterion in plan_state.completion_criteria],
        "请回复“批准”开始执行，回复“拒绝”放弃该方案；也可以直接说明需要调整的内容。"
        "命令方式同样可用：/plan approve 或 /plan reject。",
    ]
    message = AssistantMessage(
        content=[TextContent(text="\n".join(line for line in lines if line))],
        stop_reason="stop",
    )
    message.metadata["generated_by"] = "runner"
    _mark_plan_summary_message(message, plan_state)
    return message


async def _commit_outcome_boundary(
    committer: _BoundaryCommitter,
    outcome: AgentLoopOutcome,
) -> None:
    if outcome.status == "waiting_approval":
        challenge = outcome.interruptions[0] if outcome.interruptions else None
        request_id = challenge.approval_id if challenge is not None else outcome.run_id
        payload: dict[str, object] = {}
        if challenge is not None:
            payload = {
                "tool_call_id": challenge.tool_call_id,
                "tool_name": challenge.tool_name,
            }
        await committer.commit_state(
            "waiting_tool_approval",
            outcome.run_state,
            waiting=CoreWaitingRequest(
                kind="tool_approval",
                request_id=request_id,
                payload=payload,
            ),
            tool_recovery_state=committer.tool_checkpoint_state(),
        )
        return
    if outcome.status == "waiting_user":
        if outcome.stop_reason == "plan_approval_required":
            request_id = outcome.plan.plan_id if outcome.plan is not None else outcome.run_id
            await committer.commit_state(
                "waiting_plan_confirmation",
                outcome.run_state,
                waiting=CoreWaitingRequest(
                    kind="plan_confirmation",
                    request_id=request_id,
                    payload={"stop_reason": outcome.stop_reason},
                ),
                tool_recovery_state=committer.tool_checkpoint_state(),
            )
            return
        await committer.commit_state(
            "waiting_user_input",
            outcome.run_state,
            waiting=CoreWaitingRequest(
                kind="user_input",
                request_id=f"{outcome.run_id}:{outcome.stop_reason}",
                payload={"stop_reason": outcome.stop_reason},
            ),
            tool_recovery_state=committer.tool_checkpoint_state(),
        )
        return
    await committer.commit_state("before_finalization", outcome.run_state)


def _outcome(
    *,
    input: AgentLoopInput,
    status: AgentRunStatus,
    stop_reason: AgentRunStopReason | str,
    new_messages: list[Message],
    final_message: AssistantMessage | None,
    run_state: RunState,
    plan_state: PlanState | None,
    observations: list[ToolResult],
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
        run_state=run_state.to_dict(),
        error=error,
    )


def _apply_plan_updates(
    plan_state: PlanState | None,
    tool_results: list[ToolResult],
    *,
    input: AgentLoopInput,
    recorder: "_EventRecorder",
    qualified_failure_count: int,
) -> PlanState | None:
    current = plan_state
    for result in tool_results:
        raw_state = result.data.get("plan_state")
        if not isinstance(raw_state, dict) and not hasattr(raw_state, "items"):
            continue
        try:
            next_state = load_plan_state(dict(raw_state))
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
    observations: list[ToolResult],
) -> list[ToolResultMessage]:
    return [
        to_tool_result_message(result)
        for result in observations
        if result.status not in {"approval_required", "user_input_required"}
    ]


def _synthetic_control(
    *,
    kind: str,
    scope: str,
    instruction: str,
    reason: str = "",
) -> _SyntheticControlFrame:
    return _SyntheticControlFrame(
        id=f"synthetic_{now_ms()}",
        kind=kind,
        scope=scope,
        instruction=instruction,
        reason=reason or kind,
    )


def _synthetic_control_for_guard(reason: str, instruction: str) -> _SyntheticControlFrame:
    if reason == "empty_final_answer":
        return _synthetic_control(
            kind="empty_final_answer",
            scope="final_answer_only",
            instruction=instruction,
            reason=reason,
        )
    if reason == "verification_failed":
        return _synthetic_control(
            kind="verification_failed_summary",
            scope="summary_only",
            instruction=instruction,
            reason=reason,
        )
    return _synthetic_control(
        kind=reason or "runner_control",
        scope="summary_only",
        instruction=instruction,
        reason=reason,
    )


def _with_synthetic_control(
    input: AgentLoopInput,
    control: _SyntheticControlFrame,
) -> AgentLoopInput:
    values = {
        **dict(input.context),
        "synthetic_control": control.to_dict(),
    }
    return replace(input, context=PreparedContext(values))


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
