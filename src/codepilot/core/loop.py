from __future__ import annotations

"""The core agent loop.

This file is the whole orchestration story:

1. ask the model for one assistant message;
2. run any tool calls in that message;
3. append tool observations;
4. repeat until there is a final answer, an approval pause, or a limit/error.
"""

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
    ErrorInfo,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    ensure_runtime_event_type,
)
from codepilot.tools.ports import ToolObservation, ToolResumeDecision

from .contracts import (
    AgentLoopInput,
    AgentLoopOutcome,
    AgentLoopPorts,
    AgentResumeInput,
    PreparedContext,
    RetryPolicy,
    TaskStrategy,
    WorkspaceEffects,
)
from .model_step import ModelTurnResult, run_model_turn, tool_catalog_for_request
from .state import RunState
from .task import (
    CompletionCheck,
    TaskController,
    TaskPlanningState,
    budget_for_profile,
    policy_for_mode,
)
from .task.state import TaskState
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


@dataclass
class _TaskRuntime:
    controller: TaskController
    task: TaskState
    run_state: RunState

    @classmethod
    def create(
        cls,
        *,
        strategy: TaskStrategy,
        messages: list[Message],
        run_state: RunState,
    ) -> "_TaskRuntime | None":
        if not strategy.enabled:
            return None
        policy = policy_for_mode(strategy.mode)
        planning = strategy.planning
        if planning is None and policy.planner_required:
            planning = TaskPlanningState(
                phase="execution",
                source="default",
                budget=budget_for_profile(strategy.planning_budget_profile),
            )
        controller = TaskController()
        task = controller.initialize(
            messages,
            goal=strategy.goal,
            proposed_steps=strategy.steps,
            mode=policy.mode,
            planning=planning,
            max_replans_per_run=strategy.max_replans_per_run,
            task_state=strategy.task_state,
        )
        return cls(controller=controller, task=task, run_state=run_state)

    def event_payload(self) -> dict[str, object]:
        return self.controller.event_payload(self.task)

    def control_signal(self) -> dict[str, object]:
        return self.controller.control_signal(self.task)

    def after_tool_results(self, results: list[ToolResultMessage]):
        return self.controller.after_tool_results(self.task, self.run_state, results)

    def check_completion(self) -> CompletionCheck:
        return self.controller.check_completion(self.task, self.run_state)

    def completion_steering(self, check: CompletionCheck) -> UserMessage:
        return self.controller.completion_steering(check)

    def summary(self):
        return self.controller.summarize(self.task)


async def run_agent_loop(
    input: AgentLoopInput,
    ports: AgentLoopPorts,
) -> AgentLoopOutcome:
    recorder = _EventRecorder(input, ports)
    run_state = RunState(
        run_id=input.run_id,
        session_id=input.correlation.session_id,
    )
    new_messages: list[Message] = []
    messages = list(input.messages)

    recorder.emit({"type": "agent_start"})
    recorder.emit({"type": "turn_start"})

    if input.user_prompt is not None:
        user_message = UserMessage(content=input.user_prompt)
        messages.append(user_message)
        _emit_message(recorder, user_message)

    task_runtime = _TaskRuntime.create(
        strategy=input.task_strategy,
        messages=messages,
        run_state=run_state,
    )
    if task_runtime is not None:
        recorder.emit({"type": "task_plan_created", "task": task_runtime.event_payload()})

    return await _drive_loop(
        input=input,
        ports=ports,
        recorder=recorder,
        messages=messages,
        new_messages=new_messages,
        run_state=run_state,
        task_runtime=task_runtime,
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
    recorder = _EventRecorder(loop_input, ports)
    run_state = RunState(input.run_id, input.correlation.session_id)
    messages = list(input.messages)
    new_messages: list[Message] = []
    observations: list[ToolObservation] = []

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
    observations.append(observation)
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

    task_runtime = _TaskRuntime.create(
        strategy=input.task_strategy,
        messages=messages,
        run_state=run_state,
    )
    if task_runtime is not None:
        decision = task_runtime.after_tool_results([tool_message])
        recorder.emit({"type": "task_step_updated", "task": task_runtime.event_payload()})
        recorder.emit(
            {
                "type": "task_decision",
                "decision": {
                    "action": decision.action,
                    "reason": decision.reason,
                    "next_action": decision.next_action,
                },
                "task": task_runtime.event_payload(),
            }
        )

    return await _drive_loop(
        input=loop_input,
        ports=ports,
        recorder=recorder,
        messages=messages,
        new_messages=new_messages,
        run_state=run_state,
        task_runtime=task_runtime,
        observations=observations,
        first_turn_started=True,
    )


async def _drive_loop(
    *,
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    recorder: "_EventRecorder",
    messages: list[Message],
    new_messages: list[Message],
    run_state: RunState,
    task_runtime: _TaskRuntime | None,
    observations: list[ToolObservation] | None = None,
    first_turn_started: bool,
) -> AgentLoopOutcome:
    all_observations = list(observations or [])
    usage = None
    max_model_turns = max(1, input.limits.max_model_turns)

    for turn_index in range(max_model_turns):
        if turn_index > 0 or not first_turn_started:
            recorder.emit({"type": "turn_start"})

        model_turn = await _model_turn_with_retries(
            input=input,
            ports=ports,
            recorder=recorder,
            messages=messages,
            run_state=run_state,
            task_runtime=task_runtime,
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
                observations=all_observations,
                events=recorder.events,
                task_runtime=task_runtime,
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
            outcome = _finish_or_steer(
                input=input,
                recorder=recorder,
                messages=messages,
                new_messages=new_messages,
                assistant=assistant,
                run_state=run_state,
                task_runtime=task_runtime,
                observations=all_observations,
                usage=usage,
                turn_index=turn_index,
                max_model_turns=max_model_turns,
            )
            if outcome is not None:
                return outcome
            continue

        if ports.tools is None:
            error = {"code": "core.missing_tool_port", "message": "Model requested tools but no ToolPort was provided"}
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
                observations=all_observations,
                events=recorder.events,
                task_runtime=task_runtime,
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
            observations=all_observations,
            task_runtime=task_runtime,
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
            task_signal=_task_signal(input, task_runtime),
            metadata={
                "model_provider": input.model.provider,
                "model_id": input.model.model_id,
            },
            tools=ports.tools,
            tool_calls=tool_calls,
            emit=recorder.emit,
        )
        all_observations.extend(turn_observations)
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

        task_decision = None
        if task_runtime is not None:
            task_decision = task_runtime.after_tool_results(tool_messages)
            recorder.emit({"type": "task_step_updated", "task": task_runtime.event_payload()})
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

        approvals = approval_observations(turn_observations)
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
                observations=all_observations,
                events=recorder.events,
                task_runtime=task_runtime,
                usage=usage,
                interruptions=[item.interruption for item in approvals if item.interruption],
            )

        if any(message.status == "cancelled" for message in tool_messages):
            recorder.emit({"type": "turn_end", "message": assistant, "toolResults": visible_tool_messages})
            recorder.emit({"type": "agent_end", "status": "aborted"})
            return _outcome(
                input=input,
                status="aborted",
                stop_reason="aborted",
                new_messages=new_messages,
                final_message=assistant,
                run_state=run_state,
                observations=all_observations,
                events=recorder.events,
                task_runtime=task_runtime,
                usage=usage,
            )

        if task_decision is not None and task_decision.action == "stop":
            recorder.emit({"type": "turn_end", "message": assistant, "toolResults": visible_tool_messages})
            recorder.emit({"type": "agent_end", "status": "waiting_user"})
            return _outcome(
                input=input,
                status="waiting_user",
                stop_reason="task_blocked",
                new_messages=new_messages,
                final_message=assistant,
                run_state=run_state,
                observations=all_observations,
                events=recorder.events,
                task_runtime=task_runtime,
                usage=usage,
            )

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
        stop_reason="max_model_turns",
        new_messages=new_messages,
        final_message=last_assistant(new_messages),
        run_state=run_state,
        observations=all_observations,
        events=recorder.events,
        task_runtime=task_runtime,
        usage=usage,
    )


async def _model_turn_with_retries(
    *,
    input: AgentLoopInput,
    ports: AgentLoopPorts,
    recorder: "_EventRecorder",
    messages: list[Message],
    run_state: RunState,
    task_runtime: _TaskRuntime | None,
) -> ModelTurnResult:
    retries = 0
    while True:
        result = await run_model_turn(
            _with_task_context(input, task_runtime),
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
    task_runtime: _TaskRuntime | None,
    observations: list[ToolObservation],
    usage: Any,
    turn_index: int,
    max_model_turns: int,
) -> AgentLoopOutcome | None:
    if task_runtime is None:
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
        recorder.emit({"type": "agent_end", "status": "completed"})
        return _outcome(
            input=input,
            status="completed",
            stop_reason="final_answer",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            observations=observations,
            events=recorder.events,
            task_runtime=None,
            usage=usage,
        )

    check = task_runtime.check_completion()
    recorder.emit(
        {
            "type": "completion_checked",
            "task": task_runtime.event_payload(),
            "completion_satisfied": check.satisfied,
            "completion_reason": check.reason,
        }
    )
    if check.satisfied:
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
        recorder.emit({"type": "agent_end", "status": "completed"})
        return _outcome(
            input=input,
            status="completed",
            stop_reason="final_answer",
            new_messages=new_messages,
            final_message=assistant,
            run_state=run_state,
            observations=observations,
            events=recorder.events,
            task_runtime=task_runtime,
            usage=usage,
        )

    if check.can_continue and turn_index + 1 < max_model_turns:
        steering = task_runtime.completion_steering(check)
        messages.append(steering)
        new_messages.append(steering)
        _emit_message(recorder, steering)
        recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
        return None

    recorder.emit({"type": "turn_end", "message": assistant, "toolResults": []})
    recorder.emit({"type": "agent_end", "status": "waiting_user"})
    stop_reason = "task_blocked" if check.reason == "blocked_steps" else "task_incomplete"
    return _outcome(
        input=input,
        status="waiting_user",
        stop_reason=stop_reason,
        new_messages=new_messages,
        final_message=assistant,
        run_state=run_state,
        observations=observations,
        events=recorder.events,
        task_runtime=task_runtime,
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
    observations: list[ToolObservation],
    task_runtime: _TaskRuntime | None,
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
            observations=observations,
            events=recorder.events,
            task_runtime=task_runtime,
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
            observations=observations,
            events=recorder.events,
            task_runtime=task_runtime,
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
            observations=observations,
            events=recorder.events,
            task_runtime=task_runtime,
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
            observations=observations,
            events=recorder.events,
            task_runtime=task_runtime,
            usage=usage,
            error=error,
        )

    return None


def _outcome(
    *,
    input: AgentLoopInput,
    status: AgentRunStatus,
    stop_reason: AgentRunStopReason | str,
    new_messages: list[Message],
    final_message: AssistantMessage | None,
    run_state: RunState,
    observations: list[ToolObservation],
    events: list[AgentEvent],
    task_runtime: _TaskRuntime | None,
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
        status=status,  # type: ignore[arg-type]
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
        task=task_runtime.summary() if task_runtime is not None else None,
        error=error,
    )


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


def _with_task_context(
    input: AgentLoopInput,
    task_runtime: _TaskRuntime | None,
) -> AgentLoopInput:
    if task_runtime is None:
        return input
    return replace(
        input,
        context=PreparedContext(
            {
                **dict(input.context),
                "task_state": task_runtime.event_payload(),
                "task_signal": task_runtime.control_signal(),
            }
        ),
    )


def _task_signal(
    input: AgentLoopInput,
    task_runtime: _TaskRuntime | None,
) -> dict[str, Any]:
    if task_runtime is not None:
        return task_runtime.control_signal()
    signal = input.context.get("task_signal")
    return dict(signal) if isinstance(signal, dict) else {}


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


def last_assistant(messages: list[Message]) -> AssistantMessage | None:
    for message in reversed(messages):
        if isinstance(message, AssistantMessage):
            return message
    return None


class _EventRecorder:
    def __init__(self, input: AgentLoopInput, ports: AgentLoopPorts) -> None:
        self._input = input
        self._ports = ports
        self.events: list[AgentEvent] = []
        self._turn_id = 0
        self._event_seq = 0

    def emit(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
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
