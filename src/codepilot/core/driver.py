from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field

from codepilot.protocols import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    Usage,
)
from codepilot.tools.registry import ToolCatalogSnapshot
from codepilot.tools.results import ToolResult

from .contracts import (
    CallModel,
    CoreBoundary,
    CoreBoundaryKind,
    CoreOutcome,
    CorePorts,
    CoreReason,
    CoreRunInput,
    CoreWait,
    ExecuteTools,
    ModelEntry,
    Terminate,
    ToolResultEntry,
    Wait,
)
from .events import CoreDomainEvent
from .errors import CoreInvariantError
from .model_step import call_model_once
from .observations import (
    CancellationObservation,
    CoreObservation,
    ModelObservation,
    ToolBatchObservation,
    UserInputObservation,
)
from .policy import CorePolicy, PolicyContext
from .reducer import ReductionContext, apply_decision, reduce_observation
from .reducer import CoreReduction
from .state import CoreState
from .tool_step import (
    execute_core_tool_batch,
    interrupted_tool_results,
    prepare_core_tool_batch,
    project_core_command_results,
    project_core_commands,
    project_final_tool_messages,
)


@dataclass
class _MessageJournal:
    initial: tuple[Message, ...]
    _messages: list[Message] = field(init=False)
    _added: list[Message] = field(default_factory=list, init=False)
    _pending: list[Message] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._messages = list(self.initial)

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(self._messages)

    @property
    def added(self) -> tuple[Message, ...]:
        return tuple(self._added)

    @property
    def pending(self) -> tuple[Message, ...]:
        return tuple(self._pending)

    def extend(self, messages: tuple[Message, ...]) -> None:
        self._messages.extend(messages)
        self._added.extend(messages)
        self._pending.extend(messages)

    def mark_committed(self) -> None:
        self._pending.clear()


@dataclass
class _EventJournal:
    _pending: list[CoreDomainEvent] = field(default_factory=list)

    @property
    def pending(self) -> tuple[CoreDomainEvent, ...]:
        return tuple(self._pending)

    def extend(self, events: tuple[CoreDomainEvent, ...]) -> None:
        self._pending.extend(events)

    def mark_committed(self) -> None:
        self._pending.clear()


@dataclass
class _DriverState:
    core: CoreState
    observation: CoreObservation
    messages: _MessageJournal
    events: _EventJournal = field(default_factory=_EventJournal)
    usage: Usage | None = None
    catalog_snapshot: ToolCatalogSnapshot | None = None
    sequence: int = 0

    def observation_id(self, kind: str) -> str:
        self.sequence += 1
        return f"core:{kind}:{self.sequence}"


async def run_core(input: CoreRunInput, ports: CorePorts) -> CoreOutcome:
    """Drive Core through deterministic reduce-decide-execute transitions."""

    reduction_context = ReductionContext(run_id=input.run_id, mode=input.mode, now_ms=0)
    policy_context = PolicyContext(mode=input.mode, limits=input.limits)
    driver = _DriverState(
        core=input.state,
        observation=_initial_observation(input),
        messages=_MessageJournal(input.messages),
    )

    cancellation = _cancellation_observation(input, ports, driver)
    if cancellation is not None:
        _reduce(driver, cancellation, reduction_context)
    elif isinstance(input.entry, ToolResultEntry):
        calls = _unsettled_tool_calls(driver.messages.messages)
        observation = ToolBatchObservation(
            observation_id=driver.observation_id("entry_tools"),
            calls=calls,
            results=_complete_results(calls, input.entry.results),
            commands=project_core_commands(input.entry.results),
        )
        reduction = _reduce(driver, observation, reduction_context)
        visible_results = project_core_command_results(
            observation.results,
            reduction.command_results,
        )
        driver.messages.extend(project_final_tool_messages(visible_results))
        await _commit_boundary(input, ports, driver, "after_tools")
    elif input.entry.message is None:
        _reduce(driver, driver.observation, reduction_context)

    while True:
        cancellation = _cancellation_observation(input, ports, driver)
        if cancellation is not None:
            _reduce(driver, cancellation, reduction_context)

        decision = CorePolicy.decide(driver.core, driver.observation, policy_context)
        unsettled = _unsettled_tool_calls(driver.messages.messages)
        must_settle = isinstance(decision, Terminate) or (
            isinstance(decision, Wait)
            and decision.wait.kind not in {"tool_approval", "user_input"}
        )
        if unsettled and must_settle:
            await _settle_unexecuted_calls(
                input,
                ports,
                driver,
                unsettled,
                reduction_context,
                reason=(
                    decision.wait.reason.code
                    if isinstance(decision, Wait)
                    else decision.reason_code
                ),
            )

        decision_reduction = apply_decision(driver.core, decision, reduction_context)
        driver.core = decision_reduction.state
        driver.events.extend(decision_reduction.events)

        if isinstance(decision, CallModel):
            await _commit_boundary(input, ports, driver, "before_model")
            action = await call_model_once(
                input,
                ports,
                driver.messages.messages,
                decision.directive,
                observation_id=driver.observation_id("model"),
            )
            if action.observation.message is not None:
                driver.messages.extend((action.observation.message,))
            driver.usage = action.usage or driver.usage
            driver.catalog_snapshot = action.catalog_snapshot
            _reduce(driver, action.observation, reduction_context)
            await _commit_boundary(input, ports, driver, "after_model")
            continue

        if isinstance(decision, ExecuteTools):
            prepared = prepare_core_tool_batch(
                input,
                ports,
                decision,
                driver.catalog_snapshot,
            )
            if prepared.preparation.results:
                executed = False
                results = _complete_results(
                    prepared.calls,
                    prepared.preparation.results,
                )
            else:
                executed = True
                await _commit_boundary(input, ports, driver, "before_tools")
                results = _complete_results(
                    prepared.calls,
                    await execute_core_tool_batch(ports, prepared),
                )
            observation = ToolBatchObservation(
                observation_id=driver.observation_id("tools"),
                calls=prepared.calls,
                results=results,
                commands=project_core_commands(results),
            )
            reduction = _reduce(driver, observation, reduction_context)
            visible_results = project_core_command_results(
                results,
                reduction.command_results,
            )
            driver.messages.extend(project_final_tool_messages(visible_results))
            if executed or not _has_external_wait(results):
                await _commit_boundary(input, ports, driver, "after_tools")
            continue

        if isinstance(decision, Wait):
            await _commit_boundary(
                input,
                ports,
                driver,
                "waiting",
                wait=decision.wait,
            )
            return _outcome(driver, "waiting", decision.wait.reason, wait=decision.wait)

        if isinstance(decision, Terminate):
            await _commit_boundary(input, ports, driver, "before_terminal")
            return _outcome(driver, decision.status, decision.reason)

        raise TypeError(f"Unknown Core decision: {type(decision).__name__}")


def _initial_observation(input: CoreRunInput) -> CoreObservation:
    if isinstance(input.entry, ToolResultEntry):
        return UserInputObservation(
            observation_id="core:entry:tool_results",
            text=input.state.task.current_goal,
            current_goal=input.state.task.current_goal,
        )
    if not isinstance(input.entry, ModelEntry):
        raise TypeError(f"Unknown Core entry: {type(input.entry).__name__}")
    if input.entry.message is not None:
        return ModelObservation(
            observation_id="core:entry:persisted_model",
            message=input.entry.message,
        )
    return UserInputObservation(
        observation_id="core:entry:model",
        text=_last_user_text(input.messages) or input.state.task.original_request,
        current_goal=input.state.task.current_goal,
    )


def _reduce(
    driver: _DriverState,
    observation: CoreObservation,
    context: ReductionContext,
) -> CoreReduction:
    reduction = reduce_observation(driver.core, observation, context)
    driver.core = reduction.state
    driver.observation = observation
    driver.events.extend(reduction.events)
    return reduction


async def _settle_unexecuted_calls(
    input: CoreRunInput,
    ports: CorePorts,
    driver: _DriverState,
    calls: tuple[ToolCall, ...],
    context: ReductionContext,
    *,
    reason: str,
) -> None:
    results = interrupted_tool_results(
        calls,
        code="core.tool_not_executed",
        message=f"Core did not execute this Tool call before {reason}.",
    )
    observation = ToolBatchObservation(
        observation_id=driver.observation_id("settlement"),
        calls=calls,
        results=results,
    )
    _reduce(driver, observation, context)
    driver.messages.extend(project_final_tool_messages(results))
    await _commit_boundary(input, ports, driver, "after_tools")


async def _commit_boundary(
    input: CoreRunInput,
    ports: CorePorts,
    driver: _DriverState,
    kind: CoreBoundaryKind,
    *,
    wait: CoreWait | None = None,
) -> None:
    del input
    boundary = CoreBoundary(
        kind=kind,
        state=driver.core,
        new_messages=driver.messages.pending,
        domain_events=driver.events.pending,
        wait=wait,
    )
    committed = ports.boundary.commit(boundary)
    if inspect.isawaitable(committed):
        await committed
    driver.messages.mark_committed()
    driver.events.mark_committed()


def _cancellation_observation(
    input: CoreRunInput,
    ports: CorePorts,
    driver: _DriverState,
) -> CancellationObservation | None:
    if ports.cancellation is None or isinstance(
        driver.observation, CancellationObservation
    ):
        return None
    try:
        ports.cancellation.raise_if_cancelled()
    except asyncio.CancelledError:
        reason = str(getattr(ports.cancellation, "reason", "run.cancelled")).strip()
        return CancellationObservation(
            observation_id=driver.observation_id("cancelled"),
            reason=reason or "run.cancelled",
        )
    return None


def _complete_results(
    calls: tuple[ToolCall, ...],
    results: tuple[ToolResult, ...] | list[ToolResult],
) -> tuple[ToolResult, ...]:
    expected_ids = {call.id for call in calls}
    by_call_id: dict[str, ToolResult] = {}
    for result in results:
        if result.tool_call_id not in expected_ids:
            raise CoreInvariantError(
                f"ToolPort returned a result for unknown ToolCall: {result.tool_call_id}"
            )
        if result.tool_call_id in by_call_id:
            raise CoreInvariantError(
                f"ToolPort returned duplicate results for ToolCall: {result.tool_call_id}"
            )
        by_call_id[result.tool_call_id] = result
    missing = tuple(call for call in calls if call.id not in by_call_id)
    for result in interrupted_tool_results(
        missing,
        code="tool.batch.incomplete",
        message="Tool execution returned no final result for this call.",
    ):
        by_call_id[result.tool_call_id] = result
    return tuple(by_call_id[call.id] for call in calls)


def _unsettled_tool_calls(messages: tuple[Message, ...]) -> tuple[ToolCall, ...]:
    calls: dict[str, ToolCall] = {}
    settled: set[str] = set()
    for message in messages:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolCall):
                    calls[block.id] = block
        elif isinstance(message, ToolResultMessage):
            settled.add(message.tool_call_id)
    return tuple(call for call_id, call in calls.items() if call_id not in settled)


def _has_external_wait(results: tuple[ToolResult, ...]) -> bool:
    return any(
        result.status in {"approval_required", "user_input_required"}
        for result in results
    )


def _last_user_text(messages: tuple[Message, ...]) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, UserMessage):
            continue
        if isinstance(message.content, str):
            text = message.content.strip()
        else:
            text = "".join(
                block.text
                for block in message.content
                if isinstance(block, TextContent)
            ).strip()
        if text:
            return text
    return None


def _last_assistant(messages: tuple[Message, ...]) -> AssistantMessage | None:
    return next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, AssistantMessage)
        ),
        None,
    )


def _outcome(
    driver: _DriverState,
    status: str,
    reason: CoreReason,
    *,
    wait: CoreWait | None = None,
) -> CoreOutcome:
    failure = driver.core.facts.failures.latest
    error = None
    if status in {"failed", "cancelled"} and failure is not None:
        error = {
            "code": failure.code,
            "source": failure.source,
            "message": failure.message,
            "recoverable": failure.recoverable,
            "evidence_refs": list(failure.evidence_refs),
        }
    return CoreOutcome(
        status=status,  # type: ignore[arg-type]
        reason=reason,
        state=driver.core,
        new_messages=driver.messages.added,
        final_message=_last_assistant(driver.messages.messages),
        wait=wait,
        usage=driver.usage,
        error=error,
    )


__all__ = ["run_core"]
