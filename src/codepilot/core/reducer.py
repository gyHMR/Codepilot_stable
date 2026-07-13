from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import cast

from codepilot.protocols import TextContent, ToolCall
from codepilot.tools.results import ToolResult, workspace_effect_summary

from .commands import (
    AbandonPlan,
    ApprovePlan,
    ApprovePlanRevision,
    CommandResult,
    CoreCommand,
    ProposePlanRevision,
    RejectPlan,
    RejectPlanRevision,
    ReportVerificationUnavailable,
    RequestPlanClose,
    SubmitPlan,
    UpdatePlanProgress,
)
from .errors import CoreInvariantError
from .events import CoreDomainEvent
from .observations import (
    CancellationObservation,
    CoreCommandObservation,
    CoreObservation,
    ModelObservation,
    ToolBatchObservation,
    UserInputObservation,
)
from .plan import (
    QUALIFIED_FAILURES_FOR_REVISION,
    PendingPlanRevision,
    PlanCloseRequest,
    PlanState,
    PlanStep,
    PlanStepDefinition,
    RunMode,
    ensure_run_mode,
)
from .state import (
    CoreState,
    FailureCount,
    FailureFacts,
    FailureRecord,
    LoopGuardFacts,
    ObservationLedger,
    TaskBlocker,
    TaskStatus,
    VerificationFactStatus,
    VerificationFacts,
    WorkspaceFacts,
)


@dataclass(frozen=True)
class ReductionContext:
    run_id: str
    mode: RunMode
    now_ms: int

    def __post_init__(self) -> None:
        run_id = str(self.run_id).strip()
        if not run_id:
            raise ValueError("run_id is required")
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "mode", ensure_run_mode(self.mode))
        if (
            not isinstance(self.now_ms, int)
            or isinstance(self.now_ms, bool)
            or self.now_ms < 0
        ):
            raise ValueError("now_ms must be a non-negative integer")


@dataclass(frozen=True)
class CoreReduction:
    state: CoreState
    events: tuple[CoreDomainEvent, ...] = ()
    command_results: tuple[CommandResult, ...] = ()


def reduce_observation(
    state: CoreState,
    observation: CoreObservation,
    context: ReductionContext,
) -> CoreReduction:
    observation_id = observation.observation_id
    if observation_id in state.facts.observation_ledger.applied_observation_ids:
        return CoreReduction(state=state)

    if isinstance(observation, ModelObservation):
        reduction = _reduce_model(state, observation)
    elif isinstance(observation, ToolBatchObservation):
        reduction = _reduce_tools(state, observation, context)
    elif isinstance(observation, UserInputObservation):
        reduction = _reduce_user_input(state, observation)
    elif isinstance(observation, CoreCommandObservation):
        reduction = apply_core_command(state, observation.command, context)
    elif isinstance(observation, CancellationObservation):
        reduction = _reduce_cancellation(state, observation)
    else:  # pragma: no cover - protected by the closed union
        raise CoreInvariantError(
            f"Unknown Core observation: {type(observation).__name__}"
        )

    ledger = ObservationLedger(
        applied_observation_ids=(
            *reduction.state.facts.observation_ledger.applied_observation_ids,
            observation_id,
        )
    )
    facts = replace(reduction.state.facts, observation_ledger=ledger)
    return replace(reduction, state=replace(reduction.state, facts=facts))


def apply_decision(
    state: CoreState,
    decision: object,
    context: ReductionContext,
) -> CoreReduction:
    del context
    from .contracts import CallModel, ExecuteTools, Terminate, Wait

    if isinstance(decision, CallModel):
        if decision.purpose != "recovery":
            return CoreReduction(state=state)
        counters = replace(
            state.facts.counters,
            total_recoveries=state.facts.counters.total_recoveries + 1,
        )
        return CoreReduction(
            state=replace(state, facts=replace(state.facts, counters=counters)),
            events=(
                CoreDomainEvent("recovery_requested", {"reason": decision.reason.code}),
            ),
        )
    if isinstance(decision, (ExecuteTools, Wait)):
        return CoreReduction(state=state)
    if not isinstance(decision, Terminate):
        raise CoreInvariantError(f"Unknown Core decision: {type(decision).__name__}")

    status_map = {
        "completed": "satisfied",
        "failed": "blocked",
        "cancelled": "abandoned",
    }
    event_map = {
        "completed": "task_satisfied",
        "failed": "task_blocked",
        "cancelled": "task_abandoned",
    }
    task_status = status_map[decision.status]
    plan = state.task.plan
    if (
        decision.status == "completed"
        and plan is not None
        and plan.status == "active"
        and plan.close_request is not None
        and all(step.status == "completed" for step in plan.steps)
    ):
        plan = replace(plan, status="completed", revision=plan.revision + 1)
    elif (
        decision.status in {"failed", "cancelled"}
        and plan is not None
        and plan.status in {"proposed", "active"}
    ):
        plan = replace(
            plan,
            status="abandoned",
            pending_revision=None,
            close_request=None,
            revision=plan.revision + 1,
        )
    task = replace(
        state.task,
        status=cast(TaskStatus, task_status),
        plan=plan,
        blockers=() if decision.status == "completed" else state.task.blockers,
    )
    return CoreReduction(
        state=replace(state, task=task),
        events=(
            CoreDomainEvent(
                event_map[decision.status], {"reason": decision.reason.code}
            ),
        ),
    )


def _reduce_model(state: CoreState, observation: ModelObservation) -> CoreReduction:
    counters = replace(
        state.facts.counters,
        model_turns=state.facts.counters.model_turns + 1,
    )
    next_state = replace(state, facts=replace(state.facts, counters=counters))
    events: list[CoreDomainEvent] = [CoreDomainEvent("model_observed")]
    failure = observation.error
    if observation.status == "completed" and _model_message_is_empty(observation):
        failure = FailureRecord(
            code="model.empty_response",
            source="model",
            message="Model returned no text or tool calls",
            recoverable=True,
            evidence_refs=(observation.observation_id,),
        )
    if failure is not None:
        next_state = _record_failure(next_state, failure)
        events.append(_failure_event(failure))
    return CoreReduction(state=next_state, events=tuple(events))


def _reduce_tools(
    state: CoreState,
    observation: ToolBatchObservation,
    context: ReductionContext,
) -> CoreReduction:
    counted_call_ids = set(state.facts.loop_guards.seen_tool_call_ids)
    batch_call_ids: set[str] = set()
    unique_results: list[ToolResult] = []
    for result in observation.results:
        if result.tool_call_id in batch_call_ids:
            continue
        batch_call_ids.add(result.tool_call_id)
        unique_results.append(result)
    results = tuple(unique_results)
    new_call_count = sum(
        result.tool_call_id not in counted_call_ids for result in results
    )
    counters = replace(
        state.facts.counters,
        tool_iterations=state.facts.counters.tool_iterations + 1,
        tool_calls=state.facts.counters.tool_calls + new_call_count,
    )
    next_state = replace(state, facts=replace(state.facts, counters=counters))
    events: list[CoreDomainEvent] = []

    workspace, workspace_changed = _reduce_workspace(
        next_state.facts.workspace, results
    )
    next_state = replace(
        next_state, facts=replace(next_state.facts, workspace=workspace)
    )
    if workspace_changed:
        events.append(
            CoreDomainEvent(
                "workspace_changed",
                {
                    "revision": workspace.revision,
                    "affected_paths": list(workspace.affected_paths),
                },
                workspace.evidence_refs,
            )
        )

    verification = _reduce_verification(
        next_state.facts.verification,
        results,
        workspace=workspace,
        workspace_changed=workspace_changed,
    )
    next_state = replace(
        next_state, facts=replace(next_state.facts, verification=verification)
    )
    if verification != state.facts.verification:
        events.append(
            CoreDomainEvent(
                "verification_recorded",
                {
                    "status": verification.status,
                    "revision": verification.verified_revision,
                },
                verification.evidence_refs,
            )
        )
    next_state = _sync_verification_blocker(next_state)

    for result in results:
        failure = _failure_from_tool_result(result)
        if failure is not None:
            next_state = _record_failure(next_state, failure)
            events.append(_failure_event(failure))
            if result.error is not None and (
                result.error.kind == "unavailable"
                or result.error.code == "tool_not_found"
            ):
                next_state = _put_blocker(
                    next_state,
                    TaskBlocker(
                        kind="tool_unavailable",
                        reason=result.error.message,
                        evidence_refs=(result.tool_call_id,),
                        recoverable=failure.recoverable,
                    ),
                )

    command_results: list[CommandResult] = []
    for command in observation.commands:
        command_reduction = apply_core_command(next_state, command, context)
        next_state = command_reduction.state
        events.extend(command_reduction.events)
        command_results.extend(command_reduction.command_results)

    loop_guards = _reduce_loop_guards(
        next_state.facts.loop_guards, observation.calls, results
    )
    next_state = replace(
        next_state, facts=replace(next_state.facts, loop_guards=loop_guards)
    )
    return CoreReduction(
        state=next_state,
        events=tuple(events),
        command_results=tuple(command_results),
    )


def _reduce_user_input(
    state: CoreState, observation: UserInputObservation
) -> CoreReduction:
    blockers = tuple(
        item for item in state.task.blockers if item.kind != "user_input_required"
    )
    task = replace(
        state.task,
        current_goal=observation.current_goal or state.task.current_goal,
        status="active" if state.task.status == "blocked" else state.task.status,
        blockers=blockers,
    )
    return CoreReduction(
        state=replace(state, task=task),
        events=(CoreDomainEvent("user_input_received"),),
    )


def apply_core_command(
    state: CoreState,
    command: CoreCommand,
    context: ReductionContext,
) -> CoreReduction:
    if command.command_id in state.facts.observation_ledger.applied_observation_ids:
        return CoreReduction(state=state)

    reduction = _reduce_command(state, command, context)
    return replace(reduction, state=_mark_command_seen(reduction.state, command.command_id))


def _reduce_command(
    state: CoreState,
    command: CoreCommand,
    context: ReductionContext,
) -> CoreReduction:
    if isinstance(command, SubmitPlan):
        return _submit_plan(state, command, context)
    if isinstance(command, UpdatePlanProgress):
        return _update_plan_progress(state, command, context)
    if isinstance(command, ProposePlanRevision):
        return _propose_plan_revision(state, command, context)
    if isinstance(command, RequestPlanClose):
        return _request_plan_close(state, command, context)
    if isinstance(command, ApprovePlan):
        return _approve_plan(state, command)
    if isinstance(command, RejectPlan):
        return _reject_plan(state, command)
    if isinstance(command, ApprovePlanRevision):
        return _approve_plan_revision(state, command)
    if isinstance(command, RejectPlanRevision):
        return _reject_plan_revision(state, command)
    if isinstance(command, AbandonPlan):
        return _abandon_plan(state, command)
    if not isinstance(command, ReportVerificationUnavailable):
        raise CoreInvariantError(f"Unknown Core command: {type(command).__name__}")

    if not command.attempted_checks or not command.evidence_refs:
        return _reject_command(state, command, "verification_evidence_required")
    verification = VerificationFacts(
        status="unavailable",
        verified_revision=state.facts.workspace.revision,
        attempted_checks=command.attempted_checks,
        evidence_refs=command.evidence_refs,
        unavailable_reason=command.reason,
    )
    next_state = replace(state, facts=replace(state.facts, verification=verification))
    next_state = _remove_blocker(next_state, "verification_failed")
    return CoreReduction(
        state=next_state,
        events=(
            CoreDomainEvent(
                "verification_recorded",
                {"status": "unavailable", "revision": verification.verified_revision},
                verification.evidence_refs,
            ),
        ),
        command_results=(CommandResult(command.command_id, "applied"),),
    )


def _submit_plan(
    state: CoreState,
    command: SubmitPlan,
    context: ReductionContext,
) -> CoreReduction:
    if context.mode == "read":
        return _reject_command(state, command, "plan.read_only")
    if context.mode == "plan":
        try:
            command.definition.require_plan_mode_details()
        except ValueError:
            return _reject_command(state, command, "plan.definition_incomplete")
    current = state.task.plan
    if current is not None:
        if context.mode != "plan" or current.status != "proposed":
            return _reject_command(state, command, "plan.already_exists")
        replacement = tuple(
            _new_step(current.plan_id, definition, index)
            for index, definition in enumerate(command.steps, start=1)
        )
        plan = replace(
            current,
            definition=command.definition,
            steps=_preserve_matching_steps(current.steps, replacement),
            revision=current.revision + 1,
            pending_revision=None,
            close_request=None,
        )
        next_state = replace(state, task=replace(state.task, plan=plan))
        return _applied_plan_command(
            next_state,
            command,
            "plan_proposal_revised",
            {"plan_id": plan.plan_id, "revision": plan.revision},
        )
    plan_id = f"plan:{command.command_id}"
    plan = PlanState(
        plan_id=plan_id,
        origin="plan_mode" if context.mode == "plan" else "build_mode",
        status="proposed" if context.mode == "plan" else "active",
        revision=1,
        definition=command.definition,
        steps=tuple(
            _new_step(plan_id, definition, index)
            for index, definition in enumerate(command.steps, start=1)
        ),
    )
    next_state = replace(state, task=replace(state.task, plan=plan))
    return _applied_plan_command(
        next_state,
        command,
        "plan_submitted",
        {"plan_id": plan.plan_id, "status": plan.status, "revision": plan.revision},
    )


def _update_plan_progress(
    state: CoreState,
    command: UpdatePlanProgress,
    context: ReductionContext,
) -> CoreReduction:
    problem = _active_plan_problem(state, command.expected_revision, context)
    if problem is not None:
        return _reject_command(state, command, problem)
    plan = state.task.plan
    assert plan is not None
    by_id = {step.step_id: step for step in plan.steps}
    if any(update.step_id not in by_id for update in command.updates):
        return _reject_command(state, command, "plan.unknown_step")
    known_evidence = _known_evidence(state)
    next_by_id = dict(by_id)
    for update in command.updates:
        previous = by_id[update.step_id]
        if previous.status == "completed" and update.status != "completed":
            return _reject_command(state, command, "plan.completed_step_regression")
        if update.status == "completed" and not update.completion_note:
            return _reject_command(state, command, "plan.completion_note_required")
        if set(update.evidence_refs) - known_evidence:
            return _reject_command(state, command, "plan.unknown_evidence")
        try:
            next_by_id[update.step_id] = PlanStep(
                step_id=previous.step_id,
                step=previous.step,
                details=previous.details,
                verification=previous.verification,
                status=update.status,
                completion_note=(
                    update.completion_note if update.status == "completed" else ""
                ),
                evidence_refs=(
                    update.evidence_refs if update.status == "completed" else ()
                ),
            )
        except ValueError:
            return _reject_command(state, command, "plan.invalid_progress")
    steps = tuple(next_by_id[step.step_id] for step in plan.steps)
    if sum(step.status == "in_progress" for step in steps) > 1:
        return _reject_command(state, command, "plan.multiple_in_progress")
    next_plan = replace(
        plan,
        steps=steps,
        revision=plan.revision + 1,
        close_request=None,
    )
    next_state = replace(state, task=replace(state.task, plan=next_plan))
    return _applied_plan_command(
        next_state,
        command,
        "plan_progress_updated",
        {"plan_id": plan.plan_id, "revision": next_plan.revision},
    )


def _propose_plan_revision(
    state: CoreState,
    command: ProposePlanRevision,
    context: ReductionContext,
) -> CoreReduction:
    problem = _active_plan_problem(state, command.expected_revision, context)
    if problem is not None:
        return _reject_command(state, command, problem)
    plan = state.task.plan
    assert plan is not None
    if plan.pending_revision is not None:
        return _reject_command(state, command, "plan.revision_pending")
    replacement = tuple(
        _new_step(
            f"{plan.plan_id}:revision:{plan.revision + 1}",
            definition,
            index,
        )
        for index, definition in enumerate(command.steps, start=1)
    )
    merged = _preserve_matching_steps(plan.steps, replacement)
    auto_apply = (
        plan.origin == "build_mode"
        and command.reason == "repeated_execution_failure"
        and max((item.count for item in state.facts.failures.counts), default=0)
        >= QUALIFIED_FAILURES_FOR_REVISION
    )
    if auto_apply:
        next_plan = replace(
            plan,
            definition=command.definition,
            steps=merged,
            revision=plan.revision + 1,
            pending_revision=None,
            close_request=None,
        )
        event_type = "plan_revision_applied"
    else:
        pending = PendingPlanRevision(
            reason=command.reason,
            definition=command.definition,
            steps=merged,
            proposed_at_revision=plan.revision,
        )
        next_plan = replace(
            plan,
            pending_revision=pending,
            revision=plan.revision + 1,
        )
        event_type = "plan_revision_proposed"
    next_state = replace(state, task=replace(state.task, plan=next_plan))
    return _applied_plan_command(
        next_state,
        command,
        event_type,
        {"plan_id": plan.plan_id, "revision": next_plan.revision},
    )


def _request_plan_close(
    state: CoreState,
    command: RequestPlanClose,
    context: ReductionContext,
) -> CoreReduction:
    problem = _active_plan_problem(state, command.expected_revision, context)
    if problem is not None:
        return _reject_command(state, command, problem)
    if set(command.evidence_refs) - _known_evidence(state):
        return _reject_command(state, command, "plan.unknown_evidence")
    plan = state.task.plan
    assert plan is not None
    request = PlanCloseRequest(
        summary=command.summary,
        evidence_refs=command.evidence_refs,
        requested_at_revision=plan.revision,
    )
    next_plan = replace(
        plan,
        close_request=request,
        revision=plan.revision + 1,
    )
    next_state = replace(state, task=replace(state.task, plan=next_plan))
    return _applied_plan_command(
        next_state,
        command,
        "plan_close_requested",
        {"plan_id": plan.plan_id, "revision": next_plan.revision},
    )


def _approve_plan(state: CoreState, command: ApprovePlan) -> CoreReduction:
    plan = state.task.plan
    problem = _plan_revision_problem(plan, command.expected_revision)
    if problem is not None:
        return _reject_command(state, command, problem)
    assert plan is not None
    if plan.status != "proposed":
        return _reject_command(state, command, "plan.not_proposed")
    next_plan = replace(plan, status="active", revision=plan.revision + 1)
    return _applied_plan_command(
        replace(state, task=replace(state.task, plan=next_plan)),
        command,
        "plan_approved",
        {"plan_id": plan.plan_id, "revision": next_plan.revision},
    )


def _reject_plan(state: CoreState, command: RejectPlan) -> CoreReduction:
    plan = state.task.plan
    problem = _plan_revision_problem(plan, command.expected_revision)
    if problem is not None:
        return _reject_command(state, command, problem)
    assert plan is not None
    if plan.status != "proposed":
        return _reject_command(state, command, "plan.not_proposed")
    next_plan = replace(plan, status="rejected", revision=plan.revision + 1)
    return _applied_plan_command(
        replace(state, task=replace(state.task, plan=next_plan)),
        command,
        "plan_rejected",
        {"plan_id": plan.plan_id, "reason": command.reason},
    )


def _approve_plan_revision(
    state: CoreState,
    command: ApprovePlanRevision,
) -> CoreReduction:
    plan = state.task.plan
    problem = _plan_revision_problem(plan, command.expected_revision)
    if problem is not None:
        return _reject_command(state, command, problem)
    assert plan is not None
    pending = plan.pending_revision
    if pending is None:
        return _reject_command(state, command, "plan.no_pending_revision")
    next_plan = replace(
        plan,
        definition=pending.definition,
        steps=_preserve_matching_steps(plan.steps, pending.steps),
        pending_revision=None,
        close_request=None,
        revision=plan.revision + 1,
    )
    return _applied_plan_command(
        replace(state, task=replace(state.task, plan=next_plan)),
        command,
        "plan_revision_approved",
        {"plan_id": plan.plan_id, "revision": next_plan.revision},
    )


def _reject_plan_revision(
    state: CoreState,
    command: RejectPlanRevision,
) -> CoreReduction:
    plan = state.task.plan
    problem = _plan_revision_problem(plan, command.expected_revision)
    if problem is not None:
        return _reject_command(state, command, problem)
    assert plan is not None
    if plan.pending_revision is None:
        return _reject_command(state, command, "plan.no_pending_revision")
    next_plan = replace(
        plan,
        pending_revision=None,
        revision=plan.revision + 1,
    )
    return _applied_plan_command(
        replace(state, task=replace(state.task, plan=next_plan)),
        command,
        "plan_revision_rejected",
        {"plan_id": plan.plan_id, "reason": command.reason},
    )


def _abandon_plan(state: CoreState, command: AbandonPlan) -> CoreReduction:
    plan = state.task.plan
    problem = _plan_revision_problem(plan, command.expected_revision)
    if problem is not None:
        return _reject_command(state, command, problem)
    assert plan is not None
    if plan.status not in {"proposed", "active"}:
        return _reject_command(state, command, "plan.not_abandonable")
    next_plan = replace(
        plan,
        status="abandoned",
        pending_revision=None,
        close_request=None,
        revision=plan.revision + 1,
    )
    return _applied_plan_command(
        replace(state, task=replace(state.task, plan=next_plan)),
        command,
        "plan_abandoned",
        {"plan_id": plan.plan_id, "reason": command.reason},
    )


def _active_plan_problem(
    state: CoreState,
    expected_revision: int,
    context: ReductionContext,
) -> str | None:
    if context.mode == "read":
        return "plan.read_only"
    if context.mode != "build":
        return "plan.build_mode_required"
    plan = state.task.plan
    problem = _plan_revision_problem(plan, expected_revision)
    if problem is not None:
        return problem
    assert plan is not None
    if plan.status != "active":
        return "plan.not_active"
    return None


def _plan_revision_problem(
    plan: PlanState | None,
    expected_revision: int,
) -> str | None:
    if plan is None:
        return "plan.missing"
    if plan.revision != expected_revision:
        return "plan.revision_conflict"
    return None


def _new_step(
    plan_id: str,
    definition: PlanStepDefinition,
    index: int,
) -> PlanStep:
    return PlanStep(
        step_id=f"{plan_id}:step:{index}",
        step=definition.step,
        details=definition.details,
        verification=definition.verification,
    )


def _preserve_matching_steps(
    previous: tuple[PlanStep, ...],
    replacement: tuple[PlanStep, ...],
) -> tuple[PlanStep, ...]:
    by_definition = {
        (step.step, step.details, step.verification): step for step in previous
    }
    return tuple(
        by_definition.get(
            (step.step, step.details, step.verification),
            step,
        )
        for step in replacement
    )


def _known_evidence(state: CoreState) -> set[str]:
    refs = set(state.facts.observation_ledger.applied_observation_ids)
    refs.update(state.facts.loop_guards.seen_tool_call_ids)
    refs.update(state.facts.workspace.evidence_refs)
    refs.update(state.facts.verification.evidence_refs)
    if state.facts.failures.latest is not None:
        refs.update(state.facts.failures.latest.evidence_refs)
    return refs


def _applied_plan_command(
    state: CoreState,
    command: CoreCommand,
    event_type: str,
    payload: dict[str, object],
) -> CoreReduction:
    return CoreReduction(
        state=state,
        events=(CoreDomainEvent(event_type, payload),),
        command_results=(CommandResult(command.command_id, "applied"),),
    )


def _reject_command(
    state: CoreState,
    command: CoreCommand,
    reason: str,
) -> CoreReduction:
    return CoreReduction(
        state=state,
        events=(
            CoreDomainEvent(
                "command_rejected",
                {"command_id": command.command_id, "reason": reason},
            ),
        ),
        command_results=(CommandResult(command.command_id, "rejected", reason),),
    )


def _reduce_cancellation(
    state: CoreState,
    observation: CancellationObservation,
) -> CoreReduction:
    failure = FailureRecord(
        code="run.cancelled",
        source="runtime",
        message=observation.reason,
        recoverable=False,
        evidence_refs=(observation.observation_id,),
    )
    next_state = _record_failure(state, failure)
    return CoreReduction(
        state=next_state,
        events=(
            CoreDomainEvent("cancellation_observed", {"reason": observation.reason}),
        ),
    )


def _reduce_workspace(
    current: WorkspaceFacts,
    results: tuple[ToolResult, ...],
) -> tuple[WorkspaceFacts, bool]:
    paths = set(current.affected_paths)
    evidence = list(current.evidence_refs)
    changed = False
    for result in results:
        uris, result_changed = workspace_effect_summary(result.effects)
        if result_changed:
            changed = True
            evidence.append(result.tool_call_id)
        paths.update(uri.removeprefix("workspace:///") or "." for uri in uris)
    if not changed:
        return current, False
    return (
        WorkspaceFacts(
            revision=current.revision + 1,
            changed=True,
            affected_paths=tuple(paths),
            evidence_refs=tuple(evidence),
        ),
        True,
    )


def _reduce_verification(
    current: VerificationFacts,
    results: tuple[ToolResult, ...],
    *,
    workspace: WorkspaceFacts,
    workspace_changed: bool,
) -> VerificationFacts:
    next_value = (
        replace(current, status="stale", verified_revision=None)
        if workspace_changed and current.status not in {"none", "unknown"}
        else current
    )
    for result in results:
        value = result.data.get("verification")
        if not hasattr(value, "get"):
            continue
        raw_status = str(value.get("status") or "unknown").strip().lower()
        status: VerificationFactStatus = cast(
            VerificationFactStatus,
            raw_status if raw_status in {"passed", "failed", "unknown"} else "unknown",
        )
        explicit_revision = value.get("verified_revision")
        verified_revision = (
            explicit_revision
            if isinstance(explicit_revision, int)
            and not isinstance(explicit_revision, bool)
            else workspace.revision
        )
        if (
            workspace_changed
            and status == "passed"
            and explicit_revision != workspace.revision
        ):
            status = "stale"
            verified_revision = None
        elif (
            workspace_changed
            and status == "failed"
            and explicit_revision != workspace.revision
        ):
            verified_revision = None
        command = str(value.get("command") or "").strip()
        next_value = VerificationFacts(
            status=status,
            verified_revision=verified_revision,
            attempted_checks=(command,) if command else (),
            evidence_refs=(result.tool_call_id,),
        )
    return next_value


def _sync_verification_blocker(state: CoreState) -> CoreState:
    verification = state.facts.verification
    if verification.status == "failed":
        failure = FailureRecord(
            code="verification.failed",
            source="verification",
            message="Verification failed",
            recoverable=True,
            evidence_refs=verification.evidence_refs,
        )
        state = _record_failure(state, failure)
        return _put_blocker(
            state,
            TaskBlocker(
                kind="verification_failed",
                reason="Verification failed",
                evidence_refs=verification.evidence_refs,
                recoverable=True,
            ),
        )
    if verification.status in {"passed", "unavailable"}:
        return _remove_blocker(state, "verification_failed")
    return state


def _failure_from_tool_result(result: ToolResult) -> FailureRecord | None:
    if result.error is None:
        return None
    recoverable = result.error.retryable or result.error.kind in {
        "validation",
        "unavailable",
        "permission",
        "approval",
        "execution",
        "output_validation",
    }
    return FailureRecord(
        code=result.error.code,
        source="tools",
        message=result.error.message,
        recoverable=recoverable,
        evidence_refs=(result.tool_call_id,),
    )


def _record_failure(state: CoreState, failure: FailureRecord) -> CoreState:
    counts = {item.code: item.count for item in state.facts.failures.counts}
    counts[failure.code] = counts.get(failure.code, 0) + 1
    failures = FailureFacts(
        latest=failure,
        counts=tuple(FailureCount(code, count) for code, count in counts.items()),
    )
    return replace(state, facts=replace(state.facts, failures=failures))


def _mark_command_seen(state: CoreState, command_id: str) -> CoreState:
    ledger = ObservationLedger(
        applied_observation_ids=(
            *state.facts.observation_ledger.applied_observation_ids,
            command_id,
        )
    )
    return replace(state, facts=replace(state.facts, observation_ledger=ledger))


def _put_blocker(state: CoreState, blocker: TaskBlocker) -> CoreState:
    blockers = tuple(item for item in state.task.blockers if item.kind != blocker.kind)
    return replace(state, task=replace(state.task, blockers=(*blockers, blocker)))


def _remove_blocker(state: CoreState, kind: str) -> CoreState:
    blockers = tuple(item for item in state.task.blockers if item.kind != kind)
    return replace(state, task=replace(state.task, blockers=blockers))


def _reduce_loop_guards(
    current: LoopGuardFacts,
    calls: tuple[ToolCall, ...],
    results: tuple[ToolResult, ...],
) -> LoopGuardFacts:
    fingerprint = _tool_fingerprint(calls) if calls else current.last_tool_fingerprint
    repeated = current.repeated_tool_calls
    if calls:
        repeated = repeated + 1 if fingerprint == current.last_tool_fingerprint else 1
    seen = (*current.seen_tool_call_ids, *(result.tool_call_id for result in results))
    return LoopGuardFacts(
        last_tool_fingerprint=fingerprint,
        repeated_tool_calls=repeated,
        repeated_no_progress=current.repeated_no_progress,
        seen_tool_call_ids=seen,
    )


def _tool_fingerprint(calls: tuple[ToolCall, ...]) -> str:
    return json.dumps(
        [(call.name, call.arguments) for call in calls],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def _model_message_is_empty(observation: ModelObservation) -> bool:
    if observation.message is None:
        return True
    return not any(
        isinstance(block, ToolCall)
        or (isinstance(block, TextContent) and bool(block.text.strip()))
        for block in observation.message.content
    )


def _failure_event(failure: FailureRecord) -> CoreDomainEvent:
    return CoreDomainEvent(
        "failure_recorded",
        {
            "code": failure.code,
            "source": failure.source,
            "recoverable": failure.recoverable,
        },
        failure.evidence_refs,
    )


__all__ = [
    "CoreReduction",
    "ReductionContext",
    "apply_core_command",
    "apply_decision",
    "reduce_observation",
]
