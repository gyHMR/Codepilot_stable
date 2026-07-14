from __future__ import annotations

"""Runtime projections derived from the single authoritative CoreOutcome."""

from typing import Literal

from codepilot.core.contracts import CoreOutcome, CoreReason
from codepilot.core.events import CoreDomainEvent
from codepilot.protocols import AgentRunCounters, RunSignalsSummary, RunVerification


RuntimeExecutionState = Literal[
    "new",
    "preparing",
    "executing",
    "waiting",
    "resuming",
    "cancelling",
    "finalizing",
    "terminal",
    "released",
]
TerminalOutcome = Literal["completed", "failed", "cancelled"]


def terminal_outcome_for_status(status: str) -> TerminalOutcome | None:
    normalized = _required_text(status, "status")
    if normalized == "completed":
        return "completed"
    if normalized == "failed":
        return "failed"
    if normalized in {"cancelled", "aborted"}:
        return "cancelled"
    if normalized in {"waiting", "waiting_approval", "waiting_user"}:
        return None
    raise ValueError(f"Unknown execution outcome status: {status}")


def external_status(outcome: CoreOutcome) -> str:
    if outcome.status == "waiting":
        return (
            "waiting_approval"
            if outcome.wait is not None and outcome.wait.kind == "tool_approval"
            else "waiting_user"
        )
    return "aborted" if outcome.status == "cancelled" else outcome.status


def external_stop_reason(reason: CoreReason) -> str:
    """Map structured Core reasons to the stable interface vocabulary."""

    if not isinstance(reason, CoreReason):
        raise TypeError("reason must be CoreReason")
    exact = {
        "task.completed": "final_answer",
        "plan.completed": "final_answer",
        "tool.approval_required": "approval_required",
        "tool.approval_denied": "approval_denied",
        "plan.confirmation_required": "plan_approval_required",
        "plan.revision_confirmation_required": "plan_approval_required",
        "plan.clarification_required": "plan_clarification_required",
        "run.max_model_turns": "max_iterations",
        "run.max_tool_iterations": "max_iterations",
        "run.max_tool_calls": "tool_call_limit",
        "run.max_tool_calls_per_turn": "tool_call_limit",
        "run.repeated_tool_call": "repeated_tool_call",
        "run.cancelled": "aborted",
        "runtime.cancelled": "aborted",
        "runtime.stream_cancelled": "aborted",
        "runtime.deadline_exceeded": "deadline_exceeded",
    }
    if reason.code in exact:
        return exact[reason.code]
    if reason.code in {"tool.unavailable", "tool.not_found"}:
        return "tool_unavailable"
    if reason.code.startswith("model.") or reason.code.startswith("llm."):
        return "model_error"
    if reason.code.startswith("plan.") and "incomplete" in reason.code:
        return "plan_incomplete"
    if reason.source == "runtime":
        return "runtime_error"
    return "completion_blocked"


def project_core_counters(outcome: CoreOutcome) -> AgentRunCounters:
    counters = outcome.state.facts.counters
    return AgentRunCounters(
        model_attempts=int(counters.model_attempts or 0),
        tool_iterations=counters.tool_iterations,
        tool_calls=counters.tool_calls,
    )


def project_core_signals(outcome: CoreOutcome) -> RunSignalsSummary:
    state = outcome.state
    counters = project_core_counters(outcome)
    verification = state.facts.verification
    verification_status = (
        verification.status
        if verification.status in {"passed", "failed", "stale"}
        else "unknown"
    )
    failure = state.facts.failures.latest
    return RunSignalsSummary(
        workspace_changed=state.facts.workspace.changed,
        affected_paths=list(state.facts.workspace.affected_paths),
        verification_status=verification_status,
        last_error=(
            {
                "code": failure.code,
                "message": failure.message,
                "source": failure.source,
                "recoverable": failure.recoverable,
                "evidence_refs": list(failure.evidence_refs),
            }
            if failure is not None
            else None
        ),
        approval_required=(
            outcome.wait is not None and outcome.wait.kind == "tool_approval"
        ),
        tool_unavailable=any(
            blocker.kind == "tool_unavailable" for blocker in state.task.blockers
        ),
        cancelled=outcome.status == "cancelled",
        counters=counters,
    )


def project_core_verification(outcome: CoreOutcome) -> list[RunVerification]:
    verification = outcome.state.facts.verification
    if verification.status in {"none", "unknown", "stale"}:
        return []
    status = (
        verification.status
        if verification.status in {"passed", "failed"}
        else "unknown"
    )
    return [
        RunVerification(
            tool_call_id=(
                verification.evidence_refs[0]
                if verification.evidence_refs
                else "core_verification"
            ),
            tool_name="core",
            status=status,
            command=(
                verification.attempted_checks[0]
                if verification.attempted_checks
                else None
            ),
            summary=verification.unavailable_reason or verification.status,
        )
    ]


def project_core_domain_event(
    event: CoreDomainEvent,
    *,
    event_id: str,
    run_id: str,
    session_id: str,
) -> dict[str, object]:
    """Add the Runtime-owned persistence envelope to one domain event."""

    if not isinstance(event, CoreDomainEvent):
        raise TypeError("event must be CoreDomainEvent")
    payload = dict(event.payload)
    payload.update(
        {
            "event_id": _required_text(event_id, "event_id"),
            "run_id": _required_text(run_id, "run_id"),
            "session_id": _required_text(session_id, "session_id"),
            "type": event.kind,
        }
    )
    if event.evidence_refs:
        payload["evidence_refs"] = list(event.evidence_refs)
    return payload


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


__all__ = [
    "RuntimeExecutionState",
    "TerminalOutcome",
    "external_status",
    "external_stop_reason",
    "project_core_counters",
    "project_core_domain_event",
    "project_core_signals",
    "project_core_verification",
    "terminal_outcome_for_status",
]
