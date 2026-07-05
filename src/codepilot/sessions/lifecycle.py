from __future__ import annotations

import inspect
import logging
from uuid import uuid4
from typing import Any

from codepilot.core.contracts import (
    AgentLoopInput,
    AgentLoopLimits,
    AgentLoopOutcome,
    AgentResumeInput,
    RunCorrelation,
)
from codepilot.core.events import maybe_await
from codepilot.core.types import AgentContext, ContextPreparationRequest
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import (
    AgentRunResult,
    Message,
    ToolResultMessage,
    UserMessage,
)
from codepilot.protocols.commands import SessionLifecycleContext, SessionLifecycleView

from .context.freshness import build_context_freshness_notice
from .history.git_rollback import GitRollbackBaseline
from .history.git_rollback import build_rollback_metadata
from . import command_state
from .contracts import (
    PreparedAgentRun,
    SessionResumeIntent,
    SessionRunIntent,
    SessionRunRecord,
    SessionView,
)


logger = logging.getLogger("codepilot.sessions.lifecycle")


def new_v2_run_id() -> str:
    return f"run_{uuid4().hex[:12]}"


def capture_rollback_baseline_ref(session_id: str, run_id: str) -> dict[str, str]:
    return {
        "kind": "rollback_baseline_ref",
        "session_id": session_id,
        "run_id": run_id,
    }


def describe_runtime_session(session: Any, *, last_run_id: str | None) -> SessionView:
    """Return the controller-facing view for a live session runtime."""

    return SessionView(
        session_id=session.session_id,
        message_count=len(session.conversation.messages),
        last_run_id=last_run_id,
        task_mode=session.task_mode,
        context=runtime_session_state(session),
    )


def runtime_session_state(session: Any) -> dict[str, Any]:
    """Return command/runtime-facing session state without exposing live stores."""

    return {
        "session_id": session.session_id,
        "message_count": len(session.conversation.messages),
        "entry_ids": command_state.list_entry_ids(session),
        "entries": command_state.list_entries(session),
        "tree": command_state.get_session_tree(session),
        "leaf_id": command_state.get_leaf_id(session),
        "task_mode": session.task_mode,
        "planning_budget_profile": session.planning_budget_profile,
    }


def close_runtime_session(session: Any) -> None:
    """Close a live session runtime without exposing its private close hook."""

    session._close()


def _session_messages_for_loop(session: Any) -> list[Message]:
    """Return the current session transcript plus pending steering messages."""

    return [
        *session.conversation.messages,
        *session.conversation.drain_steering_messages(),
    ]


def _loop_context(session: Any) -> dict[str, Any]:
    return {
        "system_prompt": session.conversation.system_prompt,
        "session_id": session.session_id,
    }


def _prompt_hook_text(prepared: PreparedAgentRun) -> str:
    for message in prepared.input_messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
    return ""


async def prepare_runtime_run(
    session: Any,
    intent: SessionRunIntent,
    *,
    run_id: str,
    model: ModelDescriptor,
) -> PreparedAgentRun:
    """Prepare a live session runtime for a new agent loop."""

    rollback_baseline = await begin_run_lifecycle(
        session,
        text=intent.text,
        run_id=run_id,
        is_continue=False,
    )
    session_messages = _session_messages_for_loop(session)
    return PreparedAgentRun(
        run_id=run_id,
        session_id=session.session_id,
        loop_input=AgentLoopInput(
            run_id=run_id,
            correlation=RunCorrelation(session_id=session.session_id),
            messages=session_messages,
            user_prompt=intent.text,
            context=_loop_context(session),
            model=model,
            tools=[],
            task_strategy=runtime_task_strategy(session, mode_hint=intent.mode_hint),
            limits=runtime_loop_limits(session),
            retry_policy=runtime_retry_policy(session),
        ),
        context_port=RuntimeSessionContextPort(session),
        input_messages=[UserMessage(content=intent.text)],
        rollback_baseline=rollback_baseline,
        context_refs={"governor": "session_context_governor"},
        memory_refs={"enabled": session.memory_enabled},
        recovery_refs={"projection": session._active_task_recovery_projection()},
    )


async def prepare_runtime_resume(
    session: Any,
    intent: SessionResumeIntent,
    *,
    run_id: str,
    model: ModelDescriptor,
) -> PreparedAgentRun:
    """Prepare a live session runtime for approval resume."""

    rollback_baseline = await begin_run_lifecycle(
        session,
        text="",
        run_id=run_id,
        is_continue=True,
    )
    session_messages = _session_messages_for_loop(session)
    resume_input = AgentResumeInput(
        run_id=run_id,
        correlation=RunCorrelation(session_id=session.session_id),
        messages=session_messages,
        context=_loop_context(session),
        model=model,
        tools=[],
        approval_id=intent.approval_id,
        decision=intent.decision,
        reason=intent.reason,
        task_strategy=runtime_task_strategy(session),
        retry_policy=runtime_retry_policy(session),
    )
    return PreparedAgentRun(
        run_id=run_id,
        session_id=session.session_id,
        loop_input=AgentLoopInput(
            run_id=run_id,
            correlation=RunCorrelation(session_id=session.session_id),
            messages=session_messages,
            context=_loop_context(session),
            model=model,
            tools=[],
            task_strategy=runtime_task_strategy(session),
            limits=runtime_loop_limits(session),
            retry_policy=runtime_retry_policy(session),
        ),
        resume_input=resume_input,
        context_port=RuntimeSessionContextPort(session),
        rollback_baseline=rollback_baseline,
        recovery_refs={"projection": session._active_task_recovery_projection()},
    )


async def commit_runtime_run(
    session: Any,
    prepared: PreparedAgentRun,
    outcome: AgentLoopOutcome,
    result: AgentRunResult,
    *,
    store_outcome: bool,
) -> SessionRunRecord:
    """Commit a core loop outcome back to a live session runtime."""

    if store_outcome:
        for event in outcome.events:
            session.store.append_event(dict(event))
            await session.conversation.dispatch_event(event)
        committed_messages = [*prepared.input_messages, *outcome.new_messages]
        session.conversation.append_messages(committed_messages)
        for message in committed_messages:
            session.store.append_message(message)
            if session.memory_enabled and isinstance(message, ToolResultMessage):
                observe_tool_memory(
                    session,
                    message,
                    run_id=prepared.run_id,
                )
        session.conversation.remember_result(result)
    committed = await complete_run_lifecycle(
        session,
        result,
        rollback_baseline=prepared.rollback_baseline,
        hook_text=_prompt_hook_text(prepared),
        is_continue=not prepared.input_messages,
    )
    record = session_run_record_from_result(prepared, committed, outcome=outcome)
    session._last_session_run_record = record
    return record


def session_run_record_from_result(
    prepared: PreparedAgentRun,
    result: AgentRunResult,
    *,
    outcome: AgentLoopOutcome,
) -> SessionRunRecord:
    return SessionRunRecord(
        run_id=result.run_id,
        session_id=prepared.session_id,
        status=result.status,
        stop_reason=result.stop_reason,
        new_messages=list(result.messages),
        final_text=outcome.final_text,
        events=list(outcome.events),
        outcome=outcome,
        snapshots={
            "context": prepared.context_refs,
            "memory": prepared.memory_refs,
            "recovery": prepared.recovery_refs,
            "rollback": prepared.rollback_baseline,
        },
    )


def runtime_retry_policy(session: Any) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(session, "retry_enabled", False)),
        "max_retries": int(getattr(session, "max_retries", 0) or 0),
        "base_delay_ms": int(getattr(session, "retry_base_delay_ms", 0) or 0),
    }


def runtime_task_strategy(
    session: Any,
    *,
    mode_hint: str | None = None,
) -> dict[str, Any]:
    return {
        "enabled": bool(getattr(session, "task_control_enabled", False)),
        "mode": mode_hint or getattr(session, "task_mode", "edit"),
        "recovery_projection": session._active_task_recovery_projection(),
        "planning_budget_profile": getattr(
            session,
            "planning_budget_profile",
            "balanced",
        ),
        "max_task_replans_per_run": int(
            getattr(session, "max_task_replans_per_run", 2) or 2
        ),
    }


def runtime_loop_limits(session: Any) -> AgentLoopLimits:
    return AgentLoopLimits(
        max_tool_calls_per_turn=getattr(session, "max_tool_calls_per_turn", None),
    )


class RuntimeSessionContextPort:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._session
        prepared = await maybe_await(
            session.prepare_context(
                AgentContext(
                    system_prompt=str(request.get("system_prompt", "")),
                    messages=list(request.get("messages", ())),
                    tools=list(request.get("tools", ())),
                    task_recovery_projection=session._active_task_recovery_projection(),
                ),
                ContextPreparationRequest(
                    session_id=session.session_id,
                    model_context_window=session.conversation.model.context_window,
                    model_max_output_tokens=session.conversation.model.max_tokens,
                ),
            )
        )
        report = prepared.report.to_dict()
        session.latest_context_report = report
        session.store.append_event(
            {
                "type": "context_prepared",
                "sessionId": session.session_id,
                "report": report,
            }
        )
        memory_ids = report.get("retrieved_memory_ids")
        if session.memory_enabled and isinstance(memory_ids, list) and memory_ids:
            session.store.append_event(
                {
                    "type": "memory_retrieved",
                    "sessionId": session.session_id,
                    "runId": request.get("run_id"),
                    "memoryIds": memory_ids,
                    "reasons": report.get("memory_retrieval_reasons", {}),
                }
            )
        return {
            "system_prompt": prepared.system_prompt,
            "messages": list(prepared.messages),
            "tools": list(prepared.tools),
            "context_report": report,
        }


async def begin_run_lifecycle(
    session: Any,
    *,
    text: str,
    run_id: str,
    is_continue: bool,
    rollback_baseline: GitRollbackBaseline | None = None,
) -> GitRollbackBaseline:
    """Prepare session-owned state before core runs the agent loop."""

    rollback_baseline = rollback_baseline or command_state.capture_run_rollback_baseline(session)
    await run_lifecycle_hooks(
        session,
        text=text,
        is_continue=is_continue,
        hooks=session.before_prompt_hooks,
    )

    if not is_continue:
        if session.memory_enabled:
            admit_prompt_memory(session, text, run_id=run_id)
        begin_task_recovery(session, text, run_id=run_id)

    check_context_freshness(session)
    return rollback_baseline


async def complete_run_lifecycle(
    session: Any,
    result: AgentRunResult,
    *,
    rollback_baseline: GitRollbackBaseline,
    hook_text: str,
    is_continue: bool,
) -> AgentRunResult:
    """Commit session-owned side effects after core returns an outcome."""

    session.store.append_run_result(result)
    write_rollback_metadata(session, result, rollback_baseline)
    finalize_task_recovery(session, result)
    if session.memory_enabled:
        finalize_memory(session, result)
    session.context_governor.finalize_run(result)
    await run_lifecycle_hooks(
        session,
        text=hook_text,
        is_continue=is_continue,
        hooks=session.after_prompt_hooks,
    )
    return result


def write_rollback_metadata(
    session: Any,
    result: AgentRunResult,
    baseline: GitRollbackBaseline,
) -> None:
    session.store.write_rollback_metadata(
        result.run_id,
        build_rollback_metadata(
            baseline,
            affected_paths=list(result.affected_paths),
            workspace_changed=bool(result.workspace_changed),
        ),
    )


def admit_prompt_memory(session: Any, text: str, *, run_id: str | None) -> None:
    """Admit durable project memory from the user prompt when policy allows it."""

    try:
        record = session.memory_writer.admit_prompt_memory(text, run_id=run_id)
        if record is None:
            return
        session.store.append_event(
            {
                "type": "memory_updated",
                "sessionId": session.session_id,
                "memoryId": record.id,
                "kind": record.kind,
            }
        )
    except Exception as exc:
        logger.warning("failed to admit prompt memory: %s", exc)
        session.store.append_event(
            {
                "type": "memory_warning",
                "sessionId": session.session_id,
                "operation": "prompt_memory_admission",
                "message": str(exc),
            }
        )


def begin_task_recovery(session: Any, text: str, *, run_id: str | None) -> None:
    """Persist the current task projection outside durable memory."""

    try:
        projection = session.task_recovery.begin_task(text, run_id=run_id)
        session.store.append_event(
            {
                "type": "task_recovery_updated",
                "sessionId": session.session_id,
                "runId": run_id,
                "goal": projection.get("goal"),
            }
        )
    except Exception as exc:
        logger.warning("failed to write task recovery: %s", exc)
        session.store.append_event(
            {
                "type": "task_recovery_warning",
                "sessionId": session.session_id,
                "operation": "task_recovery_begin",
                "message": str(exc),
            }
        )


def observe_tool_memory(
    session: Any,
    message: ToolResultMessage,
    *,
    run_id: str | None,
) -> None:
    """Let the memory writer observe a tool result without exposing memory stores."""

    try:
        records = session.memory_writer.observe_tool_result(message, run_id=run_id)
        for record in records:
            session.store.append_event(
                {
                    "type": "memory_updated",
                    "sessionId": session.session_id,
                    "memoryId": record.id,
                    "kind": record.kind,
                }
            )
    except Exception as exc:
        logger.warning("failed to observe tool memory: %s", exc)
        session.store.append_event(
            {
                "type": "memory_warning",
                "sessionId": session.session_id,
                "operation": "observe_tool_result",
                "message": str(exc),
            }
        )


def finalize_memory(session: Any, result: AgentRunResult) -> None:
    """Extract durable memory after a run has completed."""

    try:
        records = session.memory_writer.finalize_run(result)
        for record in records:
            session.store.append_event(
                {
                    "type": "memory_updated",
                    "sessionId": session.session_id,
                    "memoryId": record.id,
                    "kind": record.kind,
                }
            )
    except Exception as exc:
        logger.warning("failed to finalize memory: %s", exc)
        session.store.append_event(
            {
                "type": "memory_warning",
                "sessionId": session.session_id,
                "operation": "finalize_run",
                "message": str(exc),
            }
        )


def finalize_task_recovery(session: Any, result: AgentRunResult) -> None:
    """Update session task recovery from the structured run result."""

    try:
        projection = session.task_recovery.update_from_result(result)
        if projection is None:
            return
        session.store.append_event(
            {
                "type": "task_recovery_updated",
                "sessionId": session.session_id,
                "runId": result.run_id,
                "goal": projection.get("goal"),
                "completionSatisfied": (
                    projection.get("task_progress", {}) or {}
                ).get("completion_satisfied")
                if isinstance(projection.get("task_progress"), dict)
                else None,
            }
        )
    except Exception as exc:
        logger.warning("failed to finalize task recovery: %s", exc)
        session.store.append_event(
            {
                "type": "task_recovery_warning",
                "sessionId": session.session_id,
                "operation": "task_recovery_finalize",
                "message": str(exc),
            }
        )


def check_context_freshness(session: Any) -> None:
    """Check whether previous run-tracked context has become stale."""

    freshness = session.store.run_store.evaluate_freshness()
    if not freshness.should_record_event():
        return
    payload = freshness.to_event_payload()
    session.store.append_event(
        {
            "type": "context_freshness_checked",
            "sessionId": session.session_id,
            "freshness": payload,
        }
    )
    if not freshness.requires_steering():
        return
    notice = build_context_freshness_notice(freshness)
    if notice is not None:
        session.conversation.add_steering_message(notice)


async def run_lifecycle_hooks(
    session: Any,
    *,
    text: str,
    is_continue: bool,
    hooks: list,
) -> None:
    """Run prompt lifecycle hooks with a snapshot view of the session."""

    if not hooks:
        return
    ctx = SessionLifecycleContext(
        text=text,
        is_continue=is_continue,
        message_count=len(session.conversation.messages),
        session_view=SessionLifecycleView(
            session_id=session.session_id,
            workspace_dir=str(session.workspace_dir),
            message_count=len(session.conversation.messages),
            task_mode=str(session.task_mode),
        ),
    )
    for hook in hooks:
        value = hook(ctx)
        if inspect.isawaitable(value):
            await value
