from __future__ import annotations

# 新手导读：commit.py 负责把 core outcome 写回 session，并完成记忆、任务恢复和 rollback 元数据。
# 关注点：这里是 run 后副作用的唯一入口，prepare.py 不负责收尾写回。

import logging
from typing import Any

from codepilot.core.contracts import AgentLoopOutcome
from codepilot.protocols import AgentRunResult

from .contracts import PreparedAgentRun, RollbackBaselineRef, SessionRunRecord
from .history.git_rollback import GitRollbackBaseline, build_rollback_metadata
from .prepare import run_lifecycle_hooks

logger = logging.getLogger("codepilot.sessions.commit")


def _prompt_hook_text(prepared: PreparedAgentRun) -> str:
    for message in prepared.input_messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
    return ""

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
        session.conversation.remember_result(result)
    committed = await complete_run_lifecycle(
        session,
        result,
        rollback_baseline=_take_rollback_baseline(session, prepared.rollback_baseline),
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

def _take_rollback_baseline(
    session: Any,
    ref: RollbackBaselineRef | None,
) -> GitRollbackBaseline:
    if ref is None:
        return GitRollbackBaseline(
            eligible=False,
            reason="missing_rollback_baseline_ref",
        )
    if ref.session_id != session.session_id:
        return GitRollbackBaseline(
            eligible=False,
            reason="rollback_baseline_session_mismatch",
        )
    baselines = getattr(session, "_rollback_baselines", None)
    if not isinstance(baselines, dict):
        return GitRollbackBaseline(
            eligible=False,
            reason="missing_rollback_baseline_store",
        )
    baseline = baselines.pop(ref.run_id, None)
    if baseline is None:
        return GitRollbackBaseline(
            eligible=False,
            reason="missing_rollback_baseline",
        )
    return baseline

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
                    "kind": record.type,
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
                "completionSatisfied": _task_state_completion_satisfied(projection),
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


def _task_state_completion_satisfied(projection: dict[str, object]) -> bool | None:
    steps = projection.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    return all(
        isinstance(step, dict) and step.get("status") == "completed"
        for step in steps
    )


__all__ = [
    "commit_runtime_run",
    "complete_run_lifecycle",
    "finalize_memory",
    "finalize_task_recovery",
    "session_run_record_from_result",
    "write_rollback_metadata",
]
