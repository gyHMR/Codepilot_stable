from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from codepilot.runtime.contracts import (
    CommitReceipt,
    RunCommitIdentity,
    terminal_outcome_for_status,
)
from codepilot.runtime.environment import RunEnvironment, RunResourceScope
from codepilot.runtime.executor import RunExecutionCompleted, RunExecutor
from codepilot.runtime.lifecycle import RuntimeLifecycle
from codepilot.core.contracts import AgentLoopOutcome


def _environment() -> RunEnvironment:
    resources = RunResourceScope()
    return RunEnvironment(
        run_id="run_1",
        session_id="session_1",
        trigger="prompt",
        model=None,
        tools=None,
        context=None,
        state=None,
        cancellation=resources.cancellation,
        deadline_at_ms=None,
        event_sink=None,
        resources=resources,
    )


async def _collect_execution(executor, environment, prepared):
    updates = [update async for update in executor.execute(environment, prepared)]
    completed = [update for update in updates if isinstance(update, RunExecutionCompleted)]
    assert len(completed) == 1
    return completed[0].outcome


def test_runtime_lifecycle_accepts_normal_and_resume_paths() -> None:
    lifecycle = RuntimeLifecycle("run_1")
    lifecycle.transition("preparing")
    lifecycle.transition("executing")
    lifecycle.transition("waiting")

    with pytest.raises(ValueError, match="waiting -> executing"):
        lifecycle.transition("executing")

    lifecycle.transition("resuming")
    lifecycle.transition("executing")
    lifecycle.transition("finalizing")
    lifecycle.transition("terminal", terminal_outcome="completed")
    lifecycle.mark_released()
    lifecycle.mark_released()

    assert lifecycle.state == "released"
    assert lifecycle.terminal_outcome == "completed"


def test_runtime_lifecycle_requires_terminal_commit_semantics() -> None:
    lifecycle = RuntimeLifecycle("run_1")
    lifecycle.transition("preparing")
    lifecycle.transition("executing")
    lifecycle.transition("finalizing")

    with pytest.raises(ValueError, match="requires terminal_outcome"):
        lifecycle.transition("terminal")
    with pytest.raises(ValueError, match="finalizing -> released"):
        lifecycle.mark_released()

    lifecycle.transition("terminal", terminal_outcome="failed")
    with pytest.raises(ValueError, match="terminal -> finalizing"):
        lifecycle.transition("finalizing")


def test_runtime_lifecycle_uses_single_cancellation_path() -> None:
    lifecycle = RuntimeLifecycle("run_1")
    lifecycle.transition("preparing")
    lifecycle.transition("executing")
    lifecycle.transition("cancelling")
    lifecycle.transition("finalizing")
    lifecycle.transition("terminal", terminal_outcome="cancelled")

    assert lifecycle.terminal_outcome == "cancelled"


def test_commit_identity_and_receipt_freeze_idempotency_key() -> None:
    identity = RunCommitIdentity("commit_1", expected_revision=4, kind="waiting")
    same_retry = RunCommitIdentity("commit_1", expected_revision=4, kind="waiting")
    receipt = CommitReceipt("commit_1", revision=5, kind="waiting")

    assert identity == same_retry
    assert receipt.matches(same_retry)
    assert not receipt.matches(RunCommitIdentity("commit_1", 4, "terminal"))

    with pytest.raises(ValueError, match="cannot be negative"):
        RunCommitIdentity("commit_2", expected_revision=-1, kind="progress")


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("waiting_approval", None),
        ("waiting_user", None),
    ],
)
def test_terminal_outcome_mapping_has_one_runtime_vocabulary(status, expected) -> None:
    assert terminal_outcome_for_status(status) == expected


def test_run_executor_normalizes_task_cancellation(monkeypatch) -> None:
    async def cancelled(_input, _ports):
        raise asyncio.CancelledError

    monkeypatch.setattr("codepilot.runtime.executor.run_agent_loop", cancelled)
    environment = _environment()
    prepared = SimpleNamespace(loop_input=object())

    outcome = asyncio.run(_collect_execution(RunExecutor(), environment, prepared))

    assert outcome.status == "cancelled"
    assert outcome.stop_reason == "cancelled"
    assert outcome.signals.cancelled is True


def test_run_executor_maps_deadline_cancellation_to_failed_timeout(monkeypatch) -> None:
    async def cancelled(_input, _ports):
        raise asyncio.CancelledError

    monkeypatch.setattr("codepilot.runtime.executor.run_agent_loop", cancelled)
    environment = _environment()
    environment.resources.cancel("deadline_exceeded")
    prepared = SimpleNamespace(loop_input=object())

    outcome = asyncio.run(_collect_execution(RunExecutor(), environment, prepared))

    assert outcome.status == "failed"
    assert outcome.stop_reason == "deadline_exceeded"


def test_run_executor_uses_same_core_entry_for_resume(monkeypatch) -> None:
    received = []

    async def execute(input_value, _ports):
        received.append(input_value)
        return AgentLoopOutcome(
            run_id="run_1",
            status="completed",
            stop_reason="final_answer",
        )

    monkeypatch.setattr("codepilot.runtime.executor.run_agent_loop", execute)
    loop_input = SimpleNamespace(entry="resume")
    environment = _environment()

    outcome = asyncio.run(
        _collect_execution(RunExecutor(), environment, SimpleNamespace(loop_input=loop_input))
    )

    assert received == [loop_input]
    assert outcome.status == "completed"
