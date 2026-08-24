from __future__ import annotations

import asyncio
from types import SimpleNamespace

from codepilot.core.contracts import CoreBoundary, CoreReason, CoreWait, ToolResultEntry
from codepilot.core.state import CoreState
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import Model
from codepilot.sessions.contracts import (
    SessionContinuationIntent,
    SessionOptions,
    SessionResumeIntent,
    SessionRunIntent,
)
from codepilot.tools.contracts import ToolResumePreparation
from codepilot.tools.results import ToolResult


def _model() -> Model:
    return Model(
        id="unit",
        name="Unit",
        api="unit",
        provider="unit",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=4000,
        max_tokens=500,
    )


def test_approval_resume_persists_prepared_tool_state_before_execution(tmp_path) -> None:
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator

    coordinator = RuntimeSessionCoordinator(
        SessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            session_id="session_tool_resume",
            memory_enabled=False,
        )
    )
    order: list[str] = []

    class _Tools:
        def checkpoint_state(self):
            return None

        def restore_checkpoint_state(self, state):
            order.append("restore")
            assert state == {"waiting": True}

        def approval_challenge(self, approval_id):
            order.append("challenge")
            return SimpleNamespace(
                approval_id=approval_id,
                run_id="run_tool_resume",
                session_id="session_tool_resume",
                request_fingerprint="fingerprint_1",
            )

        def prepare_resume(self, _response):
            order.append("prepare")
            return ToolResumePreparation(
                resume_id="resume_1",
                checkpoint_state={"prepared": "resume_1"},
            )

        async def execute_prepared_resume(self, resume_id):
            run = coordinator.state_service.get_run("run_tool_resume")
            assert run is not None and run.checkpoint is not None
            assert run.checkpoint.waiting is None
            tools = next(
                component
                for component in run.checkpoint.components
                if component.owner == "tools"
            )
            assert tools.state == {"prepared": "resume_1"}
            order.append("execute")
            return ToolResult(
                tool_call_id="call_1",
                tool_name="shell",
                status="success",
                registration_id="registration_1",
            )

    tools = _Tools()

    async def run_case():
        prepared = await coordinator._prepare_run(  # noqa: SLF001
            SessionRunIntent(text="run tool", request_id="request_1"),
            run_id="run_tool_resume",
            model=ModelDescriptor(provider="unit", model_id="unit"),
        )
        assert prepared.state_port is not None
        prepared.state_port.bind_tool_state(lambda: {"waiting": True})
        await prepared.state_port.commit(
            CoreBoundary(
                kind="after_model",
                state=CoreState.new("run tool"),
            )
        )
        await prepared.state_port.commit(
            CoreBoundary(
                kind="waiting",
                state=CoreState.new("run tool"),
                wait=CoreWait(
                    "tool_approval",
                    "approval_1",
                    CoreReason("tool.approval_required", recoverable=True),
                ),
            )
        )
        return await coordinator._prepare_resume(  # noqa: SLF001
            SessionResumeIntent(
                approval_id="approval_1",
                decision="approve",
                run_id="run_tool_resume",
            ),
            run_id="run_tool_resume",
            model=ModelDescriptor(provider="unit", model_id="unit"),
            tools=tools,
        )

    try:
        resumed = asyncio.run(run_case())
    finally:
        coordinator.close()

    assert isinstance(resumed.loop_input.entry, ToolResultEntry)
    assert order == ["restore", "challenge", "prepare", "execute"]


def test_automatic_continuation_executes_persisted_prepared_resume(tmp_path) -> None:
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator

    session_id = "session_prepared_resume_recovery"
    run_id = "run_prepared_resume_recovery"
    first = RuntimeSessionCoordinator(
        SessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            session_id=session_id,
            memory_enabled=False,
        )
    )

    class _CrashingTools:
        def restore_checkpoint_state(self, _state):
            return None

        def approval_challenge(self, approval_id):
            return SimpleNamespace(
                approval_id=approval_id,
                run_id=run_id,
                session_id=session_id,
                request_fingerprint="fingerprint_1",
            )

        def prepare_resume(self, _response):
            return ToolResumePreparation(
                resume_id="resume_1",
                checkpoint_state={"prepared_resume": "resume_1"},
            )

        async def execute_prepared_resume(self, _resume_id):
            raise RuntimeError("process stopped before Tool execution")

    async def persist_then_stop():
        prepared = await first._prepare_run(  # noqa: SLF001
            SessionRunIntent(text="run tool", request_id="request_1"),
            run_id=run_id,
            model=ModelDescriptor(provider="unit", model_id="unit"),
        )
        assert prepared.state_port is not None
        prepared.state_port.bind_tool_state(lambda: {"waiting": True})
        await prepared.state_port.commit(
            CoreBoundary(kind="after_model", state=CoreState.new("run tool"))
        )
        await prepared.state_port.commit(
            CoreBoundary(
                kind="waiting",
                state=CoreState.new("run tool"),
                wait=CoreWait(
                    "tool_approval",
                    "approval_1",
                    CoreReason("tool.approval_required", recoverable=True),
                ),
            )
        )
        await first._prepare_resume(  # noqa: SLF001
            SessionResumeIntent(
                approval_id="approval_1",
                decision="approve",
                run_id=run_id,
            ),
            run_id=run_id,
            model=ModelDescriptor(provider="unit", model_id="unit"),
            tools=_CrashingTools(),
        )

    try:
        try:
            asyncio.run(persist_then_stop())
        except RuntimeError as exc:
            assert str(exc) == "process stopped before Tool execution"
        persisted = first.state_service.get_run(run_id)
        assert persisted is not None and persisted.checkpoint is not None
        assert persisted.checkpoint.waiting is None
    finally:
        first.close()

    second = RuntimeSessionCoordinator(
        SessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            session_id=session_id,
            memory_enabled=False,
        )
    )

    class _RecoveredTools:
        def __init__(self) -> None:
            self.resume_id = ""

        def restore_checkpoint_state(self, state):
            self.resume_id = str(state["prepared_resume"])

        def pending_prepared_resume(self):
            return ToolResumePreparation(
                resume_id=self.resume_id,
                checkpoint_state={"prepared_resume": self.resume_id},
            )

        async def execute_prepared_resume(self, resume_id):
            assert resume_id == "resume_1"
            return ToolResult(
                tool_call_id="call_1",
                tool_name="shell",
                status="success",
                registration_id="registration_1",
            )

    async def recover():
        return await second._prepare_continuation(  # noqa: SLF001
            SessionContinuationIntent(
                kind="automatic_continuation",
                run_id=run_id,
            ),
            run_id=run_id,
            model=ModelDescriptor(provider="unit", model_id="unit"),
            tools=_RecoveredTools(),
        )

    try:
        resumed = asyncio.run(recover())
    finally:
        second.close()

    assert isinstance(resumed.loop_input.entry, ToolResultEntry)
    assert resumed.loop_input.entry.results[0].status == "success"
