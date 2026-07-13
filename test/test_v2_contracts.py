from __future__ import annotations

import dataclasses
import importlib
from typing import get_args

import pytest


def test_runtime_actions_are_immutable_user_intents_and_frames() -> None:
    from codepilot.runtime.actions import (
        ApprovalDecided,
        ApprovalRequiredFrame,
        CancelledFrame,
        CommandFinishedFrame,
        CommandSubmitted,
        FailedFrame,
        ProgressFrame,
        PromptSubmitted,
        RunCancelled,
        RunFinishedFrame,
        UserAction,
    )

    prompt = PromptSubmitted(text="  hello  ", images=[" img.png "], mode_hint=" plan ")
    assert prompt.text == "hello"
    assert prompt.images == ("img.png",)
    assert prompt.mode_hint == "plan"
    with pytest.raises(dataclasses.FrozenInstanceError):
        prompt.text = "mutate"  # type: ignore[misc]

    assert isinstance(CommandSubmitted(text="/status"), get_args(UserAction))
    assert ApprovalDecided(approval_id=" a1 ", decision="APPROVE").decision == "approve"
    assert RunCancelled(reason=" user ").reason == "user"
    assert ProgressFrame(event={"type": "agent_start"}).kind == "progress"
    assert ApprovalRequiredFrame(approval={"approval_id": "a1"}).kind == "approval_required"
    assert RunFinishedFrame(record={"run_id": "run1"}).kind == "run_finished"
    assert CommandFinishedFrame(record={"handled": True}).kind == "command_finished"
    assert CancelledFrame(session_id="s1").kind == "cancelled"
    assert FailedFrame(error={"code": "x"}).kind == "failed"


def test_runtime_describe_view_is_app_session_view_not_legacy_snapshot() -> None:
    from codepilot.runtime.actions import ApprovalView, AppSessionView, SessionStatus
    from codepilot.runtime.commands import builtin_commands
    import codepilot.runtime.actions as views
    from codepilot.sessions.contracts import SessionView

    view = AppSessionView(
        session=SessionView(session_id="s1", message_count=1),
        status=SessionStatus(
            session_id="s1",
            model_id="fake/model",
            workspace=".",
            permission_mode="workspace-write",
            message_count=1,
            leaf_id="entry_1",
        ),
        state={"leaf_id": "entry_1"},
        commands=tuple(builtin_commands()),
        pending_approvals=(
            ApprovalView(
                approval_id="a1",
                session_id="s1",
                run_id="r1",
                tool_call_id="call1",
                tool_name="shell",
            ),
        ),
    )

    assert view.session.session_id == "s1"
    assert view.status.model_id == "fake/model"
    assert view.commands[0].name
    assert view.pending_approvals[0].approval_id == "a1"
    assert not hasattr(views, "SessionSnapshot")
    with pytest.raises(TypeError):
        view.state["leaf_id"] = "mutate"  # type: ignore[index]


def test_core_contracts_describe_one_run_vocabulary() -> None:
    from typing import get_type_hints

    from codepilot.core.contracts import CoreRunInput

    hints = get_type_hints(CoreRunInput)

    assert "entry" in hints
    assert "state" in hints
    assert "mode" in hints
    assert "retry_policy" not in hints
    assert "deadline_at_ms" not in hints
    assert "approval_id" not in hints
    assert "decision" not in hints


def test_core_limits_support_mode_specific_coding_budgets(tmp_path) -> None:
    from codepilot.protocols import Model
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
    from codepilot.sessions.contracts import SessionOptions

    model = Model(
        id="unit",
        name="Unit",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=4000,
        max_tokens=500,
    )
    read_session = RuntimeSessionCoordinator(
        SessionOptions(model=model, workspace_dir=tmp_path / "read", current_mode="read")
    )
    plan_session = RuntimeSessionCoordinator(
        SessionOptions(model=model, workspace_dir=tmp_path / "plan", current_mode="plan")
    )
    build_session = RuntimeSessionCoordinator(
        SessionOptions(model=model, workspace_dir=tmp_path / "build", current_mode="build")
    )
    wide_build_session = RuntimeSessionCoordinator(
        SessionOptions(
            model=model,
            workspace_dir=tmp_path / "build-wide",
            current_mode="build",
            planning_budget_profile="wide",
        )
    )
    try:
        read_limits = read_session.core_limits()
        plan_limits = plan_session.core_limits()
        build_limits = build_session.core_limits()
        wide_limits = wide_build_session.core_limits()
    finally:
        read_session.close()
        plan_session.close()
        build_session.close()
        wide_build_session.close()

    assert build_limits.max_tool_iterations >= 200
    assert build_limits.max_model_turns > build_limits.max_tool_iterations
    assert build_limits.max_tool_calls_per_turn == 16
    assert read_limits.max_tool_iterations < plan_limits.max_tool_iterations
    assert plan_limits.max_tool_iterations < build_limits.max_tool_iterations
    assert wide_limits.max_tool_iterations > build_limits.max_tool_iterations


def test_prepared_run_has_one_core_input() -> None:
    from typing import get_type_hints

    from codepilot.core.contracts import CoreRunInput
    from codepilot.sessions.contracts import PreparedAgentRun

    hints = get_type_hints(PreparedAgentRun)

    assert hints["loop_input"] is CoreRunInput
    assert "resume_input" not in hints


def test_core_ports_use_named_live_event_sink_contract() -> None:
    from typing import get_type_hints

    from codepilot.core.contracts import CorePorts, LiveEventSink

    hints = get_type_hints(CorePorts)

    assert hints["live_events"] == LiveEventSink | None


def test_prepared_agent_run_context_port_uses_core_context_port_contract() -> None:
    from typing import get_type_hints

    from codepilot.core.contracts import ContextPort
    from codepilot.sessions.contracts import PreparedAgentRun

    hints = get_type_hints(PreparedAgentRun)

    assert hints["context_port"] == ContextPort | None


def test_prepared_agent_run_rollback_baseline_is_public_ref() -> None:
    from typing import get_type_hints

    from codepilot.sessions.contracts import PreparedAgentRun, RollbackBaselineRef

    hints = get_type_hints(PreparedAgentRun)
    ref = RollbackBaselineRef(session_id="session1", run_id="run1")

    assert hints["rollback_baseline"] == RollbackBaselineRef | None
    assert ref.kind == "rollback_baseline_ref"


def test_session_run_record_status_uses_public_run_status_contract() -> None:
    from typing import get_type_hints

    from codepilot.protocols import AgentRunStatus
    from codepilot.sessions.contracts import SessionRunRecord

    hints = get_type_hints(SessionRunRecord)

    assert hints["status"] == AgentRunStatus


def test_session_intents_normalize_resume_and_cancel_values() -> None:
    from codepilot.sessions.contracts import CancelRunIntent, SessionResumeIntent

    resume = SessionResumeIntent(
        approval_id=" approval1 ",
        decision="APPROVE",
        reason=" ok ",
        run_id=" run1 ",
    )
    cancel = CancelRunIntent(run_id=" run2 ", reason=" user ")

    assert resume.approval_id == "approval1"
    assert resume.decision == "approve"
    assert resume.reason == "ok"
    assert resume.run_id == "run1"
    assert cancel.run_id == "run2"
    assert cancel.reason == "user"
    with pytest.raises(ValueError):
        SessionResumeIntent(approval_id="approval1", decision="maybe")


def test_tool_port_approval_contract_uses_typed_challenge_and_response() -> None:
    from codepilot.tools.results import ToolResult
    from codepilot.tools.security import ApprovalChallenge, ApprovalResponse, ToolResource

    challenge = ApprovalChallenge(
        approval_id="approval1",
        request_fingerprint="sha256:test",
        run_id="run1",
        session_id="session1",
        tool_call_id="call1",
        tool_name="shell",
        registration_id="reg1",
        actions=("shell",),
        resources=(ToolResource("workspace:///"),),
        effects=frozenset({"process_spawn"}),
        risk="medium",
        reason="requires user approval",
        safe_preview={"command": "git status"},
    )
    result = ToolResult(
        tool_call_id="call1",
        tool_name="shell",
        status="approval_required",
        approval=challenge,
        registration_id="reg1",
    )
    response = ApprovalResponse(
        approval_id="approval1",
        request_fingerprint="sha256:test",
        decision="approve",
        reason="ok",
    )
    assert result.approval is challenge
    assert response.decision == "approve"


def test_v2_contract_modules_do_not_import_higher_layers() -> None:
    forbidden = {
        "codepilot.core.contracts": ("codepilot.sessions", "codepilot.runtime", "codepilot.interfaces"),
        "codepilot.sessions.contracts": ("codepilot.runtime", "codepilot.interfaces"),
        "codepilot.runtime.actions": ("codepilot.core.agent", "codepilot.runtime.session_coordinator", "codepilot.tools.runtime"),
        "codepilot.llm.ports": ("codepilot.sessions", "codepilot.runtime", "codepilot.interfaces"),
        "codepilot.tools.contracts": ("codepilot.sessions", "codepilot.runtime", "codepilot.interfaces"),
    }
    for module_name, forbidden_imports in forbidden.items():
        module = importlib.import_module(module_name)
        names = set(getattr(module, "__dict__", {}))
        imported_modules = {
            value.__name__
            for value in module.__dict__.values()
            if hasattr(value, "__name__") and hasattr(value, "__package__")
        }
        haystack = names | imported_modules
        assert not any(
            item == forbidden_name or item.startswith(f"{forbidden_name}.")
            for item in haystack
            for forbidden_name in forbidden_imports
        )


def test_v2_contract_modules_export_only_named_contract_surface() -> None:
    import codepilot.core.contracts as core_contracts
    import codepilot.llm.ports as llm_ports
    import codepilot.runtime.actions as runtime_actions
    import codepilot.sessions.contracts as session_contracts
    import codepilot.tools.contracts as tool_ports

    modules = (
        runtime_actions,
        session_contracts,
        core_contracts,
        llm_ports,
        tool_ports,
    )

    for module in modules:
        exported = set(module.__all__)
        assert exported
        assert all(hasattr(module, name) for name in exported)
        assert not any(name.startswith("_") for name in exported)
        assert "_require_text" not in exported
        assert "_validate_capabilities" not in exported
