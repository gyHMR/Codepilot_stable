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
    from codepilot.runtime.approvals import ApprovalView
    from codepilot.runtime.views import builtin_commands
    from codepilot.runtime.opening import AppSessionView
    from codepilot.runtime.views import SessionStatus
    import codepilot.runtime.views as views
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


def test_core_contracts_describe_loop_stage_without_session_objects() -> None:
    from codepilot.core.contracts import (
        AgentLoopInput,
        AgentLoopLimits,
        AgentLoopOutcome,
        AgentLoopPorts,
        RunCorrelation,
    )
    from codepilot.llm.ports import ModelDescriptor
    from codepilot.protocols import AgentRunCounters, AssistantMessage, TextContent

    final = AssistantMessage(content=[TextContent(text="done")])
    outcome = AgentLoopOutcome(
        run_id="run1",
        status="completed",
        stop_reason="final_answer",
        new_messages=[final],
        final_message=final,
        counters=AgentRunCounters(model_attempts=1),
    )
    assert outcome.status == "completed"
    assert outcome.final_message is final

    loop_input = AgentLoopInput(
        run_id="run1",
        correlation=RunCorrelation(session_id="s1"),
        messages=[],
        context={"system_prompt": "sys"},
        model=ModelDescriptor(provider="fake", model_id="unit"),
        tools=[],
        mode="build",
        plan_state=None,
        limits=AgentLoopLimits(max_model_turns=3),
    )
    assert loop_input.correlation.session_id == "s1"
    assert loop_input.limits.max_model_turns == 3
    assert AgentLoopPorts(model=None, tools=None).events is None


def test_agent_loop_default_tool_iteration_budget_supports_coding_tasks(tmp_path) -> None:
    from codepilot.core.contracts import AgentLoopLimits
    from codepilot.protocols import Model
    from codepilot.sessions.contracts import SessionOptions
    from codepilot.sessions.runtime import SessionRuntime

    defaults = AgentLoopLimits()
    assert defaults.max_model_turns >= 256
    assert defaults.max_tool_iterations >= 200
    assert defaults.max_model_turns > defaults.max_tool_iterations
    assert defaults.max_tool_calls_per_turn == 16
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
    read_session = SessionRuntime(
        SessionOptions(model=model, workspace_dir=tmp_path / "read", current_mode="read")
    )
    plan_session = SessionRuntime(
        SessionOptions(model=model, workspace_dir=tmp_path / "plan", current_mode="plan")
    )
    build_session = SessionRuntime(
        SessionOptions(model=model, workspace_dir=tmp_path / "build", current_mode="build")
    )
    wide_build_session = SessionRuntime(
        SessionOptions(
            model=model,
            workspace_dir=tmp_path / "build-wide",
            current_mode="build",
            planning_budget_profile="wide",
        )
    )

    read_limits = read_session.loop_limits()
    plan_limits = plan_session.loop_limits()
    build_limits = build_session.loop_limits()
    wide_limits = wide_build_session.loop_limits()

    assert build_limits.max_tool_iterations >= 200
    assert build_limits.max_model_turns > build_limits.max_tool_iterations
    assert build_limits.max_tool_calls_per_turn == 16
    assert read_limits.max_tool_iterations < plan_limits.max_tool_iterations
    assert plan_limits.max_tool_iterations < build_limits.max_tool_iterations
    assert wide_limits.max_tool_iterations > build_limits.max_tool_iterations


def test_agent_loop_retry_policy_is_explicit_contract() -> None:
    from typing import get_type_hints

    from codepilot.core.contracts import AgentLoopInput, AgentResumeInput, RetryPolicy

    input_hints = get_type_hints(AgentLoopInput)
    resume_hints = get_type_hints(AgentResumeInput)

    assert input_hints["retry_policy"] is RetryPolicy
    assert resume_hints["retry_policy"] is RetryPolicy
    assert RetryPolicy(enabled=True, max_retries=-1, base_delay_ms=-5).max_retries == 0


def test_agent_loop_mode_and_plan_state_are_explicit_contracts() -> None:
    from typing import get_type_hints

    from codepilot.core.contracts import AgentLoopInput, AgentResumeInput, RunCorrelation

    input_hints = get_type_hints(AgentLoopInput)
    resume_hints = get_type_hints(AgentResumeInput)
    plan_state = {"plan_id": "plan_1", "interpreted_goal": "ship it"}

    assert input_hints["mode"].__args__ == ("read", "plan", "build")
    assert resume_hints["mode"].__args__ == ("read", "plan", "build")
    assert input_hints["plan_state"] == dict[str, object] | None
    assert resume_hints["plan_state"] == dict[str, object] | None

    loop_input = AgentLoopInput(
        run_id="run_plan_contract",
        correlation=RunCorrelation(),
        mode="plan",
        plan_state=plan_state,
    )
    plan_state["interpreted_goal"] = "mutated"
    assert loop_input.mode == "plan"
    assert loop_input.plan_state == {"plan_id": "plan_1", "interpreted_goal": "ship it"}


def test_agent_loop_context_is_named_prepared_context_contract() -> None:
    from typing import get_type_hints

    from codepilot.core import PreparedContext as PublicPreparedContext
    from codepilot.core.contracts import (
        AgentLoopInput,
        AgentResumeInput,
        PreparedContext,
        RunCorrelation,
    )

    input_hints = get_type_hints(AgentLoopInput)
    resume_hints = get_type_hints(AgentResumeInput)
    source = {"system_prompt": "Base rules", "session_id": "s1"}

    loop_input = AgentLoopInput(
        run_id="run_context",
        correlation=RunCorrelation(session_id="s1"),
        context=source,
    )
    resume_input = AgentResumeInput(
        run_id="run_context_resume",
        correlation=RunCorrelation(session_id="s1"),
        context=source,
    )
    source["system_prompt"] = "mutated"

    assert PublicPreparedContext is PreparedContext
    assert input_hints["context"] is PreparedContext
    assert resume_hints["context"] is PreparedContext
    assert loop_input.context.system_prompt == "Base rules"
    assert resume_input.context.session_id == "s1"
    assert not hasattr(resume_input, "tool_call_id")
    assert not hasattr(resume_input, "tool_name")
    assert not hasattr(resume_input, "arguments")
    assert dict(loop_input.context) == {
        "system_prompt": "Base rules",
        "session_id": "s1",
    }
    with pytest.raises(TypeError):
        loop_input.context["new"] = "value"  # type: ignore[index]


def test_agent_loop_ports_use_named_event_sink_contract() -> None:
    from typing import get_type_hints

    from codepilot.core import EventSink as PublicEventSink
    from codepilot.core.contracts import AgentLoopPorts, EventSink

    hints = get_type_hints(AgentLoopPorts)

    assert PublicEventSink is EventSink
    assert hints["events"] == EventSink | None


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


def test_session_run_record_status_follows_core_loop_status_contract() -> None:
    from typing import get_type_hints

    from codepilot.core.contracts import AgentLoopStatus
    from codepilot.sessions.contracts import SessionRunRecord

    hints = get_type_hints(SessionRunRecord)

    assert hints["status"] == AgentLoopStatus


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
        "codepilot.runtime.actions": ("codepilot.core.agent", "codepilot.sessions.runtime", "codepilot.tools.runtime"),
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
