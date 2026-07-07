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
        TaskStrategy,
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
        task_strategy=TaskStrategy(mode="build"),
        limits=AgentLoopLimits(max_model_turns=3),
    )
    assert loop_input.correlation.session_id == "s1"
    assert loop_input.limits.max_model_turns == 3
    assert AgentLoopPorts(model=None, tools=None).events is None


def test_agent_loop_default_tool_iteration_budget_supports_coding_tasks() -> None:
    from types import SimpleNamespace

    from codepilot.core.contracts import AgentLoopLimits
    from codepilot.sessions.runtime import runtime_loop_limits

    assert AgentLoopLimits().max_tool_iterations >= 32
    assert (
        runtime_loop_limits(
            SimpleNamespace(task_mode="build", max_tool_calls_per_turn=8)
        ).max_tool_iterations
        >= 48
    )
    assert (
        runtime_loop_limits(
            SimpleNamespace(task_mode="read", max_tool_calls_per_turn=8)
        ).max_tool_iterations
        < runtime_loop_limits(
            SimpleNamespace(task_mode="build", max_tool_calls_per_turn=8)
        ).max_tool_iterations
    )


def test_agent_loop_retry_policy_is_explicit_contract() -> None:
    from typing import get_type_hints

    from codepilot.core.contracts import AgentLoopInput, AgentResumeInput, RetryPolicy

    input_hints = get_type_hints(AgentLoopInput)
    resume_hints = get_type_hints(AgentResumeInput)

    assert input_hints["retry_policy"] is RetryPolicy
    assert resume_hints["retry_policy"] is RetryPolicy
    assert RetryPolicy(enabled=True, max_retries=-1, base_delay_ms=-5).max_retries == 0


def test_agent_loop_task_strategy_is_explicit_contract() -> None:
    from typing import get_type_hints

    from codepilot.core.contracts import AgentLoopInput, AgentResumeInput, TaskStrategy

    input_hints = get_type_hints(AgentLoopInput)
    resume_hints = get_type_hints(AgentResumeInput)
    strategy = TaskStrategy(
        enabled=True,
        mode="plan",
        goal="  ship it  ",
        steps=[{"title": "Inspect"}],
        max_replans_per_run=-1,
        task_state={"goal": {"value": "ship it"}},
    )

    assert input_hints["task_strategy"] is TaskStrategy
    assert resume_hints["task_strategy"] is TaskStrategy
    assert strategy.goal == "ship it"
    assert strategy.steps == ({"title": "Inspect"},)
    assert strategy.max_replans_per_run is None
    assert strategy.task_state == {"goal": {"value": "ship it"}}


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


def test_tool_port_approval_contract_is_explicit_interruption() -> None:
    from codepilot.tools.ports import (
        ToolInterruption,
        ToolInvocation,
        ToolObservation,
        ToolResumeDecision,
        ToolRiskView,
    )

    interruption = ToolInterruption(
        approval_id="approval1",
        run_id="run1",
        tool_call_id="call1",
        tool_name="shell",
        arguments={"cmd": "git status"},
        reason="requires user approval",
        risk=ToolRiskView(level="medium", summary="workspace write"),
    )
    observation = ToolObservation(
        tool_call_id="call1",
        name="shell",
        status="approval_required",
        interruption=interruption,
    )
    decision = ToolResumeDecision(
        approval_id="approval1",
        decision="approve",
        reason="ok",
    )
    assert observation.interruption is interruption
    assert decision.decision == "approve"
    assert ToolInvocation(run_id="run1", tool_call_id="call1", name="shell").source == "agent"


def test_v2_contract_modules_do_not_import_higher_layers() -> None:
    forbidden = {
        "codepilot.core.contracts": ("codepilot.sessions", "codepilot.runtime", "codepilot.interfaces"),
        "codepilot.sessions.contracts": ("codepilot.runtime", "codepilot.interfaces"),
        "codepilot.runtime.actions": ("codepilot.core.agent", "codepilot.sessions.runtime", "codepilot.tools.engine"),
        "codepilot.llm.ports": ("codepilot.sessions", "codepilot.runtime", "codepilot.interfaces"),
        "codepilot.tools.ports": ("codepilot.sessions", "codepilot.runtime", "codepilot.interfaces"),
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
    import codepilot.tools.ports as tool_ports

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
